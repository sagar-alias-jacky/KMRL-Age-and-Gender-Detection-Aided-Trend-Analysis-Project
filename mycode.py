import io
import logging
import pickle
import jsonpickle
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from model.model import ResMLP
from utils import enable_dropout, forward_mc, read_json
from insightface.app.face_analysis import FaceAnalysis as FaceDetectionRecognition
from utils import resize_square_image, get_original_bbox, get_original_lm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

fdr = FaceDetectionRecognition(det_name='retinaface_r50_v1',
                               rec_name='arcface_r100_v1',
                               ga_name=None)


def my_face_detection_recognition(data) -> None:
    """Receive everything in json!!!
    """
    # app.logger.debug(f"Receiving data ...")
    # data = request.json
    data = jsonpickle.decode(data)

    # app.logger.debug(f"decompressing image ...")
    image = data['image']
    image = io.BytesIO(image)

    # app.logger.debug(f"Reading a PIL image ...")
    image = Image.open(image)
    image_size_original = image.size

    # app.logger.debug(f"Resizing a PIL image to 640 by 640 ...")
    image = resize_square_image(image, 640, background_color=(0, 0, 0))
    image_size_new = image.size

    # app.logger.debug(f"Conveting a PIL image to a numpy array ...")
    image = np.array(image)

    if len(image.shape) != 3:
        # app.logger.error(f"image shape: {image.shape} is not RGB!")
        del data, image
        response = {'face_detection_recognition': None}
        response_pickled = jsonpickle.encode(response)
        return response_pickled

    # app.logger.info(f"extraing features ...")
    list_of_features = fdr.get(image)
    # app.logger.info(f"features extracted!")

    # app.logger.info(f"In total of {len(list_of_features)} faces detected!")

    results_frame = []
    for features in list_of_features:
        bbox = get_original_bbox(
            features.bbox, image_size_original, image_size_new)
        landmark = get_original_lm(
            features.landmark, image_size_original, image_size_new)
        feature_dict = {'bbox': bbox,
                        'det_score': features.det_score,
                        'landmark': landmark,
                        'normed_embedding': features.normed_embedding
                        }
        results_frame.append(feature_dict)

    response = {'face_detection_recognition': results_frame}
    # app.logger.info("json-pickle is done.")

    response_pickled = jsonpickle.encode(response)

    return response_pickled


models = {"age": None, "gender": None}

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

for model_ in ["age", "gender"]:
    model = ResMLP(**read_json(f"./models/{model_}.json")["arch"]["args"])
    checkpoint = f"models/{model_}.pth"
    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    enable_dropout(model)

    models[model_] = model


def predict_age_gender(data):
    """Receive everything in json!!!"""
    # app.logger.debug(f"Receiving data ...")
    # data = request.json
    data = jsonpickle.decode(data)

    # app.logger.debug(f"loading embeddings ...")
    embeddings = data["embeddings"]
    embeddings = io.BytesIO(embeddings)

    # This assumes that the client has serialized the embeddings with pickle.
    # before sending it to the server.
    embeddings = np.load(embeddings, allow_pickle=True)

    # -1 accounts for the batch size.
    embeddings = embeddings.reshape(-1, 512).astype(np.float32)

    # app.logger.debug(
    # f"extracting gender and age from {embeddings.shape[0]} faces ...")

    genders = []
    ages = []

    for embedding in tqdm(embeddings):
        embedding = embedding.reshape(1, 512)
        gender_mean, gender_entropy = forward_mc(models["gender"], embedding)
        age_mean, age_entropy = forward_mc(models["age"], embedding)
        gender = {"m": 1 - gender_mean,
                  "f": gender_mean, "entropy": gender_entropy}
        age = {"mean": age_mean, "entropy": age_entropy}

        genders.append(gender)
        ages.append(age)

    # app.logger.debug(f"gender and age extracted!")

    response = {"ages": ages, "genders": genders}

    response_pickled = jsonpickle.encode(response)
    # app.logger.info("json-pickle is done.")

    return response_pickled


def send_to_servers(binary_image) -> None:
    """Send a binary image to the two servers.

    Args
    ----
    binary_image: binary image
    url_face: url of the face-detection-recognition server
    url_age_gender: url of the age-gender server.

    Returns
    -------
    genders, ages, bboxes, det_scores, landmarks, embeddings

    """
    data = {"image": binary_image}
    logging.info(f"image loaded!")

    logging.debug(f"sending image to server...")
    data = jsonpickle.encode(data)
    # TODO implement face redognition here instead of Flask server setup
    # response = requests.post(url_face, json=data)
    response = my_face_detection_recognition(data)
    logging.info(f"got {response} from server!...")
    response = jsonpickle.decode(response.text)

    face_detection_recognition = response["face_detection_recognition"]
    logging.info(f"{len(face_detection_recognition)} faces deteced!")

    bboxes = [fdr["bbox"] for fdr in face_detection_recognition]
    det_scores = [fdr["detjpg_score"] for fdr in face_detection_recognition]
    landmarks = [fdr["landmark"] for fdr in face_detection_recognition]
    embeddings = [fdr["normed_embedding"]
                  for fdr in face_detection_recognition]

    # -1 accounts for the batch size.
    data = np.array(embeddings).reshape(-1, 512).astype(np.float32)
    data = pickle.dumps(data)
    data = {"embeddings": data}
    data = jsonpickle.encode(data)

    logging.debug(f"sending embeddings to server ...")
    # TODO implement age gender here instead of Flask server setup
    # response = requests.post(url_age_gender, json=data)
    response = predict_age_gender(data)
    logging.info(f"got {response} from server!...")

    response = jsonpickle.decode(response.text)
    ages = response["ages"]
    genders = response["genders"]

    return genders, ages, bboxes, det_scores, landmarks, embeddings


def annotate_image(image: Image.Image, genders: list, ages: list, bboxes: list) -> None:
    """Annotate a given image. This is done in-place. Nothing is returned.

    Args
    ----
    image: Pillow image
    genders, ages, bboxes

    """
    logging.debug(f"annotating image ...")

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("fonts/arial.ttf", 25)

    for gender, age, bbox in zip(genders, ages, bboxes):
        draw.rectangle(bbox.tolist(), outline=(0, 0, 0))
        draw.text(
            (bbox[0], bbox[1]),
            f"AGE: {round(age['mean'])}, ENTROPY: {round(age['entropy'], 4)}",
            fill=(255, 0, 0),
            font=font,
        )
        draw.text(
            (bbox[0], bbox[3]),
            "MALE " + str(round(gender["m"] * 100)) + str("%") + ", "
            "FEMALE "
            + str(round(gender["f"] * 100))
            + str("%")
            + f", ENTROPY: {round(gender['entropy'], 4)}",
            fill=(0, 255, 0),
            font=font,
        )


def save_annotated_image(
    image: Image.Image,
    save_path: str,
    bboxes: list,
    det_scores: list,
    landmarks: list,
    embeddings: list,
    genders: list,
    ages: list,
) -> None:
    """Save the annotated image.

    Args
    ----
    image: Pilow image
    bboxes:
    det_scores:
    landmarks:
    embeddings:
    genders:
    ages:

    """
    image.save(save_path)
    logging.info(f"image annotated and saved at {save_path}")

    to_dump = {
        "bboxes": bboxes,
        "det_scores": det_scores,
        "landmarks": landmarks,
        "embeddings": embeddings,
        "genders": genders,
        "ages": ages,
    }

    with open(save_path + ".pkl", "wb") as stream:
        pickle.dump(to_dump, stream)
    logging.info(f"features saved at at {save_path + '.pkl'}")


def run_image(image_path: str):
    """Run age-gender on the image.

    Args
    ----
    url_face: url of the face-detection-recognition server
    url_age_gender: url of the age-gender server.
    image_path

    """
    logging.debug(f"loading image ...")
    with open(image_path, "rb") as stream:
        binary_image = stream.read()

    genders, ages, bboxes, det_scores, landmarks, embeddings = send_to_servers(
        binary_image
    )

    image = Image.open(image_path)

    annotate_image(image, genders, ages, bboxes)

    save_path = image_path + ".ANNOTATED.jpg"

    save_annotated_image(
        image, save_path, bboxes, det_scores, landmarks, embeddings, genders, ages
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Extract face, gender, and age.")
    # parser.add_argument("--url-face", type=str,
    #                     default="http://127.0.0.1:10002/")
    # parser.add_argument("--url-age-gender", type=str,
    #                     default="http://127.0.0.1:10003/")
    # parser.add_argument("--image-path", type=str, default=None)
    # parser.add_argument("--camera-id", type=int,
    #                     default="0", help="ffplay /dev/video0")
    # parser.add_argument("--mode", type=str, default="image",
    #                     help="image or webcam")

    # args = vars(parser.parse_args())

    # logging.info(f"arguments given to {__file__}: {args}")

    # mode = args.pop("mode")
    # if mode == "image":
    #     assert args["image_path"] is not None
    #     del args["camera_id"]
    #     run_image(**args)
    # else:
    #     del args["image_path"]
    #     run_webcam(**args)

    run_image('my_images/group/test2.jpg')
