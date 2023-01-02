"""
This is just a simple client example. Hack it as much as you want.
"""
import argparse
import io
import logging
import pickle
import shutil
import cv2
import os
import jsonpickle
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
import csv
from natsort import natsorted
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

gender_graph_dict = {}
age_graph_dict = {'1-2': 0, '3-9': 0, '10-20': 0,
                  '21-27': 0, '28-45': 0, '46-65': 0, 'Above 65': 0}


def initialize_csv():
    # path = 'output.csv'
    # isExist = os.path.exists(path)
    # print(isExist)
    # if isExist == False:
    header = ['Gender', 'Age']
    with open('output.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)


def select_age_group(age):
    if 1 <= age <= 2:
        return '1-2'
    elif 3 <= age <= 9:
        return '3-9'
    elif 10 <= age <= 20:
        return '10-20'
    elif 21 <= age <= 27:
        return '21-27'
    elif 28 <= age <= 45:
        return '28-45'
    elif 46 <= age <= 65:
        return '46-65'
    else:
        return 'Above 65'


def extract_data_into_csv(genders, ages):
    for index in range(0, len(genders)):

        data = []

        if genders[index]['m'] > genders[index]['f']:
            gender = 'Male'
        else:
            gender = 'Female'

        data.append(gender)

        age = round(ages[index]['mean'])

        age = select_age_group(age)

        data.append(age)
        print(data)

        if data[0] not in gender_graph_dict.keys():
            gender_graph_dict[data[0]] = 1
        else:
            gender_graph_dict[data[0]] += 1

        if data[1] not in age_graph_dict.keys():
            age_graph_dict[data[1]] = 1
        else:
            age_graph_dict[data[1]] += 1

        with open('output.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(data)


def video_to_frames(path):
    directory = 'video_frames'
    os.makedirs(directory, exist_ok=True)
    shutil.rmtree('video_frames')
    os.makedirs(directory, exist_ok=True)
    capture = cv2.VideoCapture(path)
    frameNr = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(fps)
    gap = round(fps)
    print(gap)

    while (True):
        success, frame = capture.read()

        if success == False:
            capture.release()
            break

        if frameNr == 0 or frameNr % gap == 0:
            cv2.imwrite(f'video_frames/frame_{frameNr}.jpg', frame)

        frameNr += 1


def send_to_servers(binary_image, url_face: str, url_age_gender: str) -> None:
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
    response = requests.post(url_face, json=data)
    print(response)
    print(response.text)
    logging.info(f"got {response} from server!...")
    response = jsonpickle.decode(response.text)

    face_detection_recognition = response["face_detection_recognition"]
    logging.info(f"{len(face_detection_recognition)} faces deteced!")

    bboxes = [fdr["bbox"] for fdr in face_detection_recognition]
    det_scores = [fdr["det_score"] for fdr in face_detection_recognition]
    landmarks = [fdr["landmark"] for fdr in face_detection_recognition]
    embeddings = [fdr["normed_embedding"]
                  for fdr in face_detection_recognition]

    # -1 accounts for the batch size.
    data = np.array(embeddings).reshape(-1, 512).astype(np.float32)
    data = pickle.dumps(data)
    data = {"embeddings": data}
    data = jsonpickle.encode(data)

    logging.debug(f"sending embeddings to server ...")
    response = requests.post(url_age_gender, json=data)
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

    # print(to_dump)

    with open(save_path + ".pkl", "wb") as stream:
        pickle.dump(to_dump, stream)
    logging.info(f"features saved at at {save_path + '.pkl'}")


def run_image(url_face: str, url_age_gender: str, image_path: str):
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
        binary_image, url_face, url_age_gender
    )

    print(genders)
    print(ages)

    extract_data_into_csv(genders, ages)

    image = Image.open(image_path)

    annotate_image(image, genders, ages, bboxes)

    save_path = image_path + ".ANNOTATED.jpg"

    save_annotated_image(
        image, save_path, bboxes, det_scores, landmarks, embeddings, genders, ages
    )


def annotate_fps(image: Image.Image, fps: int) -> None:
    """Annotate fps on a given image.

    Args
    ----
    image: Pillow image
    fps: frames per second

    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("fonts/arial.ttf", 25)
    draw.text((0, 0), f"FPS: {fps} (Press q  to exit.)",
              fill=(0, 0, 255), font=font)


def run_webcam(url_face: str, url_age_gender: str, camera_id: int):

    import time

    import cv2

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # fps = []
    while True:
        start_time = time.time()  # start time of the loop
        # Capture frame-by-frame
        ret, image_BGR = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

        image_PIL = Image.fromarray(image_RGB)
        binary_image = io.BytesIO()
        image_PIL.save(binary_image, format="JPEG")
        binary_image = binary_image.getvalue()

        genders, ages, bboxes, det_scores, landmarks, embeddings = send_to_servers(
            binary_image, url_face, url_age_gender
        )

        annotate_image(image_PIL, genders, ages, bboxes)

        # fps.append(time)
        fps = int(1.0 / (time.time() - start_time))

        annotate_fps(image_PIL, fps)

        cv2.imshow("frame", cv2.cvtColor(
            np.array(image_PIL), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def print_age_graph():
    # Print Age Graph
    x = age_graph_dict.keys()

    y = []
    for key in age_graph_dict.keys():
        y.append(age_graph_dict[key])

    # plotting the points
    plt.plot(x, y)
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    # giving a title to my graph
    plt.title('Age Trend Graph')
    # function to show the plot
    plt.savefig('age_trend_graph')
    plt.close()
    print(x)
    print(y)


def print_gender_graph():
    # Print Age Graph
    x = gender_graph_dict.keys()

    y = []
    for key in gender_graph_dict.keys():
        y.append(gender_graph_dict[key])

    # plotting the points
    plt.plot(x, y)
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    # giving a title to my graph
    plt.title('Gender Trend Graph')
    # function to show the plot
    plt.savefig('gender_trend_graph')
    plt.close()
    print(x)
    print(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract face, gender, and age.")
    parser.add_argument("--url-face", type=str,
                        default="http://127.0.0.1:10002/")
    parser.add_argument("--url-age-gender", type=str,
                        default="http://127.0.0.1:10003/")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--camera-id", type=int,
                        default="0", help="ffplay /dev/video0")
    parser.add_argument("--mode", type=str, default="image",
                        help="image or webcam or video")

    args = vars(parser.parse_args())
    print(args)

    logging.info(f"arguments given to {__file__}: {args}")

    initialize_csv()

    mode = args.pop("mode")
    if mode == "image":
        assert args["image_path"] is not None
        del args["camera_id"]
        # run_image(**args)
        run_image('http://127.0.0.1:10002/',
                  'http://127.0.0.1:10003/', args["image_path"])
    elif mode == "video":
        assert args["video_path"] is not None
        del args["camera_id"]
        video_to_frames(args["video_path"])
        path = 'video_frames'
        nl = sorted(os.listdir(path), key=lambda x: int(
            os.path.splitext(x.split('_')[1])[0]))
        for filename in nl:
            run_image('http://127.0.0.1:10002/',
                      'http://127.0.0.1:10003/', 'video_frames/'+filename)
        print_age_graph()
        print_gender_graph()
    else:
        del args["image_path"]
        run_webcam(**args)
