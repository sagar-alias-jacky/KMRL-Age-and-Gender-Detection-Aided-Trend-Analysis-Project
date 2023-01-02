import os
from natsort import natsorted

path='video_frames'
nl=sorted(os.listdir(path),key=lambda x: int(os.path.splitext(x.split('_')[1])[0]))
# nl=sorted(nl)
print(nl)

for filename in nl:
    print('video_frames/'+filename)