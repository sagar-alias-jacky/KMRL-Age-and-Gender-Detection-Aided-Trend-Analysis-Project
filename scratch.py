video_path="test_videos/SN Junction_Monday_20-03-2023_7:00.mp4"

x=video_path.split("/")
print(x)

y=x[-1].split("_")
station=y[0]
day=y[1]
date=y[2]
time=y[3].split(".")[0]
print(time)