import numpy as np
import os
import cv2
ROOT_DIR = "datasets/cholec80"
VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, './videos/'))
VIDEO_NAMES = [x for x in VIDEO_NAMES if 'mp4' in x]
TRAIN_NUMBERS = np.arange(1,41).tolist()
VAL_NUMBERS = np.arange(41,49).tolist()
TEST_NUMBERS = np.arange(49,81).tolist()

for video_name in VIDEO_NAMES:
    print(video_name)
    vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, './videos/' + video_name))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)
    if fps != 25:
        print(video_name, 'not at 25fps', fps)
    success=True
    count=0
    vid_id = int(video_name.replace('.mp4', '').replace("video", ""))
    if vid_id in TRAIN_NUMBERS:
        save_dir = './frames/train/' + video_name.strip('.mp4') +'/'
    elif vid_id in VAL_NUMBERS:
        save_dir = './frames/val/' + video_name.strip('.mp4') +'/'
    elif vid_id in TEST_NUMBERS:
        save_dir = './frames/test/' + video_name.strip('.mp4') +'/'
    save_dir = os.path.join(ROOT_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    while success is True:
        success,image = vidcap.read()
        if success:
            cv2.imwrite(save_dir + str(count) + '.jpg', image)
        count+=1
    vidcap.release()
    cv2.destroyAllWindows()