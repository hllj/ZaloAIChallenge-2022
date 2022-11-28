import argparse
import os

import cv2
import numpy as np


def read_video(name):
    cap = cv2.VideoCapture(name)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        fps = 25
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # print(count)
        if isinstance(frame, np.ndarray):
            if int(count % round(fps)) == 0:
                frames.append(frame)
            count += 1
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    return frames


if __name__ == "__main__":
        frames = read_video("train/videos/175.mp4")
        output_jpg = "test.jpg"
        output_png = "test.png"

        # cv2.imwrite(output_jpg, frames[0])
        # cv2.imwrite(output_png, frames[0])
        img1 = cv2.imread(output_jpg)
        img2 = cv2.imread(output_png)
        if img1.all() == img2.all():
            print("1=2")
        else: print("yep")
        if frames[0].all() == img2.all():
            print("frame=2")
        else: print("yep")
