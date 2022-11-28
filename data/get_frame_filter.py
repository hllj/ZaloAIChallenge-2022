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
    parser = argparse.ArgumentParser(description="Extract 1 frame/sec")
    parser.add_argument("-i", "--input", default="train/videos")
    parser.add_argument("-o", "--output", default="train/images1")
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(input_dir)
    for idx, filename in enumerate(filenames):
        filepath = os.path.join(input_dir, filename)
        print(idx, filepath)
        name, file_extension = os.path.splitext(filename)

        frames = read_video(filepath)
        output = os.path.join(output_dir, name)
        if not os.path.exists(output):
            os.mkdir(output)

        for idx, frame in enumerate(frames):
            less = np.less(frame, 205)
            frame[less] = 0
            cv2.imwrite(os.path.join(output, str(idx) + ".jpg"), frame)
