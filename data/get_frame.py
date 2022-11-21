import argparse
import os

import cv2
import numpy as np
from PIL import Image


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


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 1 frame/sec")
    parser.add_argument("-i", "--input", default="train/videos")
    parser.add_argument("-o", "--output", default="train/pil_images")
    parser.add_argument("-p", "--padding", action="store_true")
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    is_padding = args.padding
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(input_dir)
    for file_idx, filename in enumerate(filenames):
        filepath = os.path.join(input_dir, filename)
        print(file_idx, filepath)
        name, file_extension = os.path.splitext(filename)

        frames = read_video(filepath)
        output = os.path.join(output_dir, name)
        if not os.path.exists(output):
            os.mkdir(output)

        for idx, frame in enumerate(frames):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            max_size = max(height, width)
            if is_padding:
                img, ratio, (dw, dh) = letterbox(
                    img, (max_size, max_size), color=(0, 0, 0)
                )
            im_pil = Image.fromarray(img)
            im_pil.save(os.path.join(output, str(idx) + ".png"))
