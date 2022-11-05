import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_csv(fname, y, input_dir, csv_filename):
    csv_file = open(os.path.join(input_dir, csv_filename), "w")
    csv_file.write("fname,liveness_score\n")
    for filename, label in zip(fname, y):
        name = filename.split(".")[0]
        list_image_filename = sorted(
            os.listdir(os.path.join(input_dir, "images", name))
        )
        for image_filename in list_image_filename:
            image_path = os.path.join(input_dir, "images", name, image_filename)
            csv_file.write(f"{image_path},{label}\n")
    csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val")
    parser.add_argument("-dir", "--directory", default="train")
    parser.add_argument("-l", "--label", default="label.csv")
    args = parser.parse_args()
    input_dir = args.directory
    label_filename = args.label
    label_df = pd.read_csv(os.path.join(input_dir, label_filename))
    fname_train, fname_val, y_train, y_val = train_test_split(
        label_df["fname"],
        label_df["liveness_score"],
        test_size=0.33,
        random_state=42,
        stratify=label_df["liveness_score"],
    )
    print("train class count:", y_train.value_counts())
    print("val class count:", y_val.value_counts())

    create_csv(fname_train, y_train, input_dir, "train.csv")
    create_csv(fname_val, y_val, input_dir, "val.csv")
