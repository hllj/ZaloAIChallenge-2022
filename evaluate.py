import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, accuracy_score
import argparse


def evaluate(gt_file, pred_file, threshold):
    df1 = pd.read_csv(gt_file)
    df2 = pd.read_csv(pred_file)
    df_all = pd.merge(df1, df2, on="fname")
    print(df_all)
    paths = df_all["fname"]
    labels = list(map(int, df_all["liveness_gt"]))
    predicteds = list(map(float, df_all["liveness_score"]))

    ##eer prob
    fpr, tpr, thresholds = roc_curve(labels, predicteds, pos_label=1)
    # print(fpr, tpr, thresholds)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)

    #eer class
    predicteds = [1 if i > threshold else 0 for i in predicteds]
    fpr, tpr, thresholds = roc_curve(labels, predicteds, pos_label=1)
    # print(fpr, tpr, thresholds)
    class_eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(class_eer)

    #acc class
    acc = accuracy_score(labels, predicteds)
    for idx in range(len(paths)):
        if labels[idx] != predicteds[idx]:
            print(paths[idx], labels[idx], predicteds[idx])
    return eer, class_eer, acc



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='data/public/labels.csv')
    parser.add_argument('--predict', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    eer, class_eer, acc = evaluate(args.label, args.predict, args.threshold)
    print(eer, class_eer, acc)
    
