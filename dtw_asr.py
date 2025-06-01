import os
import numpy as np
import librosa
from scipy.spatial.distance import cdist
from collections import Counter

DATA_PATH = "data"

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def dtw_distance(x, y):
    D = cdist(x, y, 'euclidean')
    acc_cost = np.zeros_like(D)
    acc_cost[0, 0] = D[0, 0]
    for i in range(1, D.shape[0]):
        acc_cost[i, 0] = D[i, 0] + acc_cost[i - 1, 0]
    for j in range(1, D.shape[1]):
        acc_cost[0, j] = D[0, j] + acc_cost[0, j - 1]
    for i in range(1, D.shape[0]):
        for j in range(1, D.shape[1]):
            acc_cost[i, j] = D[i, j] + min(acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1])
    return acc_cost[-1, -1]

def load_dataset(base_dir):
    features, labels = [], []
    for fname in os.listdir(base_dir):
        if fname.endswith('.wav'):
            label = fname.split('_')[0]
            mfcc = extract_mfcc(os.path.join(base_dir, fname))
            features.append(mfcc)
            labels.append(label)
    return features, labels

def predict_dtw(train_features, train_labels, test_feat):
    distances = [dtw_distance(test_feat, train_feat) for train_feat in train_features]
    return train_labels[np.argmin(distances)]

def evaluate(preds, trues):
    classes = sorted(set(trues))
    report = {}
    for cls in classes:
        TP = sum((p == cls) and (t == cls) for p, t in zip(preds, trues))
        FP = sum((p == cls) and (t != cls) for p, t in zip(preds, trues))
        FN = sum((p != cls) and (t == cls) for p, t in zip(preds, trues))
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        report[cls] = (precision, recall, f1)
    return report

# --- Main ---
train_feat, train_label = load_dataset(os.path.join(DATA_PATH, "train"))
test_feat, test_label = load_dataset(os.path.join(DATA_PATH, "test"))

preds = [predict_dtw(train_feat, train_label, test) for test in test_feat]

report = evaluate(preds, test_label)
for cls, metrics in report.items():
    print(f"{cls}: Precision={metrics[0]:.2f}, Recall={metrics[1]:.2f}, F1={metrics[2]:.2f}")
