import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.metrics import classification_report

DATA_PATH = "data"

def load_dataset(base_dir):
    features, labels = [], []
    for fname in os.listdir(base_dir):
        if fname.endswith('.wav'):
            label = fname.split('_')[0]
            mfcc = extract_mfcc(os.path.join(base_dir, fname))
            features.append(mfcc)
            labels.append(label)
    return features, labels

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def train_hmm_models(X_train, y_train):
    models = {}
    for label in set(y_train):
        X = [x for x, y in zip(X_train, y_train) if y == label]
        lengths = [len(x) for x in X]
        X_concat = np.concatenate(X)
        model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=1000)
        model.fit(X_concat, lengths)
        models[label] = model
    return models

def predict_hmm(models, X_test):
    predictions = []
    for x in X_test:
        scores = {label: model.score(x) for label, model in models.items()}
        pred = max(scores, key=scores.get)
        predictions.append(pred)
    return predictions

# --- Main ---
X_train, y_train = load_dataset(os.path.join(DATA_PATH, "train"))
X_test, y_test = load_dataset(os.path.join(DATA_PATH, "test"))

models = train_hmm_models(X_train, y_train)
preds = predict_hmm(models, X_test)

print(classification_report(y_test, preds, zero_division=0))
