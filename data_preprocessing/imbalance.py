import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

df = pd.read_csv('https://archive.ics.uci.edu/ml'
                 '/machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data',
                 header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)

X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

# Null classifier performance on skewed dataset can be surprisingly high
y_pred = np.zeros(y_imb.shape[0])
print(np.mean(y_pred == y_imb) * 100)

print('Number of class 1 examples beffore:', X_imb[y_imb == 1].shape[0])

X_upsampled, y_upsampled = resample(
        X_imb[y_imb == 1],
        y_imb[y_imb == 1],
        replace=True,
        n_samples=X_imb[y_imb == 0].shape[0],
        random_state=123)

print('Number of class 1 examples after:', X_upsampled.shape[0])

X_bal = np.vstack((X[y == 0], X_updsampled))
y_bal = np.hstack((y[y == 0], y_updsampled))

y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100
