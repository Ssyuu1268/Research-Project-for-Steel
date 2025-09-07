#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os, random, numpy as np, tensorflow as tf

# === Preprocess for Baseline (PCA only) ===
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === 1. Load Data ===
df = pd.read_excel('Steel2.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df[(df['Date'] >= '2005-03-01') & (df['Date'] <= '2021-06-30')].reset_index(drop=True)

# === 2. Define Target Variable ===
df['Log Return'] = df['Log Return'].shift(-1)
df['Target'] = (df['Log Return'] > 0).astype(int)

# === 3. Drop NA for features used ===
feature_cols = ['S_MA5', 'S_MA20', 'USD/TWD', 'Oil Price', 'M2',
                'Discount Rate', 'VIX index', 'CPI index', 'Log Return', 'Target']
df = df.dropna(subset=feature_cols).reset_index(drop=True)

# === 4. PCA on macroeconomic variables ===
pca_columns = ['S_MA5', 'S_MA20', 'USD/TWD', 'Oil Price', 'M2',
               'Discount Rate', 'VIX index', 'CPI index']
pca_data = df[pca_columns]
scaled_data = StandardScaler().fit_transform(pca_data)
pca = PCA()
pca_scores = pd.DataFrame(pca.fit_transform(scaled_data), columns=[f'PC{i+1}' for i in range(len(pca.components_))])
pca_top = pca_scores.iloc[:, :3].reset_index(drop=True)

# === 5. Use PCA only (baseline) ===
X_all = pca_top.astype('float32')
y_all = df['Target'].iloc[:len(X_all)].reset_index(drop=True)

# === 6. Create sequences ===
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_all, y_all)

# === 7. Walk-forward validation ===
def walk_forward_validation(X_seq, y_seq, n_splits=5, test_size=0.2):
    step = int(len(X_seq) * (1 - test_size) / n_splits)
    all_reports = []

    for i in range(n_splits):
        start = i * step
        end_train = start + step
        end_test = end_train + int(len(X_seq) * test_size)

        if end_test > len(X_seq):
            break

        X_train, y_train = X_seq[start:end_train], y_seq[start:end_train]
        X_test, y_test = X_seq[end_train:end_test], y_seq[end_train:end_test]

        model = Sequential()
        model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        report = classification_report(y_test, y_pred, output_dict=True)
        all_reports.append(report)

        print(f" Fold {i+1}: "
              f"Accuracy={report['accuracy']:.4f} "
              f"Precision={report['1']['precision']:.4f} "
              f"Recall={report['1']['recall']:.4f} "
              f"F1={report['1']['f1-score']:.4f}")

    return all_reports


# === 8. Run baseline model ===
reports = walk_forward_validation(X_seq, y_seq, n_splits=5, test_size=0.2)

# === 9. Output average result ===
avg_acc = np.mean([r['accuracy'] for r in reports])
avg_precision = np.mean([r['1']['precision'] for r in reports])
avg_recall = np.mean([r['1']['recall'] for r in reports])
avg_f1 = np.mean([r['1']['f1-score'] for r in reports])

print("\n Average Model Performance (PCA only — Baseline):")
print(f"Accuracy     : {avg_acc:.4f}")
print(f"Precision (1): {avg_precision:.4f}")
print(f"Recall (1)   : {avg_recall:.4f}")
print(f"F1-score (1) : {avg_f1:.4f}")


# In[ ]:




