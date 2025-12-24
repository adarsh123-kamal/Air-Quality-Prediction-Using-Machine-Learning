# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 23:37:25 2025

@author: adars
"""

# -*- coding: utf-8 -*-
"""
Air Quality ML Project
Created on Tue Dec  9 23:37:25 2025

@author: adars
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

sns.set(style="whitegrid")

# ---------------- Load Dataset ----------------
df = pd.read_csv("C:/Users/adars/Downloads/paml/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv")

print("Dataset loaded:", df.shape)
print(df.head())

# ---------------- Preprocessing ----------------
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce', dayfirst=True)

df['hour'] = df['last_update'].dt.hour.fillna(-1)
df['day'] = df['last_update'].dt.day.fillna(-1)
df['month'] = df['last_update'].dt.month.fillna(-1)

# numeric cleaning
num_cols = ['pollutant_min','pollutant_max','pollutant_avg','latitude','longitude']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c].fillna(df[c].median(), inplace=True)

# encode categorical columns
le = LabelEncoder()

df['pollutant_id_enc'] = le.fit_transform(df['pollutant_id'].astype(str))
df['city_enc']        = le.fit_transform(df['city'].astype(str))
df['state_enc']       = le.fit_transform(df['state'].astype(str))
df['station_enc']     = le.fit_transform(df['station'].astype(str))

# ---------------- Create Labels ----------------
def aqi_label(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Moderate"
    elif x <= 200:
        return "Poor"
    else:
        return "Hazardous"

df['aq_label'] = df['pollutant_avg'].apply(aqi_label)
df['aq_label_enc'] = le.fit_transform(df['aq_label'])

print("\nAir Quality Class Counts:")
print(df['aq_label'].value_counts())

# ======================================================
#                    E D A   (Improved)
# ======================================================

# 1. Pollutant type frequency
plt.figure(figsize=(7,4))
sns.countplot(x='pollutant_id', data=df,
              order=df['pollutant_id'].value_counts().index)
plt.title("Pollutant Type Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Distribution of pollutant_avg
plt.figure(figsize=(7,4))
sns.histplot(df['pollutant_avg'], kde=True, bins=30)
plt.title("Distribution of Pollutant Average")
plt.xlabel("pollutant_avg")
plt.tight_layout()
plt.show()

# 3. Boxplot of pollutant_avg by air quality label
plt.figure(figsize=(7,4))
sns.boxplot(x='aq_label', y='pollutant_avg', data=df,
            order=['Good', 'Moderate', 'Poor', 'Hazardous'])
plt.title("Pollutant Average by Air Quality Category")
plt.tight_layout()
plt.show()

# 4. Class balance barplot
plt.figure(figsize=(6,4))
sns.countplot(x='aq_label', data=df,
              order=['Good', 'Moderate', 'Poor', 'Hazardous'])
plt.title("Air Quality Class Counts")
plt.tight_layout()
plt.show()

# 5. Average pollutant by hour of day
hourly = df.groupby('hour')['pollutant_avg'].mean().reset_index()
plt.figure(figsize=(7,4))
sns.lineplot(x='hour', y='pollutant_avg', data=hourly, marker='o')
plt.title("Average Pollutant Level by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average pollutant_avg")
plt.tight_layout()
plt.show()

# 6. Correlation heatmap (as before, but slightly expanded)
corr_cols = num_cols + ['hour', 'month']
plt.figure(figsize=(8,6))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---------------- Feature Setup ----------------
features = [
    'pollutant_min','pollutant_max','latitude','longitude',
    'pollutant_id_enc','hour','month','city_enc','state_enc'
]

X = df[features]
y_class = df['aq_label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.25, random_state=42, stratify=y_class
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Train Models ----------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "SVC": SVC()
}

model_accuracies = {}

for name, model in models.items():
    print("\n==============================")
    print("Training:", name)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    model_accuracies[name] = acc

    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, preds, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# ---------------- Compare Model Accuracies ----------------
plt.figure(figsize=(7,4))
names = list(model_accuracies.keys())
scores = list(model_accuracies.values())
sns.barplot(x=names, y=scores)
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.tight_layout()
plt.show()

print("\nModel Accuracies:")
for n, s in model_accuracies.items():
    print(f"{n}: {s:.4f}")

# ---------------- Feature Importance (RandomForest) ----------------
# Use the trained RandomForest model to see which features are important
rf_model = models["RandomForest"]
importances = rf_model.feature_importances_

plt.figure(figsize=(7,4))
sns.barplot(x=importances, y=features)
plt.title("RandomForest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ---------------- Regression Baseline (optional) ----------------
y_reg = df['pollutant_avg']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.25, random_state=42
)

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train_r, y_train_r)
preds_r = reg.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, preds_r))   # FIXED
mae = mean_absolute_error(y_test_r, preds_r)
r2 = r2_score(y_test_r, preds_r)

print("\nRegression Performance:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 :", r2)

# Scatter plot: actual vs predicted
plt.figure(figsize=(6,5))
plt.scatter(y_test_r, preds_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()],
         'r--')  # ideal line
plt.xlabel("Actual pollutant_avg")
plt.ylabel("Predicted pollutant_avg")
plt.title("Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residual distribution
residuals = y_test_r - preds_r
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Regression Residuals Distribution")
plt.xlabel("Residual")
plt.tight_layout()
plt.show()
