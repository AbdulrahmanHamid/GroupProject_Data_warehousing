# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 23:45:00 2025

@author: group2

1. One-Hot Encoded all the categorical variables.
2. applied oversampling to the whole dataset to handle imbalance data.
3. applied  normalization to the whole balanced dataset.
4. splitting data 80/20 after SMOTE and normalization.
5. Trained Logistic Regression and Decision Tree on the data.
6. Selection: Saved the Decision Tree model.
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from imblearn.over_sampling import SMOTE

# 1- Load the Cleaned Data
path = "C:\\Users\\abdul\\OneDrive\\Desktop\\Centennial S\\Semester 5\\data wherehousing\\GroupProject_Data_warehousing\\"
filename = 'cleaned_major_crime_group2.csv'
fullpath = os.path.join(path, filename)
data_group2 = pd.read_csv(fullpath)

print("Data Loaded. Shape:", data_group2.shape)

########################## 6- Feature columns

# Target and Features
feature_list = ['OCC_YEAR', 'OCC_MONTH', 'OCC_DOW', 'TIME_OF_DAY', 'DIVISION', 'PREMISES_TYPE']
target_var = 'MCI_CATEGORY'

X = data_group2[feature_list]
Y = data_group2[target_var]

# Encode Target (Text to Numbers)
le_group2 = LabelEncoder()
Y = le_group2.fit_transform(Y)
print("Target encoded:", le_group2.classes_)




# create Dummy Variables 
X = pd.get_dummies(X, columns=['OCC_MONTH', 'OCC_DOW', 'TIME_OF_DAY', 'DIVISION', 'PREMISES_TYPE'], drop_first=True)
model_columns = list(X.columns)

# target class counts  
tt = pd.DataFrame(Y)
print("\noriginal nclass distribution\n", tt.value_counts())


########################## 7- SMOTE
print("\n applying SMOTE to the whole dataset")
smote = SMOTE(random_state=42)
X_smote, Y_smote = smote.fit_resample(X, Y)
print("balanced dataset size:", X_smote.shape)

# target class counts after SMOTE
tt = pd.DataFrame(Y_smote)
print("\nclass distribution after oversampling\n", tt.value_counts())





########################## 8- Normalization 
print("Applying MinMaxScaler to Whole Balanced Dataset...")
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X_smote)
X_norm = pd.DataFrame(X_norm, columns=X_smote.columns) 
print("Data Normalized (0-1 range).")




########################### 9- Split Data 



X_train, X_test, Y_train, Y_test = train_test_split(
    X_norm, Y_smote, test_size=0.2, random_state=42, stratify=Y_smote
)
print("training data size:",X_train.shape)

print("testing Size:", X_test.shape)

########################## 10- Build Models

# 1. Logistic Regression
print("\nMODEL 1: Logistic Regression")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, Y_train)
y_pred_log = log_model.predict(X_test)

# 2. Decision Tree
print("\nMODEL 2: Decision Tree")
dt_model = DecisionTreeClassifier(criterion='entropy', min_samples_split=20, random_state=99)
dt_model.fit(X_train, Y_train)
y_pred_dt = dt_model.predict(X_test)

########################## 11- Model Comparison Table

print("\n===== MODEL COMPARISON TABLE =====")
comparison_table = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [
        accuracy_score(Y_test, y_pred_log),
        accuracy_score(Y_test, y_pred_dt)
    ],
    "Precision": [
        precision_score(Y_test, y_pred_log, average="weighted", zero_division=0),
        precision_score(Y_test, y_pred_dt, average="weighted", zero_division=0)
    ],
    "Recall": [
        recall_score(Y_test, y_pred_log, average="weighted", zero_division=0),
        recall_score(Y_test, y_pred_dt, average="weighted", zero_division=0)
    ],
    "F1 Score": [
        f1_score(Y_test, y_pred_log, average="weighted", zero_division=0),
        f1_score(Y_test, y_pred_dt, average="weighted", zero_division=0)
    ]
})

print(comparison_table)

# Select Decision Tree
best_model = dt_model
y_pred_best = y_pred_dt
best_model_name = "Decision Tree"

print(f"\nWe have selected {best_model_name}")

########################## 12- Evaluation (Confusion Matrix & ROC)

# Confusion Matrix
cm = confusion_matrix(Y_test, y_pred_best)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_group2.classes_, yticklabels=le_group2.classes_)
plt.title(f'Confusion Matrix ({best_model_name})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, y_pred_best, target_names=le_group2.classes_))

# ROC Curve
y_test_bin = label_binarize(Y_test, classes=range(len(le_group2.classes_)))
n_classes = y_test_bin.shape[1]
y_score = best_model.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(le_group2.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve ({best_model_name})')
plt.legend(loc="lower right")
plt.show()

########################## 13- Serialization (Saving)

joblib.dump(best_model, os.path.join(path, 'model_group2.pkl'))
print("Model dumped!")

joblib.dump(model_columns, os.path.join(path, 'model_columns_group2.pkl'))
print("Model columns dumped!")

joblib.dump(scaler, os.path.join(path, 'model_scaler_group2.pkl'))
print("Scaler dumped!")

joblib.dump(le_group2, os.path.join(path, 'model_encoder_group2.pkl'))
print("Encoder dumped!")