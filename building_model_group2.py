# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 23:45:00 2025

@author: group2

1. One-Hot Encoded all the categorical variables.
2. Splitting: Split data 80/20.
3. Normalization Applied  Scaling after splitting to prevent data leakage.
4. Tested SMOTE Oversampling to handle imbalance.
5. Compared Logistic Regression vs Decision Tree using Accuracy & Confusion Matrices.
6. Selection: Saved the Decision Tree model and Normalization parameters.
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
# We select 'OCC_' features for when and timing and 'DIVISION'/'PREMISES' for where
feature_list = ['OCC_YEAR', 'OCC_MONTH', 'OCC_DOW', 'TIME_OF_DAY', 'DIVISION', 'PREMISES_TYPE']
target_var = 'MCI_CATEGORY'

X = data_group2[feature_list]
Y = data_group2[target_var]

# Encode Target (Text to Numbers)
le_group2 = LabelEncoder()
Y = le_group2.fit_transform(Y)
print("Target encoded:", le_group2.classes_)

# Create Dummy Variables for Categorical Features
X = pd.get_dummies(X, columns=['OCC_MONTH', 'OCC_DOW', 'TIME_OF_DAY', 'DIVISION', 'PREMISES_TYPE'], drop_first=True)
model_columns = list(X.columns)


########################### 7- Split Data 
# We split first to make sure no data leakage

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
print(f"Original Training Size: {X_train.shape[0]}")



########################## 8- SMOTE
# Applying oversampling to see how the our model will get 
print("\nApplying SMOTE to Training Data...")
smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
print(f"Balanced Training Size: {X_train_smote.shape[0]}")




########################## 9- Normalization 
# we made two scalers one the data with Smote and the second for the orginal cleaned dataset without    

# Scaler 1: for smote
scaler_smote = MinMaxScaler()
X_train_smote_norm = scaler_smote.fit_transform(X_train_smote)
X_test_smote_norm = scaler_smote.transform(X_test)


# Scaler 2: for original data
scaler_base = MinMaxScaler()
X_train_norm = scaler_base.fit_transform(X_train)
X_test_norm = scaler_base.transform(X_test)

print("the normalized data are in (0-1 range).")

########################## 10- Build Models all cases with balancing and without

# 1. Logistic Regression Original data
print("\nMODEL 1: Logistic Regression original data")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_norm, Y_train)
y_pred_log = log_model.predict(X_test_norm)

# 2. Decision Tree Original data 
print("\nMODEL 2: Decision Tree original data")
dt_model = DecisionTreeClassifier(criterion='entropy', min_samples_split=20, random_state=99)
dt_model.fit(X_train_norm, Y_train)
y_pred_dt = dt_model.predict(X_test_norm)

# 3. Logistic Regression data used SMOTE
print("\nLogistic Regression With SMOTE ")
log_model_smote = LogisticRegression(max_iter=1000)
log_model_smote.fit(X_train_smote_norm, Y_train_smote)
y_pred_log_smote = log_model_smote.predict(X_test_smote_norm)

# 4. Decision Tree data used SMOTE
print("\nDecision Tree With SMOTE")
dt_model_smote = DecisionTreeClassifier(criterion='entropy', min_samples_split=20, random_state=99)
dt_model_smote.fit(X_train_smote_norm, Y_train_smote)
y_pred_dt_smote = dt_model_smote.predict(X_test_smote_norm)

########################## 11- Full Model Comparison Table

print("\n===== MODEL COMPARISON TABLE =====")
comparison_table = pd.DataFrame({
    "Model": [
        "Logistic Regression (Original)", 
        "Decision Tree (Original)", 
        "Logistic Regression (SMOTE)", 
        "Decision Tree (SMOTE)"
    ],
    "Accuracy": [
        accuracy_score(Y_test, y_pred_log),
        accuracy_score(Y_test, y_pred_dt),
        accuracy_score(Y_test, y_pred_log_smote),
        accuracy_score(Y_test, y_pred_dt_smote)
    ],
    "Precision": [
        precision_score(Y_test, y_pred_log, average="weighted", zero_division=0),
        precision_score(Y_test, y_pred_dt, average="weighted", zero_division=0),
        precision_score(Y_test, y_pred_log_smote, average="weighted", zero_division=0),
        precision_score(Y_test, y_pred_dt_smote, average="weighted", zero_division=0)
    ],
    "Recall": [
        recall_score(Y_test, y_pred_log, average="weighted", zero_division=0),
        recall_score(Y_test, y_pred_dt, average="weighted", zero_division=0),
        recall_score(Y_test, y_pred_log_smote, average="weighted", zero_division=0),
        recall_score(Y_test, y_pred_dt_smote, average="weighted", zero_division=0)
    ],
    "F1 Score": [
        f1_score(Y_test, y_pred_log, average="weighted", zero_division=0),
        f1_score(Y_test, y_pred_dt, average="weighted", zero_division=0),
        f1_score(Y_test, y_pred_log_smote, average="weighted", zero_division=0),
        f1_score(Y_test, y_pred_dt_smote, average="weighted", zero_division=0)
    ]
})

print(comparison_table)

# selecting original Decision Tree becuase it is best
best_model = dt_model
best_scaler = scaler_base
y_pred_best = y_pred_dt
best_model_name = "Decision Tree original data"

print(f"\n we have selected {best_model_name}")






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
# We use X_test_norm because that corresponds to the Original Data model
y_test_bin = label_binarize(Y_test, classes=range(len(le_group2.classes_)))
n_classes = y_test_bin.shape[1]
y_score = best_model.predict_proba(X_test_norm)

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

# Save the BASELINE scaler (because we selected the Baseline model)
joblib.dump(best_scaler, os.path.join(path, 'model_scaler_group2.pkl'))
print("Scaler dumped!")

joblib.dump(le_group2, os.path.join(path, 'model_encoder_group2.pkl'))
print("Encoder dumped!")