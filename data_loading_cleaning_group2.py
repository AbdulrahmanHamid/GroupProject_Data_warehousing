# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:30:00 2025

@author: group2

1. Data Loaing and Exploration
2. Data Cleaning: dropped missing values and 'NSA' (Not Specified Area) to ensure location accuracy.
3. Created a 'TIME_OF_DAY' feature to categorize crimes (morning, afternoon, etc.....)
also we removed dublicate columan by taking the 'EVENT_UNIQUE_ID' and remove anything appeared twice 
4. Feature Selection: Removed 'Date_Report' columns and identifiers to prevent data leakage.

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1- Load the Data
path = "C:\\Users\\abdul\\OneDrive\\Desktop\\Centennial S\\Semester 5\\data wherehousing\\GroupProject_Data_warehousing\\"
filename = 'Major_Crime.csv'
fullpath = os.path.join(path, filename)
data_group2_crime = pd.read_csv(fullpath)

# ----------------- DATA EXPLORATION -----------------

# description
print("Column Names:")
print(data_group2_crime.columns.values)

print("\nShape (Rows, Columns):")
print(data_group2_crime.shape)

print("\nData Types:")
print(data_group2_crime.dtypes)

pd.set_option('display.max_columns', None)
print("\nFirst 5 Rows:")
print(data_group2_crime.head())

print("\nDescriptive Statistics (Numerical):")
print(data_group2_crime.describe())

print("\nDescriptive Statistics (Categorical):")
print(data_group2_crime.describe(include=['O']))

print("\nUnique Target Categories:")
print(data_group2_crime['MCI_CATEGORY'].unique())
print(data_group2_crime['MCI_CATEGORY'].value_counts())

print("\nMissing values before cleaning:")
print(data_group2_crime.isnull().sum())


# ----------------- DATA CLEANING -----------------

# Create a copy since we do not want to alter our main df
data_cleaned_group2 = data_group2_crime.copy()

# 3- Handling Missing Values and 'NSA'


if 'NEIGHBOURHOOD_158' in data_cleaned_group2.columns:
    # Filter out rows where Neighbourhood is 'NSA' using standard pandas filtering
    nsa_count = data_cleaned_group2[data_cleaned_group2['NEIGHBOURHOOD_158'] == 'NSA'].shape[0]
    print(f"\nDropping {nsa_count} rows with 'NSA' location...")
    data_cleaned_group2 = data_cleaned_group2[data_cleaned_group2['NEIGHBOURHOOD_158'] != 'NSA']

# drop rows with any missing  (anything with 'NaN')
data_cleaned_group2.dropna(inplace=True)

print("\nTotal missing values after dropping rows:", data_cleaned_group2.isnull().sum().sum())
print("Shape after dropping missing values & NSA:", data_cleaned_group2.shape)

# Remove duplicate rows
print("\nTotal Number of Duplicate Data: ", data_cleaned_group2.duplicated(subset=['EVENT_UNIQUE_ID']).sum())
data_cleaned_group2.drop_duplicates(subset=['EVENT_UNIQUE_ID'], inplace=True)
print("Total number of Rows After Removing Duplicate Data", data_cleaned_group2.shape[0])


# ----------------- FEATURE ENGINEERING -----------------

# Create 'TIME_OF_DAY' from 'OCC_HOUR' to help the model find patterns 

data_cleaned_group2["TIME_OF_DAY"] = pd.cut(
    data_cleaned_group2["OCC_HOUR"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
    include_lowest=True
)
print("\nFeature Engineering: Added 'TIME_OF_DAY'.")


# -----------------  featured column selection -----------------

# WHY we drop these columns:
# 1. EVENT_UNIQUE_ID / OBJECTID: random IDs will cause overfitting.
# 2. REPORT_ Columns: The report date happens AFTER the crime. We predict based on occurrence
# 3. HOOD_140 / NEIGHBOURHOOD_140: Old standards, redundant with 158.
# 4. x / y: Duplicates of LAT/LONG.
# 5. OFFENCE: (Target Leakage).

columns_to_drop = [
    'EVENT_UNIQUE_ID', 'OBJECTID', 'OFFENCE', 
    'HOOD_140', 'NEIGHBOURHOOD_140', 'x', 'y',
    'UCR_CODE', 'UCR_EXT', 
    'REPORT_DATE', 'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'REPORT_DOY', 'REPORT_DOW', 'REPORT_HOUR'
]

# Drop columns
data_cleaned_group2.drop(columns=columns_to_drop, axis=1, inplace=True)

print("\nColumns after dropping unnecessary features:")
print(data_cleaned_group2.columns.values)

# Save the cleaned data in a new csv file called clean
data_cleaned_group2.to_csv(os.path.join(path, "cleaned_major_crime_group2.csv"), index=False)
print("Cleaned data saved successfully.")


# ----------------- VISUALIZATIONS -----------------

# Plot target variable distribution
plt.figure(figsize=(10,6))
sns.countplot(x='MCI_CATEGORY', data=data_cleaned_group2, palette='viridis')
plt.title('Frequency of Major Crime Categories')
plt.show()

# Plot crimes by Year
plt.figure(figsize=(20,6))
sns.countplot(x='OCC_YEAR', data=data_cleaned_group2, palette='coolwarm')
plt.title('Crimes by Occurrence Year')
plt.show()

# Plot crimes by Time of Day
plt.figure(figsize=(10,6))
sns.countplot(x='TIME_OF_DAY', data=data_cleaned_group2, palette='pastel')
plt.title('Crimes by Time of Day')
plt.show()