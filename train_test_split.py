'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engineering import run_feature_pipeline  
from data_clean import clean_data 
from ocr import MedicalOCRPipeline


df=pd.read_csv("framingham.csv")

df = clean_data(df)

print("\nCleaned dataset shape:\n", df.shape)
print("\nCleaned dataset first 5 columns\n",df.head())
print("\nisNull in cleaned dataset\n",df.isnull().sum())

df=run_feature_pipeline("framingham.csv")


df_final = df.copy()

# Stratified Train-Validation-Test Split (70-15-15)

train, temp = train_test_split(df_final, test_size=0.30, stratify=df_final['TenYearCHD'], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp['TenYearCHD'], random_state=42)

print("Shapes:")
print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

#Balanced Distribution Check

print("\nClass Distribution:")
print("Original:\n", df_final['TenYearCHD'].value_counts(normalize=True))
print("Train:\n", train['TenYearCHD'].value_counts(normalize=True))
print("Validation:\n", val['TenYearCHD'].value_counts(normalize=True))
print("Test:\n", test['TenYearCHD'].value_counts(normalize=True))

# Data Integrity Check

assert len(set(train.index).intersection(set(val.index))) == 0, "Overlap between train and val"
assert len(set(train.index).intersection(set(test.index))) == 0, "Overlap between train and test"
assert len(set(val.index).intersection(set(test.index))) == 0, "Overlap between val and test"
assert len(train) + len(val) + len(test) == len(df_final), "Row count mismatch"

print("\nData Integrity Check Passed")



# Dataset Quality Checks

def dataset_summary(dataset, name):
    print(f"\n{name} Summary:")
    print("Shape:", dataset.shape)
    print("Missing values:\n", dataset.isnull().sum().sum())
    print("Target distribution:\n", dataset['TenYearCHD'].value_counts(normalize=True))

dataset_summary(train, "Train")
dataset_summary(val, "Validation")
dataset_summary(test, "Test")




def process_new_medical_image(image_path):
    ocr = MedicalOCRPipeline()
    result = ocr.run_pipeline(image_path)
    return result["parsed_data"]

# Example
data = process_new_medical_image("Medical_Image.PNG")
print("\nOCR\n", data)
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engineering import run_feature_pipeline

def create_splits():
    # Run pipeline
    df = run_feature_pipeline(file_path="framingham.csv", plot_heatmap=False)

    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]

    # Train + Temp (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Train (70%) and Validation (15%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    )

    # Save to CSVs
    X_train.to_csv("train_X.csv", index=False)
    y_train.to_csv("train_y.csv", index=False)
    X_val.to_csv("val_X.csv", index=False)
    y_val.to_csv("val_y.csv", index=False)
    X_test.to_csv("test_X.csv", index=False)
    y_test.to_csv("test_y.csv", index=False)

    #print(pd.read_csv("train_X.csv").dtypes)

    print("Train class distribution:\n", y_train.value_counts())
    print("\nValidation class distribution:\n", y_val.value_counts())
    print("\nTest class distribution:\n", y_test.value_counts())


    print("Data split into Train/Val/Test and saved")

if __name__ == "__main__":
    create_splits()
