import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cv2
import pytesseract
from PIL import Image
import re

df=pd.read_csv("framingham.csv")

def clean_data(data):
    
    # # Make a copy 
    # cleaned_df = data.copy()
    
    # # Handle missing values
    # # Numeric columns → fill with median
    # for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
    #     median_val = cleaned_df[col].median()
    #     cleaned_df[col].fillna(median_val, inplace=True)
    
    # # Categorical columns → fill with mode
    # for col in cleaned_df.select_dtypes(include=['object']).columns:
    #     mode_val = cleaned_df[col].mode()[0]
    #     cleaned_df[col].fillna(mode_val, inplace=True)
    
    # #  Handle outliers (IQR method) for numeric columns
    # for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
    #     Q1 = cleaned_df[col].quantile(0.25)
    #     Q3 = cleaned_df[col].quantile(0.75)
    #     IQR = Q3 - Q1
        
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
        
    #     # Keep only values inside the range
    #     cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    # return cleaned_df

    # cleaned_df = data.copy()

    # # Fill missing values
    # for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
    #     median_val = cleaned_df[col].median()
    #     cleaned_df[col].fillna(median_val, inplace=True)

    # for col in cleaned_df.select_dtypes(include=['object']).columns:
    #     mode_val = cleaned_df[col].mode()[0]
    #     cleaned_df[col].fillna(mode_val, inplace=True)

    # # Handle outliers (IQR) - apply across all numeric cols at once
    # numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    # mask = pd.Series(True, index=cleaned_df.index)

    # for col in numeric_cols:
    #     Q1 = cleaned_df[col].quantile(0.25)
    #     Q3 = cleaned_df[col].quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #     mask &= (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)

    # cleaned_df = cleaned_df[mask]

    # return cleaned_df


    cleaned_df = data.copy()

    # Fill missing values
    for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
        median_val = cleaned_df[col].median()
        cleaned_df[col] = cleaned_df[col].fillna(median_val)

    for col in cleaned_df.select_dtypes(include=['object']).columns:
        mode_val = cleaned_df[col].mode()[0]
        cleaned_df[col] = cleaned_df[col].fillna(mode_val)

    # Outlier filtering
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    mask = pd.Series(True, index=cleaned_df.index)

    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask &= (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)

    # Always keep rows where TenYearCHD == 1
    cleaned_df = cleaned_df[mask | (cleaned_df["TenYearCHD"] == 1)]

    return cleaned_df


