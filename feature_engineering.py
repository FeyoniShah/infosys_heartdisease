import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind
from data_clean import clean_data   

df=pd.read_csv("framingham.csv")

def run_feature_pipeline(file_path="framingham.csv", plot_heatmap=True):
    # -------------------------------
    # 0. Load and Clean Data
    # -------------------------------
    df = pd.read_csv(file_path)
    df = clean_data(df)

    # -------------------------------
    # 1. Normalization (MinMax Scaling)
    # -------------------------------
    cols_to_normalize = ['totChol', 'sysBP']
    scaler_minmax = MinMaxScaler()
    df[cols_to_normalize] = scaler_minmax.fit_transform(df[cols_to_normalize])
    print("\nNormalized Data (totChol & sysBP only, first 5 rows):")
    print(df[cols_to_normalize].head())

    # -------------------------------
    # 2. Standardization
    # -------------------------------
    scaler = StandardScaler()
    cols_to_scale = ['totChol', 'sysBP']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print("\nStandardized Data (totChol & sysBP only, first 5 rows):")
    print(df[cols_to_scale].head())


    # -------------------------------
    # 4. Derived Features
    # -------------------------------
    df['AgeGroup'] = pd.cut(df['age'],
                            bins=[0, 30, 45, 60, 100],
                            labels=['Young', 'Adult', 'MiddleAge', 'Senior'])

    df['RiskScore'] = (df['currentSmoker'] +
                       df['diabetes'] +
                       df['prevalentStroke'] +
                       df['prevalentHyp'])

    print("\nDerived Features (AgeGroup & RiskScore):")
    print(df[['age', 'AgeGroup', 'RiskScore']].head())
    print(df.head())

    # -------------------------------
    # 5. Feature Selection Analysis
    # -------------------------------
    numeric_df = df.select_dtypes(include=[np.number])

    if plot_heatmap:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True)
        plt.title("Correlation Heatmap")
        plt.show()

    # Prepare for feature importance
    X = df.drop("TenYearCHD", axis=1)
    X['AgeGroup'] = X['AgeGroup'].astype(str)
    X['AgeGroup'] = LabelEncoder().fit_transform(X['AgeGroup'])

    y = df["TenYearCHD"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X.fillna(0), y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importance (RandomForest):")
    print(importances.head(10))

    # -------------------------------
    # 6. Statistical Validation
    # -------------------------------
    smokers = df[df['currentSmoker'] == 1]['totChol'].dropna()
    nonsmokers = df[df['currentSmoker'] == 0]['totChol'].dropna()
    t_stat, p_value = ttest_ind(smokers, nonsmokers)
    print(f"\nT-Test for Cholesterol (Smokers vs Non-Smokers): t={t_stat:.2f}, p={p_value:.4f}")


    #  # -------------------------------
    # # 3. Label Encoding (all object cols)
    # # -------------------------------
    # df_encoded = df.copy()
    # label_encoders = {}
    # object_cols = df_encoded.select_dtypes(include=['object']).columns

    # for col in object_cols:
    #     le = LabelEncoder()
    #     df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    #     label_encoders[col] = le  # keep encoders if needed

    # print("\nLabel Encoded Data (object columns transformed):")
    # print(df_encoded[object_cols].head())


     # -------------------------------
    # 3. Label Encoding (all object cols)
    # -------------------------------
    #df_encoded = df.copy()
    label_encoders = {}
    object_cols = df.select_dtypes(include=['object','category']).columns

    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # keep encoders if needed

    print("\nLabel Encoded Data (object columns transformed):")
    print(df[object_cols].head())


    print(df.dtypes)


    return df


#run_feature_pipeline(file_path="framingham.csv", plot_heatmap=False)