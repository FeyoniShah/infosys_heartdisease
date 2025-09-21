import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cv2
import pytesseract
from PIL import Image
import re



df=pd.read_csv("framingham.csv")
print("Five rows of dataset")
print(df.head())



print("\n Dataframe shape(rows,column):",df.shape)    

print(df.columns) 
print("\n Dataframeiformation")
print(df.info())


print(df.dtypes)   


#.T means transpose. In original .describe() our count,mean etc are rows so transpose makes it column
print("Statistical summary of numerical features")
print(df.describe().T  )


target_count=df['TenYearCHD'].value_counts()
print(target_count)


plt.figure(figsize=(6,4))
sns.barplot(x=target_count.index,y=target_count.values)
plt.title('Distribution of Target ariables')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()




df['age'].hist(bins=30)
plt.title("Age Distribution")
plt.show()



plt.boxplot(df['totChol'].dropna())
plt.title("Cholesterol Boxplot")
plt.show()


df.corr(numeric_only=True)['TenYearCHD'].sort_values(ascending=False)


correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()