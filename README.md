# EXNO:4-DS
## DEVELOPED BY: SANJAY KUMAR .B
## REG NO: 212224230242
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

#FEATURE SCALING
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()

![image](https://github.com/user-attachments/assets/5f69b7af-31a8-4d2c-9bd8-c513c33bfc16)

df_null_sum=df.isnull().sum()
df_null_sum

![image](https://github.com/user-attachments/assets/10790d42-70ff-4cb1-95a0-d62bf602488a)

df.dropna()

![image](https://github.com/user-attachments/assets/902a84ea-f0d5-4351-be5b-d0e0253386ea)

max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals

![image](https://github.com/user-attachments/assets/efb2f370-380b-4dc2-8c4f-c885f2de4515)

from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()

![image](https://github.com/user-attachments/assets/8ae35ab2-7a91-408e-b2d9-f05bfef55e5b)

sc=StandardScaler()


df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

![image](https://github.com/user-attachments/assets/c73590a4-c898-489c-8963-ffafd0a8538b)

#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

![image](https://github.com/user-attachments/assets/fea1716a-e0d4-4287-b45a-719e4e15578a)

#MAXIMUM ABSOLUTE SCALING:
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df

![image](https://github.com/user-attachments/assets/f813e432-0eb9-41c7-8268-ad51c8ea2303)

#ROBUST SCALING
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()

![image](https://github.com/user-attachments/assets/1fc8fe6d-ac53-4733-b515-2bc80829341b)

#FEATURE SELECTION:
df=pd.read_csv("/content/income(1) (1).csv")
df.info()

![image](https://github.com/user-attachments/assets/2f49476d-8841-4317-b346-6d7830e763da)

df_null_sum=df.isnull().sum()
df_null_sum

![image](https://github.com/user-attachments/assets/ca25e785-5f1d-4eeb-b7cb-9e8cffbf6409)

# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
# process non-numeric features. It ensures the model treats these columns as categories,
# not as continuous numerical values.
df[categorical_columns]

![image](https://github.com/user-attachments/assets/107a1b46-ef9f-4d3e-8ea9-70d4f0a94b3e)

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]

![image](https://github.com/user-attachments/assets/65a20df6-fa61-4768-a025-1ebbeac19800)

X = df.drop(columns=['SalStat'])
y = df['SalStat']


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

![image](https://github.com/user-attachments/assets/84fcd8c3-bc95-46a8-aa0e-cf462b169d27)

y_pred = rf.predict(X_test)


df=pd.read_csv("/content/income(1) (1).csv")
df.info()

![image](https://github.com/user-attachments/assets/fdc6c309-e603-479b-a0f8-82935930fc7e)

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

![image](https://github.com/user-attachments/assets/7fe3a351-ad04-46a0-a503-5a922c163a2d)

df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

![image](https://github.com/user-attachments/assets/d8f272ee-39a9-4267-9faa-857d5c4bc719)

X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)

![image](https://github.com/user-attachments/assets/13797152-5bcc-4fb3-85ad-7b0f3f69513f)

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

![image](https://github.com/user-attachments/assets/6f40ad3e-c847-4e0e-b4af-50289801403a)

y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")

![image](https://github.com/user-attachments/assets/70e79307-26a6-468c-b7c7-c0dc56149c43)

!pip install skfeature-chappers

![image](https://github.com/user-attachments/assets/1df78687-5f3c-459a-9b17-8a923c95ede5)

import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')


df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

![image](https://github.com/user-attachments/assets/dcad0fd4-c178-4344-948c-9adaccd9ec69)

X = df.drop(columns=['SalStat'])
y = df['SalStat']


k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)


selected_features_anova = X.columns[selector_anova.get_support()]


print("\nSelected features using ANOVA:")
print(selected_features_anova)

![image](https://github.com/user-attachments/assets/9c2d14db-f5c8-4f96-9a21-05972ab1876c)

# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')


df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)


df[categorical_columns]

![image](https://github.com/user-attachments/assets/9d212192-e4b4-4c9e-bc19-e30508be63f8)

X = df.drop(columns=['SalStat'])
y = df['SalStat']


logreg = LogisticRegression()


n_features_to_select =6


rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)

![image](https://github.com/user-attachments/assets/f08d12cb-8ec2-4030-85b3-6ad1e36a2bfd)


# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
