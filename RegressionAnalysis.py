#import libraries

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from collections import Counter

#defining function for regression metrics

def Reg_Models_Evaluation_Metrics(model, X_train, y_train, X_test, y_test, y_pred):
    cv_score = cross_val_score(estimator= model, X = X_train, y = y_train, cv=10)

    #calculating adjusted r squared
    r2 = model.score(X_test, y_test)

    #number of observations is the shape along axis 0 
    n = X_test.shape[0]

    #number of features (predictors, p) is the shape along axis 1
    p = X_test.shape[1]

    #adjusted r squared formula 
    adjusted_r2 = 1 - (1 - r2) * (n-1) / (n-p-1)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    R2 = model.score(X_test, y_test)
    CV_R2 = cv_score.mean()

    return R2, adjusted_r2, CV_R2, RMSE

    print('RMSE:', round(RMSE,4))
    print('R2:', round(R2,4))
    print('Adjusted R2:', round(adjusted_r2, 4) )
    print("Cross Validated R2: ", round(cv_score.mean(),4) )


#Import data

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

raw_df1 = pd.read_csv("avocado.csv")
raw_df2 = pd.read_csv("housing.csv", header=None, delimiter=r'\s+', names=column_names)


#delete a column
raw_df1 = raw_df1.drop('Unnamed: 0', axis = 1)

numeric_columns = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
categorical_columns = ['Region', 'Type']
time_columns = ['Date', 'Year']
numeric_columns_boston = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


########################
#### Visualisations ####
########################

# check the distributions for avocado prices
#############################################

def dist_custom(dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols, figsize=(14,14))
    fig.suptitle(suptitle, y=1, size = 25)
    
    axs = axs.flatten()
    
    for i, data in enumerate(columns_list):
        sns.kdeplot(dataset[data], ax=axs[i], fill=True, alpha=.5, linewidth=0)
        axs[i].set_title(data + ', skewness is ' + str(round(dataset[data].skew(axis = 0, skipna=True), 2)))


dist_custom(dataset=raw_df1, columns_list=numeric_columns, rows=3, cols=3, suptitle='Avocado Prices: distribution for each numeric variable')
plt.show()


# check the distributions of boston house prices
################################################

dist_custom(dataset=raw_df2, columns_list=numeric_columns_boston, rows = 4, cols = 3, suptitle="Boston House Prices: distribution for each numeric varable")
plt.show()


#############################
#### Data Pre-Processing ####
#############################

# some data transformations
###########################

#changing data types
for i in raw_df1.columns:
    if i == "Date":
        raw_df1[i] = raw_df1[i].astype('datetime64[ns]')
    elif raw_df1[i].dtype == 'object':
        raw_df1[i] = raw_df1[i].astype('category')

df1 = raw_df1.copy()


df1["Date"] = pd.to_datetime(df1["Date"])
df1["month"] = df1["Date"].dt.month

df1["Spring"] = df1["month"].between(3, 5, inclusive="both")
df1["Summer"] = df1["month"].between(6, 8, inclusive="both")
df1["Fall"] = df1["month"].between(9, 11, inclusive="both")

df1.Spring = df1.Spring.replace({True: 1, False: 0})
df1.Summer = df1.Summer.replace({True: 1, False: 0})
df1.Fall = df1.Fall.replace({True: 1, False: 0})



#Encode labels for type
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1["type"] = le.fit_transform(df1["type"])



#encoding region with one hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop="first", handle_unknown="ignore")
ohe = pd.get_dummies(data=df1, columns=["region"])

df1 = ohe.drop(["Date", "4046", "4225", "4770", "Small Bags", "Large Bags", "XLarge Bags"], axis = 1)


# Outlier detection and removal
###############################

#Use Tukey's method of IQR to detect and remove outliers

def IQR_method(df, n, features):
    """
    Takes a dataframe and returns an index list corresponding to the observations 
    containing more than n outliers according to the Tukey IQR method.
    """

    outlier_list = []

    for column in features:

        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)

        # 3rd quartile (75%)
        Q3 = np.percentile(df[column], 75)

        IQR = Q3 - Q1

        #outlier step
        outlier_step = 1.5 * IQR

        #determining a list of indices of outliers
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index

        outlier_list.extend(outlier_list_column)

        #selecting observations with more than x outliers
        outlier_list = Counter(outlier_list)
        
        multiple_outliers = list(k for k, v in outlier_list.items() if v > n)
        

        #calculate the number of records below and above lower and upper bound value respectively
        df1 = df[df[column] < Q1 - outlier_step] 
        df2 = df[df[column] > Q3 + outlier_step]
        

        print("Total number of deleted outliers: ", df1.shape[0] + df2.shape[0])

        return multiple_outliers



numeric_columns2 = ["Total Volume", "Total Bags"]
Outliers_IQR = IQR_method(df1, 1, numeric_columns2)

df1 = df1.drop(Outliers_IQR, axis=0, ).reset_index(drop = True)


numeric_columns2 = ['CRIM', 'ZN', 'NOX', 'RM', 'AGE', 'DIS', 'PTRATIO', 'B', 'LSTAT']
Outliers_IQR = IQR_method(raw_df2, 1, numeric_columns2)

df2 = raw_df2.drop(Outliers_IQR, axis = 0).reset_index(drop = True)


# Train and Test Split
######################


X = df1.drop("AveragePrice", axis=1)
y = df1["AveragePrice"]

X2 = raw_df2.iloc[:, :-1]
y2 = raw_df2.iloc[:, -1]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)


# Feature Scaling
##################


from sklearn.preprocessing import StandardScaler

def Standard_Scaler(df, col_names):
    features = df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df[col_names] = features

    return df


col_names = ["Total Volume", "Total Bags"]
X_train = Standard_Scaler(X_train, col_names)
X_test = Standard_Scaler(X_test, col_names)

col_names = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X2_train = Standard_Scaler(X2_train, col_names)
X2_test = Standard_Scaler(X2_test, col_names)






