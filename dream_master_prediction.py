
#%% introduction
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns


#%% Understanding Data
dataframe=pd.read_csv("Admission_Predict.csv")

#print(dataframe.head(10))
columns=(dataframe.columns)  #columns
isnull=dataframe.isnull().sum()           #there is no null value
describe= dataframe.describe().T                    


def detect_outliers(df,n,features):  #for outlier detection
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

outliers_to_drop=detect_outliers(dataframe,2,['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research'])
#to detect outliers from the numerical values features (GRE Score, TOEFL Score, University Rating, SOP, LOR , CGPA, Research)

#there are no outliers

dataframe.drop(labels='Serial No.',axis=1,inplace=True)
#drop serial number row


#%% Data Analysis
corr=dataframe.corr()
sns.heatmap(corr,annot=True,linewidths=.8,cmap="hsv",fmt='0.2f')

#Here we can see that the chance of admit is highly correlated with CGPA, GRE and TOEFEL scores are also correlated

sns.pairplot(dataframe)

#GRE score TOEFL score and CGPA all are linearly related to each other
#Research Students tend to Score higher by all means

plt.figure(figsize=(6,6))
plt.subplot(2, 1, 1)
sns.distplot(dataframe['GRE Score'])
plt.subplot(2,1,2)
sns.distplot(dataframe['TOEFL Score'])

#From the above 2 graphs its clear that people tend to score above 310 in GRE and above 100 in TOEFL

sns.scatterplot(x='University Rating',y='CGPA',data=dataframe)
#ratings increased with the CGPA

co_gre=dataframe[dataframe["GRE Score"]>=300]
co_toefel=dataframe[dataframe["TOEFL Score"]>=100]

fig, ax = plt.subplots(figsize=(15,8))
sns.barplot(x='GRE Score',y='Chance of Admit ',data=co_gre, linewidth=1.5,edgecolor="0.1")
plt.show()

fig, ax = plt.subplots(figsize=(15,8))
sns.barplot(x='TOEFL Score',y='Chance of Admit ',data=co_toefel, linewidth=3.5,edgecolor="0.8")
plt.show()
#analysing the correlation
#The above two graphs make it clear that higher the Scores better the chance of admit




print("Average GRE Score :{0:.2f} out of 340".format(dataframe['GRE Score'].mean()))
print('Average TOEFL Score:{0:.2f} out of 120'.format(dataframe['TOEFL Score'].mean()))
print('Average CGPA:{0:.2f} out of 10'.format(dataframe['CGPA'].mean()))
print('Average Chance of getting admitted:{0:.2f}%'.format(dataframe['Chance of Admit '].mean()*100))

#chechk out the toppers
toppers=dataframe[(dataframe['GRE Score']>=330) & (dataframe['TOEFL Score']>=115) & (dataframe['CGPA']>=9.5)].sort_values(by=['Chance of Admit '],ascending=False)
toppers


#%% Modelling

X=dataframe.drop('Chance of Admit ',axis=1)
y=dataframe['Chance of Admit ']   #chance of admit

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Normalisation works slightly better for Regression.
X_norm=preprocessing.normalize(X)   #normalize X
X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.20,random_state=101) #split

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,mean_squared_error


regressors=[['Linear Regression :',LinearRegression()],
       ['Decision Tree Regression :',DecisionTreeRegressor()],
       ['Random Forest Regression :',RandomForestRegressor()],
       ['Gradient Boosting Regression :', GradientBoostingRegressor()],
       ['Ada Boosting Regression :',AdaBoostRegressor()],
       ['Extra Tree Regression :', ExtraTreesRegressor()],
       ['K-Neighbors Regression :',KNeighborsRegressor()],
       ['Support Vector Regression :',SVR()]]

reg_pred=[]
print('Results...\n')
for name,model in regressors:
    model=model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    rms=np.sqrt(mean_squared_error(y_test, predictions))
    reg_pred.append(rms)
    print(name,rms)
    
y_ax=['Linear Regression' ,'Decision Tree Regression', 'Random Forest Regression','Gradient Boosting Regression', 'Ada Boosting Regression','Extra Tree Regression' ,'K-Neighbors Regression', 'Support Vector Regression' ]
x_ax=reg_pred

sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.1")

#%%  Classification
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=101)

#If Chance of Admit greater than 80% we classify it as 1
y_train_c = [1 if each > 0.8 else 0 for each in y_train]
y_test_c  = [1 if each > 0.8 else 0 for each in y_test]

classifiers=[['Logistic Regression :',LogisticRegression()],
       ['Decision Tree Classification :',DecisionTreeClassifier()],
       ['Random Forest Classification :',RandomForestClassifier()],
       ['Gradient Boosting Classification :', GradientBoostingClassifier()],
       ['Ada Boosting Classification :',AdaBoostClassifier()],
       ['Extra Tree Classification :', ExtraTreesClassifier()],
       ['K-Neighbors Classification :',KNeighborsClassifier()],
       ['Support Vector Classification :',SVC()],
       ['Gausian Naive Bayes :',GaussianNB()]]
cla_pred=[]
for name,model in classifiers:
    model=model
    model.fit(X_train,y_train_c)
    predictions = model.predict(X_test)
    cla_pred.append(accuracy_score(y_test_c,predictions))
    print(name,accuracy_score(y_test_c,predictions))
    
    
y_ax=['Logistic Regression' ,
      'Decision Tree Classifier',
      'Random Forest Classifier',
      'Gradient Boosting Classifier',
      'Ada Boosting Classifier',
      'Extra Tree Classifier' ,
      'K-Neighbors Classifier',
      'Support Vector Classifier',
       'Gaussian Naive Bayes']
x_ax=cla_pred


sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.8")
plt.xlabel('Accuracy')





















