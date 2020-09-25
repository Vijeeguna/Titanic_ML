
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Reading data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train.head())
print(df_train.info())
# Age has some missing value
# Cabin has a lot of missing values
# Embarked few missing values

# Plot for missing data
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df_train.isnull())
plt.show()

# Data Wrangling
# Feature engineering
df_train['Alone'] = 0
df_train['Alone'] = np.where((df_train['Parch'] == 0) & (df_train['SibSp'] == 0), 1, 0)
# sns.barplot('Alone', 'Survived', data=df_train)
# plt.show()
# people travelling with family had a higher survival rate than people travelling alone #

# title = ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Mme', 'Major', 'Mlle', 'Lady', 'Sir', 'Capt', 'Don', 'Rev', 'Col', 'Countess', 'Jonkheer']
df_train['title_temp'] = [x.split('.')[0] for x in df_train['Name']]
df_train['title'] = [x.split(', ')[1] for x in df_train['title_temp']]
del df_train['title_temp']
sns.barplot('title', 'Survived', data=df_train)
plt.show()
# One again children and women hold higher chances of survival with Sir having the lowest survival rate

sns.factorplot('title', 'Survived', col='Pclass', kind='bar', data=df_train)
plt.show()
# Naturally, most high ranking titles traveller in first class

df_train['Fare_bins'] = pd.cut(df_train['Fare'], bins=[-1, 30, 70, 550])
sns.factorplot('Fare_bins', 'Survived', data=df_train, col='Pclass', kind='bar')
plt.show()

# For Title
df_test['title_temp'] = [x.split('.')[0] for x in df_test['Name']]
df_test['title'] = [x.split(', ')[1] for x in df_test['title_temp']]
del df_test['title_temp']

# Filling age missing data
for df in [df_train, df_test]:
    title_arr = df['title'].unique()
    for title in title_arr:
        df.loc[(df['Age'].isnull()) & (df['title'] == title), 'Age'] = df.Age[df['title'] == title].mean()

# Age as a factor
df_train['Age_bins'] = pd.cut(df_train['Age'], bins=[0, 15, 40, 60, 100])

# Filling missing values for embarked
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode().iloc[0])
# Mirroring the data wrangling for test set as well

# For Embarked
df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode().iloc[0])
# For Alone
df_test['Alone'] = 0
df_test['Alone'] = np.where((df_test['Parch'] == 0) & (df_test['SibSp'] == 0), 1, 0)

# For Fare bins
for i in range(1, 4, 1):
    df_test.loc[(df_test['Fare'].isnull()) & (df_test['Pclass'] == i), 'Fare'] = df_test.Fare[
        df_test['Pclass'] == i].mean()
df_test['Fare_bins'] = pd.cut(df_test['Fare'], bins=[-1, 30, 70, 550])

# Age as a factor
df_test['Age_bins'] = pd.cut(df_test['Age'], bins=[0, 15, 40, 60, 100])

# Size of the family as a feature
df_train['Famsize'] = df_train['SibSp'] + df_train['Parch'] + 1
sns.factorplot('Famsize', 'Survived', kind='bar', data=df_train, col='Sex')
plt.show()

# this shows that family size, when considerably large, reduces the chances of survival
# but the chances of survival is large especially for when family size is between 2 - 4
# Men without family on board see the lowest chances of survival

# and women with very large family size exceeding 5, see the lowest chances of survival

# Creating Famsize feature for test data
df_test['Famsize'] = df_test['SibSp'] + df_test['Parch'] + 1

# Converting categorical data to numerical data
df_train['Age_bins'] = df_train['Age_bins'].astype('category').cat.codes
df_test['Age_bins'] = df_test['Age_bins'].astype('category').cat.codes

# Convert Fare bins to numerical data#

df_train['Fare_bins'] = df_train['Fare_bins'].astype('category').cat.codes
df_test['Fare_bins'] = df_test['Fare_bins'].astype('category').cat.codes
# There are different ways to encoding categorical variables to numerical features
# get_dummies is one way (unique column for each lable; codes in 0s and 1s)
# cat.codes is another (single column; unique numerical values for the categories)
# cat.codes will interpret the codes as having order green better than red etc
# dummy variables are not interpreted to have order; since only binary
# so context of the data is important when choosing

df_train['Embarked'] = df_train['Embarked'].astype('category').cat.codes
df_test['Embarked'] = df_test['Embarked'].astype('category').cat.codes
for df in [df_train, df_test]:
    df['title'] = df['title'].map({'Mr': 0,
                                   'Mrs': 1,
                                   'Miss': 2,
                                   'Master': 3,
                                   'Don': 4,
                                   'Rev': 5,
                                   'Dr': 6,
                                   'Mme': 7,
                                   'Ms': 8,
                                   'Major': 9,
                                   'Lady': 10,
                                   'Sir': 11,
                                   'Mlle': 12,
                                   'Col': 13,
                                   'Capt': 14,
                                   'the Countess': 15,
                                   'Jonkheer': 16})

df_test.loc[(df_test['title'].isnull()) & (df_test['Sex'] == 'male'), 'title'] = 0
df_test.loc[(df_test['title'].isnull()) & (df_test['Sex'] == 'female'), 'title'] = 8
df_train['title'] = df_train['title'].astype(int)
df_test['title'] = df_test['title'].astype(int)
for i in range(0, 16, 1):
    df_test.loc[(df_test['Age'].isnull()) & (df_test['title'] == i), 'Age'] = df_test.Age[df['title'] == i].mean()

# plotting for missing data
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df_test.isnull())
plt.show()
df_train['Sex'] = df_train['Sex'].astype('category').cat.codes
df_test['Sex'] = df_test['Sex'].astype('category').cat.codes
print(df_train.info())
print(df_test.info())
# Preprocessing data:

df_train_ml = df_train.copy()
df_test_ml = df_test.copy()
for df in [df_train_ml, df_test_ml]:
    df.drop(['SibSp'], axis=1, inplace=True)
    df.drop(['Parch'], axis=1, inplace=True)
    df.drop(['Alone'], axis=1, inplace=True)
    df.drop(['Name'], axis=1, inplace=True)
    df.drop(['Cabin'], axis=1, inplace=True)
    df.drop(['PassengerId'], axis=1, inplace=True)
    df.drop(['Ticket'], axis=1, inplace=True)

print('Check here:')
print(df_train_ml.info())
print(df_test_ml.info())

# StandardScaler - does feature scaling
standardscaler = StandardScaler()
df_train_ml_sc = standardscaler.fit_transform(df_train_ml)
df_train_ml_sc = pd.DataFrame(df_train_ml_sc, columns=df_train_ml.columns, dtype=int)
# StandardScaler strips column headers.
df_test_ml_sc = standardscaler.fit_transform(df_test_ml)
# Columns attributes adds the column name of the features back to the dataframe
df_test_ml_sc = pd.DataFrame(df_test_ml_sc, columns=df_test_ml.columns, dtype=int)
print(df_train_ml_sc.info())
print(df_test_ml_sc.info())

# Test and Train data split
X_test = df_test_ml
X_test_scaled = df_test_ml_sc

X_train = df_train_ml.drop(['Survived'], axis=1)
X_train_scaled = df_train_ml_sc.drop(['Survived'], axis=1)
Y_train = df_train_ml['Survived']
Y_train_scaled = df_train_ml_sc['Survived']

# Part 3
# C is the soft margin and Gamma parameter determines the influence a single data sample exerts on the decision boundary
# Read more on this

def feature_importance_plot(model, columns):
    plt.figure(figsize=(7, 7))
    plt.title('Feature Importance')
    features = pd.Series(data=model.best_estimator_.feature_importances_, index=columns).sort_values(ascending=False)
    ax = sns.barplot(x=features.values[:8], y=features.index[:8])
    plt.show()

# Support Vector Classifier 
svc = SVC(gamma=0.01, C=100)
svc_scores = cross_val_score(svc, X_train, Y_train, scoring='accuracy')
# Now, using scaled featrues to check if the accurcy improves
svc_sc = SVC(gamma=0.01, C=100)
svc_scores_sc = cross_val_score(svc_sc, X_train_scaled, Y_train_scaled, scoring='accuracy')

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc_scores = cross_val_score(rfc, X_train, Y_train, cv=10, scoring='accuracy')

# Random Grid Search
model = SVC()
param_grid = {'C': uniform(0.1, 5000), 'gamma': uniform(0.0001, 1)}
rand = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100, random_state=0)
rand.fit(X_train_scaled, Y_train_scaled)
rand_scores_svc = rand.best_score_

# Using Grid Search
grid_svc = GridSearchCV(model, param_grid={'C': [0.1, 10, 100, 1000, 5000], 'gamma': [1, 0.1, 0.001, 0.0001]}, cv=10)
grid_svc.fit(X_train_scaled, Y_train_scaled)
grid_scores_svc = grid_svc.best_score_

# Gradient Boosting:
GaBooC = GradientBoostingClassifier(random_state=0)
gabooc_param = {'n_estimators': [50, 100],
              'min_samples_split': [3, 4, 5, 6, 7],
              'max_depth': [3, 4, 5, 6]}

gabooc_grid = GridSearchCV(GaBooC, param_grid=gabooc_param, cv = 10)
gabooc_scores = gabooc_grid.fit(X_train_scaled, Y_train_scaled)
feature_importance_plot(gabooc_grid, X_train_scaled.columns)

# Extreme Gradient Boosting
xgb = XGBClassifier(random_state=0)
xgb_param = {'max_depth': [5,6,7,8], 'gamma': [1, 2, 3, 4], 'learning_rate': [0.1, 0.2, 0.3, 0.5]}
xgb_grid = GridSearchCV(xgb, param_grid=xgb_param, cv = 10)
xgb.scores  = xgb_grid.fit(X_train_scaled, Y_train_scaled)
feature_importance_plot(xgb_grid, X_train_scaled.columns)
print(xgb.scores.best_score_)

#AdaBoost
ada = AdaBoostClassifier(random_state=0)
ada_param = {'n_estimators': [30, 50, 100],
             'learning_rate': [0.001, 0.1, 1]}
ada_grid = GridSearchCV(ada, param_grid = ada_param, cv = 10)
ada_scores = ada_grid.fit(X_train_scaled, Y_train_scaled)
print(ada_scores.best_score_)
feature_importance_plot(ada_grid, X_train_scaled.columns)

#CatBoost
cat = CatBoostClassifier(random_state=0)
cat_param = {'iterations': [100, 150], 'learning_rate': [0.1, 0.2, 0.3, 0.5], 'loss_function': ['Logloss']}
cat_grid = GridSearchCV(cat, param_grid = cat_param, cv = 10)
cat_scores = cat_grid.fit(X_train_scaled, Y_train_scaled)
print(cat_scores.best_score_)
feature_importance_plot(cat_grid, X_train_scaled.columns)

#Voting Classifier
vce1 = LogisticRegression(random_state=0)
vce2 = RandomForestClassifier(random_state=0)
vce3 = GaussianNB()

vc = VotingClassifier(estimators=[('lr', vce1), ('rf', vce2), ('nb', vce3)], voting = 'soft')
vc_params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}
vc_grid = GridSearchCV(vc, param_grid= vc_params, cv = 10)
vc_scores = vc_grid.fit(X_train_scaled, Y_train_scaled)
print(vc_scores.best_score_)

#Voting classifier - 2
vce4 = RandomForestClassifier(random_state=0)
vce5 = SVC()
vce6 = GradientBoostingClassifier(random_state=0)
vc2 = VotingClassifier(estimators=[('rf', vce4), ('svc', vce5), ('gbdt', vce6)], voting='soft')
vc2_params = {'gbdt__n_estimators': [50], 'gbdt__min_samples_split': [3],
          'svc__C': [10, 100] , 'svc__gamma': [0.1,0.01,0.001] , 'svc__kernel': ['rbf'] , 'svc__probability': [True],
          'rf__max_depth': [7], 'rf__max_features': [2,3], 'rf__min_samples_split': [3] }
vc2_grid = GridSearchCV(vc2, param_grid=vc2_params, cv=10)
vc2_scores = vc2_grid.fit(X_train_scaled, Y_train_scaled)
print(vc2_scores.best_score_)

#Stacking Classifier
s1 = xgb_grid.best_estimator_
s2 = ada_grid.best_estimator_
s3 = cat_grid.best_estimator_
s4 = gabooc_grid.best_estimator_

lr = LogisticRegression()
sc = StackingClassifier(classifiers=[s1,s2,s3,s4], meta_classifier=lr)
sc_param = {'meta_classifier__C': [0.1,1.0,5.0,10.0] ,
          'use_features_in_secondary' : [True, False]
         }
sc_grid = GridSearchCV(sc, param_grid=sc_param, cv=10)
sc_score = sc_grid.fit(X_train_scaled, Y_train_scaled)
print(sc_score.best_score_)



