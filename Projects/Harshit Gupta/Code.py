# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# # Data Preproessing

# %% [code]
tr = pd.read_csv('/kaggle/input/titanic/train.csv')
te = pd.read_csv('/kaggle/input/titanic/test.csv')

# %% [code]
tr.columns

# %% [code]
main = pd.concat([tr.drop('Survived', axis = 1), te])

# %% [code]
main

# %% [code]
relatives = pd.Series(main['Parch'] + main['SibSp'], name = 'Relatives')

# %% [code]
main = pd.concat([main[['Name', 'Pclass', 'Sex', 'Age','Embarked']], relatives], axis = 1)

# %% [code]
def ext(name):
    name = name.split(',')[1].split()
    for i in name:
        if i[len(i)-1] == '.':
            return i

# %% [code]
main['Name'] = main['Name'].apply(ext)

# %% [code]
main.Name.value_counts()

# %% [code]
mean = main['Age'].mean()
std = main['Age'].std()

# %% [code]
def check(age):
    if age <= 13:
        return 1
    elif age > 13 and age <= 20:
        return 2
    elif age > 20 and age <= 30:
        return 3
    elif age > 30 and age <= 40:
        return 4
    elif age > 40 and age <= 50:
        return 5
    elif age > 50:
        return 6
    elif pd.isnull(age):
        return check(np.random.randint(mean - std, mean + std, 1)[0])

# %% [code]
main['Age'] = main['Age'].apply(check)

# %% [code]
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
main['Name']= label_encoder.fit_transform(main['Name'])

# %% [code]
main.Name.value_counts()

# %% [code]
main['Name'] = main['Name'].apply(lambda x : x if(x == 13 or x == 10 or x == 14 or x == 9) else 0)

# %% [code]
main.Name.unique()

# %% [code]
main.Sex = main.Sex.map({'female':1, 'male': 0})

# %% [code]
main.Embarked = main.Embarked.map({'S':1, 'C':2, 'Q':3})

# %% [code]
main.Embarked.fillna(int(main.Embarked.mode()[0]), inplace = True)

# %% [markdown]
# # Exploratory data analysis

# %% [code]
temp = pd.concat([main[:len(tr)], tr['Survived']], axis = 1)

# %% [code]
corr_age = temp.corr()

# %% [markdown]
# # Visualisation of data

# %% [code]
fig, axes = plt.subplots(figsize=(20,10))
sns.heatmap(corr_age, annot = True, linewidth = 1)
plt.show()

# %% [markdown]
# this corelation matrix shows us that Name, Embarked, Fare, Sex, Pclass are of high importances

# %% [markdown]
# Let's See some visualisation to see thier results

# %% [code]
sns.barplot(data = temp , x = 'Pclass', y = 'Survived')
plt.title('Survived VS Pclass WRT Titles')
plt.show()

# %% [markdown]
# Here we can see that higher the Pclass, Higher the Survival Chances

# %% [code]
sns.barplot(data = temp, x = 'Relatives', y = 'Survived')
plt.show()

# %% [markdown]
# No. of relatives have very high varince so it is not reliable feature.

# %% [code]
sns.barplot(data = temp, y = 'Survived', x = 'Embarked', hue = 'Sex')
plt.show()

# %% [markdown]
# Here we can see that passengers with embarked = 2 have more chances of survival

# %% [code]
sns.barplot(data = temp, x = 'Pclass', y = 'Survived', hue = 'Sex')
plt.show()

# %% [code]
sns.pointplot(data = temp, x='Age', y = 'Survived', hue = 'Sex')
plt.show()

# %% [markdown]
# Here we can see that their priority was Female and childrens but else than that, it doesn't show any special corealtions

# %% [code]
sns.barplot(data = temp, x='Name', y = 'Survived')
plt.show()

# %% [code]
main = main[['Name', 'Pclass', 'Sex', 'Age','Embarked']]

# %% [code]
main.isnull().sum()

# %% [code]
main.drop('Age',axis = 1, inplace = True)

# %% [code]
len(tr)

# %% [code]
X_train = main.iloc[:len(tr), :].copy()
X_test = main.iloc[len(tr):, :].copy()
Y_train = tr['Survived'].copy()

# %% [code]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %% [markdown]
# # Model Selection

# %% [code]
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1

# %% [markdown]
# 1) Logistic Regression

# %% [code]
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, Y_train)

# %% [code]
from sklearn.model_selection import cross_val_score
accuracies_LR = cross_val_score(estimator = classifier_LR, X = X_train, y = Y_train, cv = 10)

# %% [code]
print('K-Fold Cross Validation accuracy with 10 Folds is', accuracies_LR.mean()*100,'% And it\'s Standard Deviation is',  accuracies_LR.std()*100,'%')
Y_tr_pred = classifier_RF.predict(X_train)
print('Accuracy on training Set', acc(Y_train, Y_tr_pred)*100,'%')

# %% [markdown]
# 2) K Nearest Neighbors

# %% [code]
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train,Y_train)

# %% [code]
from sklearn.model_selection import cross_val_score
accuracies_KNN = cross_val_score(estimator = classifier_KNN, X = X_train, y = Y_train, cv = 10)

# %% [code]
print('K-Fold Cross Validation accuracy with 10 Folds is', accuracies_KNN.mean()*100,'% And it\'s Standard Deviation is',  accuracies_KNN.std()*100,'%')
Y_tr_pred = classifier_RF.predict(X_train)
print('Accuracy on training Set', acc(Y_train, Y_tr_pred)*100,'%')

# %% [markdown]
# 3) Support Vector Classifier

# %% [code]
from sklearn.svm import SVC
classifier_SVC = SVC(kernel = 'rbf')
classifier_SVC.fit(X_train, Y_train)

# %% [code]
from sklearn.model_selection import cross_val_score
accuracies_SVC = cross_val_score(estimator = classifier_SVC, X = X_train, y = Y_train, cv = 10)

# %% [code]
print('K-Fold Cross Validation accuracy with 10 Folds is', accuracies_SVC.mean()*100,'% And it\'s Standard Deviation is',  accuracies_SVC.std()*100,'%')
Y_tr_pred = classifier_RF.predict(X_train)
print('Accuracy on training Set', acc(Y_train, Y_tr_pred)*100,'%')

# %% [markdown]
# 4) Naive Bayes

# %% [code]
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, Y_train)

# %% [code]
from sklearn.model_selection import cross_val_score
accuracies_NB = cross_val_score(estimator = classifier_NB, X = X_train, y = Y_train, cv = 10)

# %% [code]
print('K-Fold Cross Validation accuracy with 10 Folds is', accuracies_NB.mean()*100,'% And it\'s Standard Deviation is',  accuracies_NB.std()*100,'%')
Y_tr_pred = classifier_RF.predict(X_train)
print('Accuracy on training Set', acc(Y_train, Y_tr_pred)*100,'%')

# %% [markdown]
# 5) Decision Tree

# %% [code]
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DT.fit(X_train, Y_train)

# %% [code]
from sklearn.model_selection import cross_val_score
accuracies_DT = cross_val_score(estimator = classifier_DT, X = X_train, y = Y_train, cv = 10)

# %% [code]
print('K-Fold Cross Validation accuracy with 10 Folds is', accuracies_DT.mean()*100,'% And it\'s Standard Deviation is',  accuracies_DT.std()*100,'%')
Y_tr_pred = classifier_RF.predict(X_train)
print('Accuracy on training Set', acc(Y_train, Y_tr_pred)*100,'%')

# %% [markdown]
# 6) Random Forest

# %% [code]
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 350, criterion = 'entropy',oob_score = True)
classifier_RF.fit(X_train, Y_train)

# %% [code]
from sklearn.model_selection import cross_val_score
accuracies_RF = cross_val_score(estimator = classifier_RF, X = X_train, y = Y_train, cv = 10)

# %% [code]
print('K-Fold Cross Validation accuracy with 10 Folds is', accuracies_RF.mean()*100,'% And it\'s Standard Deviation is',  accuracies_RF.std()*100,'%')
Y_tr_pred = classifier_RF.predict(X_train)
print('Accuracy on training Set', acc(Y_train, Y_tr_pred)*100,'%')

# %% [code]
Y_pred = classifier_RF.predict(X_test)

# %% [code]
output = pd.DataFrame({'PassengerId': te.PassengerId, 'Survived': Y_pred})
output.to_csv('my_submission.csv', index=False)