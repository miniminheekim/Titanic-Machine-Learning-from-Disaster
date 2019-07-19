#%%importing packages
#name package 'pandas' as 'pd'
import pandas as pd
import warnings

#load 'filterwarnings'function from package 'warnings' and ignore all the warnings
warnings.filterwarnings('ignore') 

import seaborn as sns
import matplotlib.pyplot as plt

#load 'style.use' function from plt(=matplotlib.pyplot)
plt.style.use('fivethirtyeight') 

import numpy as np

#%%
#EDA (Exploratory Data Analysis) of Tytanic Data

#%%load dataset
#pd.read_csv() is from pandas to read csv files
train=pd.read_csv("c:/ds_study/2019titanic/train.csv")
train

#%%
#showthe first 5 data from train
train.head()
#show how many rows and colums in data 'train'
train.shape

#%%
#show 'Survived' colum from dataset 'train'
train['Survived']

#count values
train['Survived'].value_counts()

#draw bar chart or countplot
#sns(seaborn).countplot('colum name',data=dataframename)
sns.countplot('Survived',data=train)

#compute mean=how much percent survived
train['Survived'].mean()

#%%
train['Sex'].value_counts()
sns.countplot('Sex',data=train)
#%%groupby
#dataframe.groupby('colname')['colname'].mean()
#want to know survival percentage for each sex
train.groupby('Sex')['Survived'].mean()

train.groupby('Sex')['Survived'].mean().plot.bar()
#%%groupby.pclass
train['Pclass'].value_counts()
sns.countplot('Pclass',data=train)
train.groupby('Pclass')['Survived'].mean()
#%%
train.groupby('Pclass')['Survived'].mean().plot.bar()

#%%groupby.embarked
train['Embarked'].value_counts()
sns.countplot('Embarked',data=train)
train.groupby('Embarked')['Survived'].mean()
#%%
train.groupby('Embarked')['Survived'].mean().plot.bar()

#%%Age
#when value_counts doesnt seem to work->describe
train['Age'].describe()
train['Age'].plot.hist()
#%%Age kde plot (kernel density plot)
sns.kdeplot(train['Age'])
#히스토그램을 연속화 시킨 것 (연속함수로 근사화시켜주는 것)
#%%특정 값 뽑아내기
train.head()
train.loc[train['Survived']==0,'Age']
#data.loc[행,[열1,열2]]
#train.loc[train['Pclass']==2,'Survived']
#train.loc[train['Age']<30,'Sex']
sns.kdeplot(train.loc[train['Survived']==0,'Age'],label='died')
sns.kdeplot(train.loc[train['Survived']==1,'Age'],label='survived')
#두 행을 이렇게 하면 그래프가 겹쳐진다.

#%%SibSp 관찰
train['SibSp'].value_counts()
train['SibSp'].describe()
train.groupby('SibSp')['Survived'].mean()
train.groupby('SibSp')['Survived'].mean().plot()
#survival rate decreases as SibSp is higher
#%%Fare
#sns.kdeplot(train['Fare'])
sns.kdeplot(train.loc[train['Survived']==0,'Fare'],label='died')
sns.kdeplot(train.loc[train['Survived']==1,'Fare'],label='survived')


#%%
train.columns
#%%
x = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = train['Survived']
#%%
x['Age'].isnull().sum()
x['Age'] = x['Age'].fillna(-1)
x['Age'].isnull().sum()
#%%
x['Pclass'].value_counts()
x['Pclass'].isnull().sum()
#%%
x['Sex'] = x['Sex'].map({'female':0,'male':1})
x['Sex'].head()


#%%
x['Age'].isnull().sum()
x['Age'].isnull().mean()


#%%
x['SibSp'].isnull().sum()


#%%
x['Embarked'] = x['Embarked'].map({'S':0,'C':1,'Q':2})
x['Embarked'].head()


#%%
x['Embarked'] = x['Embarked'].fillna(-1)
#%%
x['Parch'].isnull().sum()

#%%
x['Fare'].isnull().sum()

#%%
x.isnull().sum()
#%%
from sklearn import tree
#%%
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(x,y)
#%%
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image
dot_data = export_graphviz(clf,out_file = None, feature_names = x.columns, 
            class_names=['death','survived'],filled=True, rounded=True,special_characters=True  )
graph = pydot.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png('titanic_decision_tree.png')


# x = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
# y = train['Survived']

#%%
clf.predict_proba([[3,1,20,0,0,7.25,0],[1,0,20,0,1,75.2,0]])[:,1]


#%%
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

#%%
x_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#%%
x_test.head()


#%%
x_test.isnull().sum()


#%%
x_test['Pclass'].isnull().sum()


#%%
x_test['Sex'] = x_test['Sex'].fillna(-1)
x_test['Sex'] = x_test['Sex'].map({'female':0,'male':1})
x_test['Sex'].head()


#%%
x_test['Age'].isnull().sum()
x_test['Age'] = x_test['Age'].fillna(-1)
x_test['Age'].isnull().sum()


#%%
x_test['SibSp'].isnull().sum()


#%%
x_test['Fare'] = x_test['Fare'].fillna(-1)
x_test['Fare'].isnull().sum()


#%%
x_test['Parch'].isnull().sum()


#%%
x_test.head()


#%%
x_test['Embarked'] = x_test['Embarked'].map({'S':0, 'C':1, 'Q':2})
x_test['Embarked'].head()


#%%
x_test.isnull().sum()


#%%
x_test.head()


#%%
test['Survived'] = clf.predict_proba(x_test)[:,1]


#%%
test.head()


#%%
test[['PassengerId','Survived']].to_csv("dacon_baseline.csv",index=False)
