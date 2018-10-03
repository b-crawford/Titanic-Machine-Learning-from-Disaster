import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import math
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
import os

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

len(train_df)
len(test_df)

# All data:

all_data = pd.concat([train_df,test_df], ignore_index=False, sort = False)
all_data.isnull().sum() # So there are 418 Null values for survived which is the test data


# Extract features:
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1: # string.find returns -1 if substring not in big_string
                                                    # so string.find is not equal to -1 if substring IS in big_string
            return substring
    # print big_string # only prints the name if string.find is not -1 so if substring is not in big_string
    return np.nan

title_list = ['Mr','Mrs','Dr','Miss','Master','Rev','Major','Don','Mme','Ms','Col','Capt','Countess',
                               'Jonkheer','Mlle']

for name in train_df.Name:
    substrings_in_string(name,title_list)


# Replace all titles with Mrs, Miss, Mr and Master:
all_data['Title'] = all_data['Name'].map(lambda x:substrings_in_string(x,title_list))


def replace_titles(x):
    title = x['Title']
    if title in ['Rev','Major','Don','Col','Capt','Jonkheer']:
        return 'Mr'
    if title in ['Mme','Countess']:
        return 'Mrs'
    if title in ['Mlle','Ms']:
        return 'Miss'
    if title == 'Dr':
        if x['Sex'] == 'male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


all_data['Title'] = all_data.apply(replace_titles, axis = 1)





# Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

all_data['CabinStr'] = all_data['Cabin'].astype(str)

all_data['Deck'] = all_data['CabinStr'].map(lambda x: substrings_in_string(x, cabin_list))


# Family size:

all_data['Family_Size'] = all_data['SibSp'] + all_data['Parch']

# Age*Class:

all_data['Age*Class'] = all_data['Age'] * all_data['Pclass']

# Far per person:

all_data['Fare_Per_Person'] = all_data['Fare'] / (all_data['Family_Size']+1)


all_data.head(5)

all_data.describe()

all_data.describe(include=['O'])

corr = all_data.corr()
sns.heatmap(corr)

sns.FacetGrid(all_data, col = 'Survived').map(plt.hist,'Age')


sns.violinplot(x = 'Survived',y='Age',data=all_data)


# Young people survived more and old people less

all_data.isnull().mean()


# Cabin has too many null vaues, drop it
all_data = all_data.drop(['Cabin'], axis =1)

# Fill in fare with mean:
fare_mean = all_data['Fare'].mean()
nan_count = all_data['Fare'].isnull().sum()
all_data.loc[np.isnan(all_data['Fare']),'Fare'] = [fare_mean] * nan_count
all_data['Fare_Per_Person'] = all_data['Fare'] / (all_data['Family_Size']+1)



# Make catergorical one hot:
all_data = pd.get_dummies(all_data, columns=['Sex','Deck',
                                             'Embarked','Title'],
                           drop_first=False)

all_data.isnull().mean()


# Create a model to fill in Age
X_train = all_data.iloc[1-np.isnan(all_data['Age'])].drop(['Survived','Age','Age*Class','PassengerId','Name','Ticket','CabinStr'] , axis =1)
X_test = all_data[all_data['Age'].isnull()].drop(['Survived','Age','Age*Class','PassengerId','Name','Ticket','CabinStr'] , axis =1)

Y_train = all_data.iloc[1-np.isnan(all_data['Age'])]['Age']

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

# Add in fitted values for Age
all_data.loc[all_data['Age'].isnull(),'Age'] = regr.predict(X_test)
all_data['Age*Class'] = all_data['Age'] * all_data['Pclass']
all_data = pd.get_dummies(all_data, columns=['Pclass'], drop_first=False)

all_data.head()
all_data = all_data.drop(['Ticket','Name','CabinStr'], axis = 1)

all_data['Age'].max()
all_data['Age'].min()

# Split train data into train and validation
train_df = all_data[all_data['Survived'].notnull()]
train_df.shape
validation_indices = np.random.randint(0,len(train_df),size = math.floor(0.2*len(train_df)))
validation_df = train_df[train_df.index.isin(validation_indices)]
validation_df.shape
train_df = train_df[~ train_df.index.isin(validation_indices)]
train_df.shape

test_df = all_data[all_data['Survived'].isnull()]


X_train = train_df.drop(['Survived'],axis = 1)
Y_train = train_df['Survived']

X_val = validation_df.drop(['Survived'],axis = 1)
Y_val = validation_df['Survived']

X_test = test_df.drop(['Survived'],axis = 1)


# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# create dict to hold validation accuracies of each technique
accuracies = {}


# Logistic regression model:
logisticRegr = linear_model.LogisticRegression()
logisticRegr.fit(X_train,Y_train )

accuracies['logistic'] = 1 - np.sum((logisticRegr.predict(X_val) - Y_val)**2) / len(validation_df)
accuracies


# Support vector machine
SVMachine = svm.SVC()

param_grid = {'C': [1,2, 5, 10,20, 50],
              'gamma': [0.0001, 0.0005, 0.001, 0.005]}

svm_CV = GridSearchCV(SVMachine, param_grid, cv=5)

svm_CV.fit(X_train, Y_train)

accuracies['SVM'] = 1- np.sum((svm_CV.predict(X_val) - Y_val)**2) / len(validation_df)
accuracies



# Nearest neighbour
param_grid = {'n_neighbors': [1,2,3,4, 5, 10]}

classifier = KNeighborsClassifier()

nn_CV = GridSearchCV(classifier, param_grid, cv=5)

nn_CV.fit(X_train, Y_train)

accuracies['kNN'] = 1- np.sum((nn_CV.predict(X_val) - Y_val)**2) / len(validation_df)
accuracies


# Neural network

X_train.shape
Y_train.shape

NN_model = keras.Sequential([
    keras.layers.Dense(28, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

NN_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

NN_model.fit(np.array(X_train), np.array(Y_train), epochs=5)

val_loss, val_acc = NN_model.evaluate(np.array(X_val), np.array(Y_val))

accuracies['NeuralNet'] = val_acc
accuracies

# Radndom forest
rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train, Y_train)

accuracies['RandomForest'] = 1- np.sum((rf.predict(X_val) - Y_val)**2) / len(validation_df)
accuracies


# Random Forest predicitons as input to Neural network
X_train_new

X_train_new = pd.concat([pd.DataFrame(X_train),pd.Series(rf.predict(X_test)).reset_index()], axis=1,ignore_index = True).drop([28], axis = 1)
X_val_new = pd.concat([pd.DataFrame(X_val),pd.Series(rf.predict(X_val)).reset_index()], axis=1,ignore_index = True).drop([28], axis = 1)
X_test_new = pd.concat([pd.DataFrame(X_test),pd.Series(rf.predict(X_test)).reset_index()], axis=1,ignore_index = True).drop([28], axis = 1)


NN_model2 = keras.Sequential([
    keras.layers.Dense(29, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

NN_model2.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

NN_model2.fit(np.array(X_train_new), np.array(Y_train), epochs=20)

val_loss, val_acc = NN_model2.evaluate(np.array(X_val_new), np.array(Y_val))

accuracies['RF+NeuralNet'] = val_acc
accuracies


# Choose Random Fofest model
Y_test = rf.predict(X_test)

output = pd.concat([test_df['PassengerId'],pd.to_numeric(pd.Series(Y_test),downcast='integer')], axis = 1)

output.head()

output.columns = ['PassengerId','Survived']

output.to_csv('output.csv',index=False)
