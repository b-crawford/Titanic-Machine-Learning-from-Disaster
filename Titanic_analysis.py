import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

train_df = pd.read_csv('train.csv')

# Extract features:
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1: # string.find returns -1 if substring not in big_string
                                                    # so string.find is not equal to -1 if substring IS in big_string
            return substring
    # print big_string # only prints the name if string.find is not -1 so if substring is not in big_string
    return np.nan

title_list = ['Mr','Mrs','Dr','Miss','Master','Rev','Major','Don','Mme','Ms','Col','Capt','Countess',
                               'Jonkheer','Mlle']

for name in train_df.Name:
    substrings_in_string(name,title_list)


# Replace all titles with Mrs, Miss, Mr and Master:
train_df['Title'] = train_df['Name'].map(lambda x:substrings_in_string(x,title_list))


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


train_df['Title'] = train_df.apply(replace_titles, axis = 1)



# Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

train_df['CabinStr'] = train_df['Cabin'].astype(str)

train_df['Deck'] = train_df['CabinStr'].map(lambda x: substrings_in_string(x, cabin_list))


# Family size:

train_df['Family_Size'] = train_df['SibSp'] + train_df['Parch']

# Age*Class:

train_df['Age*Class'] = train_df['Age'] * train_df['Pclass']

# Far per person:

train_df['Fare_Per_Person'] = train_df['Fare'] / (train_df['Family_Size']+1)

# Make catergorical one hot:
train_df = pd.get_dummies(train_df, columns=['Pclass', 'Sex','Deck',
                                             'Embarked','Title'],
                           drop_first=False)

train_df.head(5)

train_df.describe()

train_df.describe(include=['O'])

corr = train_df.corr()
sns.heatmap(corr)

sns.FacetGrid(train_df, col = 'Survived').map(plt.hist,'Age')


reg = linear_model.LinearRegression()
reg.fit
