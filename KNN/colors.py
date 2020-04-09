import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#loading colors dataset
colors=pd.read_csv('/kaggle/input/simple-colors-dataset/colors.csv')
#displaying the dataset
print(colors.head())

#checking for null values
print(colors.isnull().sum())

#encoding categorical values
colors["Color"] = colors["Color"].astype('category')
colors["Color"] = colors["Color"].cat.codes
colors.head()

#importing scikit learn library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Shuffling data

colors=colors.sample(frac=1).reset_index(drop=True)

#converting and splitting data
X=colors[['X','Y']]
y=colors['Color']
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=100,test_size=50)

print(X_train)
print(y_train)

#defining model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

#checking its accuracy
print(model.score(X_train,y_train))

def output_class(y):
    if y == 0:
        return "Blue"
    elif y == 1:
        return "Red"
    else:
        return "Green"

#predicting the output    
y_pred=model.predict([[163,43]])
print(output_class(y_pred))

#prediciting test output
y_pred=model.predict(X_test[1:2])
print(output_class(y_pred))

# test accuracy
print(model.score(X_test, y_test))
