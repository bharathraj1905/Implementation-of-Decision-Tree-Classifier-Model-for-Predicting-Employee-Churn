# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data
2. Define your model
3. Define your cost function
4. Define your learning rate
5. Train your model
6. Evaluate your model
7. Tune hyperparameters
8. Deploy your model

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 212222230019
RegisterNumber: B.Barathraj
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

Initial data set:

<img width="507" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343698/d6fe68cd-0da2-42a8-8945-f34ff61302fa">

Data info:

<img width="157" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343698/dd4a1072-af40-4f78-ba1f-fbd23227fefe">

Optimization of null values:

<img width="143" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343698/a287ab5c-4b2b-442d-88d6-b7fba0758842">

Assignment of x and y values:

<img width="173" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343698/bdb6b582-fbc7-4181-98e9-b780127d4f80">


<img width="677" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343698/0e198d4d-310e-4365-a92a-7b541ab12fdb">

Converting string literals to numerical values using label encoder:

<img width="594" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343698/bce07267-bbc0-4b53-8d3f-9c5fc89f5ec3">

Accuracy:

<img width="124" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343698/5aa80a63-06fd-4b19-88c1-7fbed050b046">

Prediction:

<img width="673" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343698/b6587899-5d6b-4888-bfe8-b398f6d4a526">

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
