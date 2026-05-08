import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Jagatjyoti\Jagat_Code\08-05-2026\StudentMarkPred\student_info.csv')
df

print(df.head())
df.tail()
df.shape
df.info()
df.describe()
plt.scatter(df['study_hours'], df['student_marks'])
plt.xlabel('Student Study Hours')
plt.ylabel('Student Marks')
plt.title('Scatter Plot of Study Hours vs Student Marks')
plt.show()

#DataCleaning

df.isnull().sum()
df.mean()
df2 =df.fillna(df.mean())   
df2.isnull().sum()

df.dropna(inplace=True)

df2.info()
df.info()

#Spliting    Data
x = df2.drop('student_marks', axis=1)
y = df2.drop('study_hours', axis=1)



print("Shape of X:", x.shape)
print("Shape of Y:", y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print("Shape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# y = m * x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
print("Model Trained Successfully")

y_pred = lr.predict(x_test)
print("Predicted Values:", y_pred)
print("Actual Values:", y_test.values)

from sklearn.metrics import mean_absolute_error

lr.coef_ #m
lr.intercept_ #c

M=3.93
C=50.44
Y = M* 10 + C
print("Manual Calculation:", Y)

lr.predict([[11]]).round(2)

y_pred = lr.predict(x_test)
y_pred

df3 = pd.DataFrame(np.c_[x_test, y_test, y_pred], columns=['study_hours', 'student_marks_original', 'student_marks_predicted'])


print(df3)
lr.score(x_test, y_test) #`variance score` R^2
lr.score(x_train, y_train) #`bias score` R^2

plt.scatter(x_train, y_train)
plt.scatter(x_train,lr.predict(x_train), color='red')

import joblib # create pip install joblib
joblib.dump(lr, 'student_mark_predictor_model.pkl') 

print("Current Working Directory:" )
import os
print(os.getcwd())

model = joblib.load('student_mark_predictor_model.pkl')
import os
print(os.getcwd())