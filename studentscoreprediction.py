import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

url = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
try:
  df = pd.read_csv(url)
except :
  print("the file cant be read")

print(df.head(10))

df.plot(x='Hours' , y='Scores',kind='scatter')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

X=[]
for i in df['Hours']:
  X.append([i])
print(X)
y=[]
for i in df['Scores']:
  y.append([i])
print(y)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.3, random_state=0)
print(X_train)

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")

line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

print(X_test)
y_pred = regressor.predict(X_test)

Y_test=[]
Y_pred=[]
for i in y_test:
  Y_test.append(sum(i))
for i in y_pred:
  Y_pred.append(sum(i))
df2 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
print(df2 )

hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(own_pred[0][0]))

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, Y_pred)) 


