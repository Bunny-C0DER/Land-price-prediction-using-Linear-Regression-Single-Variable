#importing libraries

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#loading dataset

dataset = pd.read_csv('dataset.csv')

#summarizing dataset

print(dataset.shape)
print(dataset.head(5))

#visualizing dataset

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(dataset.area,dataset.price,color='red',marker='*')

#segregating dataset into X and Y

X=dataset.drop('price',axis='columns')
X
Y=dataset.price
Y

#training dataset using Linear Regression

model = LinearRegression()
model.fit(X,Y)

#predicting price for land entered by user

x = int(input("Enter Area of Land to find its Price: "))
LandAreainSqFt=[[x]]
result = model.predict(LandAreainSqFt)
print(result)

#checking our model manually(how it works)

m = model.coef_   #Coefficient(m)
print(m)
b = model.intercept_    #Intercept(b)
print(b)


y = m*x + b     #y(Price), x(Area(user-input)), m(coefficient), b(intercept)
print("The price of {0} Square feer Land is: {1}".format(x,y[0]))
