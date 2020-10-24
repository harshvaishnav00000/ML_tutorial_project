import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
diabetes = datasets.load_diabetes()
# print(diabetes.keys())
diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

model = linear_model.LinearRegression()
# training
model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_pre =  model.predict(diabetes_X_test)

print("mean square error: ", mean_squared_error(diabetes_Y_pre, diabetes_Y_test))

# plt.scatter(diabetes_X_test, diabetes_Y_test)
# plt.plot(diabetes_X_test, diabetes_Y_pre)
# plt.show()

print("W: ", model.coef_)
print("I: ", model.intercept_)

