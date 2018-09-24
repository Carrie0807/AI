import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#n=50
x = np.linspace(0, 2*np.pi, 50)    #等差数列(start,end,num)
y = np.sin(x)
plt.subplot(3, 4, 1)    #3列4行，現在在位置1
plt.scatter(x, y)   #散佈圖

#泛化
slr = LinearRegression()
#x = pd.DataFrame(x)
x = x.reshape(-1, 1)
slr.fit(x, y)
print("n=50,迴歸係數:", slr.coef_, ",截距:", slr.intercept_)
predicted_y = slr.predict(x)
plt.subplot(3, 4, 1)
plt.plot(x, predicted_y)

#1階多項式擬合
poly_features_1 = PolynomialFeatures(degree = 1, include_bias = False)
X_poly_1 = poly_features_1.fit_transform(x)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X_poly_1, y)
print("n=50,degree=1,迴歸係數:", lin_reg_1.coef_, ",截距:", lin_reg_1.intercept_)
X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_1.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_1.coef_.T) + lin_reg_1.intercept_
plt.subplot(3, 4, 2)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x, y, 'b.')
plt.title("n=50,degree=1")

#3階多項式擬合
poly_features_3 = PolynomialFeatures(degree = 3, include_bias = False)
X_poly_3 = poly_features_3.fit_transform(x)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_3, y)
print("n=50,degree=3,迴歸係數:", lin_reg_3.coef_, ",截距:", lin_reg_3.intercept_)
X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_3.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_3.coef_.T) + lin_reg_3.intercept_
plt.subplot(3, 4, 3)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x, y, 'b.')
plt.title("n=50,degree=3")

#9階多項式擬合
poly_features_9 = PolynomialFeatures(degree=9, include_bias=False)
X_poly_9 = poly_features_9.fit_transform(x)
lin_reg_9 = LinearRegression()
lin_reg_9.fit(X_poly_9, y)
print("n=50,degree=9,迴歸係數:", lin_reg_9.coef_, ",截距:", lin_reg_9.intercept_)
X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_9.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_9.coef_.T) + lin_reg_9.intercept_
plt.subplot(3, 4, 4)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x, y, 'b.')
plt.title("n=50,degree=9")

#n=100
x1 = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x1) + np.random.randn(len(x1))/5.0
plt.subplot(3, 4, 9)
plt.scatter(x1, y1)

#泛化
slr = LinearRegression()
x1 = x1.reshape(-1, 1)
slr.fit(x1, y1)
print("n=100,迴歸係數:", slr.coef_, ",截距:", slr.intercept_)
predicted_y1 = slr.predict(x1)
plt.subplot(3, 4, 9)
plt.plot(x1, predicted_y1)

#1階多項式擬合
poly_features_1 = PolynomialFeatures(degree = 1, include_bias = False)
X_poly_1 = poly_features_1.fit_transform(x1)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X_poly_1, y1)
print("n=100,degree=1,迴歸係數:", lin_reg_1.coef_, ",截距:", lin_reg_1.intercept_)
X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_1.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_1.coef_.T) + lin_reg_1.intercept_
plt.subplot(3, 4, 10)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')
plt.title("n=100,degree=1")

#3階多項式擬合
poly_features_3 = PolynomialFeatures(degree = 3, include_bias = False)
X_poly_3 = poly_features_3.fit_transform(x1)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_3, y1)
print("n=100,degree=3,迴歸係數:", lin_reg_3.coef_, ",截距:", lin_reg_3.intercept_)
X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_3.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_3.coef_.T) + lin_reg_3.intercept_
plt.subplot(3, 4, 11)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')
plt.title("n=100,degree=3")

#9階多項式擬合
poly_features_9 = PolynomialFeatures(degree=9, include_bias=False)
X_poly_9 = poly_features_9.fit_transform(x1)
lin_reg_9 = LinearRegression()
lin_reg_9.fit(X_poly_9, y1)
print("n=100,degree=9,迴歸係數:", lin_reg_9.coef_, ",截距:", lin_reg_9.intercept_)
X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_9.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_9.coef_.T) + lin_reg_9.intercept_
plt.subplot(3, 4, 12)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')
plt.title("n=100,degree=9")
