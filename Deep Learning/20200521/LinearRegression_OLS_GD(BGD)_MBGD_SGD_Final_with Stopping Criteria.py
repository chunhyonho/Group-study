# Reference : https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2
# imports
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time


class LinearRegressionUsingGD:

    def __init__(self, eta=0.05, epochs=1000):
        self.eta = eta
        self.epochs = epochs

    def fit(self, x, y):
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]
        xTx = np.matmul(x.T, x)
        yTx = np.matmul(y.T, x)

        for _ in range(self.epochs):
            gradient_vector = np.matmul(self.w_.T, xTx) - yTx
            self.w_ -= (self.eta / m) * gradient_vector.T

        return self

    def predict(self, x):
        return np.dot(x, self.w_)


class LinearRegressionUsingMBGD:

    def __init__(self, eta=0.05, epochs=1000, batch_size=32):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, x, y):
        self.w_ = np.zeros((x.shape[1], 1))
        m = self.batch_size

        for _ in range(self.epochs):
            # Choosing Mini-Batch with size m
            numdata = np.arange(x.shape[0])
            J = np.random.choice(numdata, size=m, replace=False)
            xb = x[J]
            yb = y[J]
            xbTxb = np.matmul(xb.T, xb)
            ybTxb = np.matmul(yb.T, xb)
            # Calculating & Updating Gradient
            gradient_vector = np.matmul(self.w_.T, xbTxb) - ybTxb
            self.w_ -= (self.eta / m) * gradient_vector.T

        return self

    def predict(self, x):
        return np.dot(x, self.w_)


class LinearRegressionUsingSGD:

    def __init__(self, eta=0.05, epochs=1000, batch_size=1):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, x, y):
        self.w_ = np.zeros((x.shape[1], 1))
        m = self.batch_size

        for _ in range(self.epochs):
            # Choosing a data randomly
            numdata = np.arange(x.shape[0])
            J = np.random.choice(numdata, size=m, replace=False)
            xb = x[J]
            yb = y[J]
            xbTxb = np.matmul(xb.T, xb)
            ybTxb = np.matmul(yb.T, xb)
            # Calculating & Updating Gradient
            gradient_vector = np.matmul(self.w_.T, xbTxb) - ybTxb
            self.w_ -= (self.eta / m) * gradient_vector.T

        return self

    def predict(self, x):
        return np.dot(x, self.w_)


# Generate random data-set
np.random.seed(0)
print("<Generating Data....>")
n = 100  # vector size
N = (2 ** 5) * 5 * 100  # number of data

x = np.random.rand(N, n)  # Data matrix
w = 10 * np.random.rand(n, 1)  # True parameter
yt = np.matmul(x, w)  # y value matrix induced by data x
e = np.random.randn(N, 1)  # error term
yd = np.add(yt, e)  # y data matrix we have

a = 10  # number of calculations

# Calculate the Ordinary Least Squares solution
m_time_OLS = np.zeros(a)
m_R2score_OLS = np.zeros(a)

print("<Executing OLS method....>")
for i in range(a):
    calstarted_OLS = time.time()

    b1 = inv(np.matmul(x.T, x))
    b2 = np.matmul(x.T, yd)
    B = np.matmul(b1, b2)  # solution matrix
    y_OLS = np.matmul(x, B)  # y value induced by x, B

    calended_OLS = time.time()
    caltime_OLS = calended_OLS - calstarted_OLS  # OLS calculation time

    m_time_OLS[i] = caltime_OLS  # Recording caltime to get average

    # Calculating R^2 score for OLS
    ssr = np.sum((yt - y_OLS) ** 2)
    sst = np.sum((yt - np.mean(yt)) ** 2)
    r2score_OLS = 1 - (ssr / sst)

    m_R2score_OLS[i] = r2score_OLS  # Recording R^2 score to get average

# Calculating average run-time of OLS
avg_time_OLS = np.average(m_time_OLS)
avg_R2score_OLS = np.average(m_R2score_OLS)

# Using Gradient Descent
GD = LinearRegressionUsingGD()
m_time_GD = np.zeros(a)
m_R2score_GD = np.zeros(a)

print("<Executing Gradient Descent....>")
for i in range(a):
    calstarted_GD = time.time()

    GD.fit(x, yd)
    y_GD = GD.predict(x)  # y value induced by Gradient Descent

    calended_GD = time.time()
    caltime_GD = calended_GD - calstarted_GD  # GD calculation time

    m_time_GD[i] = caltime_GD  # Recording caltime to get average

    # Calculating R^2 score for GD
    ssr = np.sum((yt - y_GD) ** 2)
    sst = np.sum((yt - np.mean(yt)) ** 2)
    r2score_GD = 1 - (ssr / sst)

    m_R2score_GD[i] = r2score_GD  # Recording R^2 score to get average

# Calculating average run-time of GD
avg_time_GD = np.average(m_time_GD)
avg_R2score_GD = np.average(m_R2score_GD)

# Using Mini-Batch Gradient Descent
MBGD = LinearRegressionUsingMBGD()
m_time_MBGD = np.zeros(a)
m_R2score_MBGD = np.zeros(a)

print("<Executing Mini-Batch Gradient Descent....>")
for i in range(a):
    calstarted_MBGD = time.time()

    MBGD.fit(x, yd)
    y_MBGD = MBGD.predict(x)  # y value induced by Mini-Batch Gradient Descent

    calended_MBGD = time.time()
    caltime_MBGD = calended_MBGD - calstarted_MBGD  # MBGD calculation time

    m_time_MBGD[i] = caltime_MBGD  # Recording caltime to get average

    # Calculating R^2 score for MBGD
    ssr = np.sum((yt - y_MBGD) ** 2)
    sst = np.sum((yt - np.mean(yt)) ** 2)
    r2score_MBGD = 1 - (ssr / sst)

    m_R2score_MBGD[i] = r2score_MBGD  # Recording R^2 score to get average

# Calculating average run-time of MBGD
avg_time_MBGD = np.average(m_time_MBGD)
avg_R2score_MBGD = np.average(m_R2score_MBGD)

# Using Stochastic Gradient Descent
SGD = LinearRegressionUsingSGD()
m_time_SGD = np.zeros(a)
m_R2score_SGD = np.zeros(a)

print("<Executing Stochastic Gradient Descent....>")
for i in range(a):
    calstarted_SGD = time.time()

    SGD.fit(x, yd)
    y_SGD = SGD.predict(x)  # y value induced by Stochastic Gradient Descent

    calended_SGD = time.time()
    caltime_SGD = calended_SGD - calstarted_SGD  # SGD calculation time

    m_time_SGD[i] = caltime_SGD  # Recording caltime to get average

    # Calculating R^2 score for SGD
    ssr = np.sum((yt - y_SGD) ** 2)
    sst = np.sum((yt - np.mean(yt)) ** 2)
    r2score_SGD = 1 - (ssr / sst)

    m_R2score_SGD[i] = r2score_SGD  # Recording R^2 score to get average

# Calculating average run-time of SGD
avg_time_SGD = np.average(m_time_SGD)
avg_R2score_SGD = np.average(m_R2score_SGD)

# Reporting every recorded values
print("<Report>")
print("| Method || Average Run-time || Average R^2 Score |")
print("|  OLS   ||%f sec        ||%f             |" % (avg_time_OLS, avg_R2score_OLS))
print("|  GD    ||%f sec        ||%f             |" % (avg_time_GD, avg_R2score_GD))
print("|  MBGD  ||%f sec        ||%f             |" % (avg_time_MBGD, avg_R2score_MBGD))
print("|  SGD   ||%f sec        ||%f             |" % (avg_time_SGD, avg_R2score_SGD))