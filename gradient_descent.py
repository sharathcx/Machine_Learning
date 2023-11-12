import numpy as np
from matplotlib import pyplot as plt

x_train = np.array([1, 2])
y_train = np.array([300, 500])


def fwb(x, y, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


def plotter(x, y):
    w, b = gradient_descent(x,y)
    plt.plot(x_train, fwb(x, y, w, b), c='b', label='Our Prediction')
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    plt.show()


plt.scatter(x_train, y_train, marker="x")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()


# calculating cost
def cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = (w * x) + b
        cost += (f_wb - y[i]) ** 2
    total_cost = cost / (2 * m)
    return total_cost


def calculate_grad(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(x, y, w=0.1, b=0.1, alpha=0.1):
    for i in range(10000):
        dj_dw, dj_db = calculate_grad(x, y, w, b)
        tmp_w = w - alpha * dj_dw
        tmp_b = b - alpha * dj_db
        w = tmp_w
        b = tmp_b
    return w, b


print(gradient_descent(x_train, y_train))
plotter(x_train, y_train)
