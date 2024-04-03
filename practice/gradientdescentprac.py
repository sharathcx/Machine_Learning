import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1, 2, 3, 4])
y_train = np.array([1, 2, 3, 4])


def derivative(w, b):
    m = x_train.shape[0]  # gets the shape of array
    dot_products = np.dot(x_train, w)
    dfdw, dfdb = 0, 0
    for i, dot_product in enumerate(dot_products):
        dfdw += ((dot_product + b) - y_train[i]) * x_train[i]
        dfdb += (dot_product + b - y_train[i])
    return dfdw/m, dfdb/m


def grad_descent(alpha, w=0.2, b=0.2):
    flag = 1
    while flag > 0.00001:
        dfdw, dfdb = derivative(w, b)
        temp_w = w - alpha * dfdw
        temp_b = b - alpha * dfdb
        flag = temp_w // w
        w = temp_w
        b = temp_b
    print(w, b)
    return w, b


def fwb(w, b):
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x_train[i] + b
    return f_wb


w, b = grad_descent(0.1)
fw_b = fwb(w, b)
plt.plot(x_train, fw_b)
plt.scatter(x_train, y_train)
plt.show() #showing the plot
