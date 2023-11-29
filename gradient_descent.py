import numpy as np
from matplotlib import pyplot as plt


class GradientDecent:
    def __init__(self, x_train=None, y_train=None):
        self.x_train = x_train
        self.y_train = y_train

    def fwb(self, w, b):
        m = self.x_train.shape[0]
        f_wb = np.zeros(m)
        for i in range(m):
            f_wb[i] = w * self.x_train[i] + b
        return f_wb

    def calculate_grad(self, w, b):
        m = self.x_train.shape[0]
        dj_dw = 0
        dj_db = 0
        for i in range(m):
            f_wb = w * self.x_train[i] + b
            dj_dw_i = (f_wb - self.y_train[i]) * self.x_train[i]
            dj_db_i = (f_wb - self.y_train[i])
            dj_dw += dj_dw_i
            dj_db += dj_db_i
        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db

    def gradient_descent(self, w=0.1, b=0.1, alpha=0.01):
        tmp_w = w
        flag = tmp_w // w
        while flag > 0.00001:
            dj_dw, dj_db = self.calculate_grad(w, b)
            tmp_w = w - alpha * dj_dw
            tmp_b = b - alpha * dj_db
            flag = tmp_w - w
            w = tmp_w
            b = tmp_b
        return w, b

    def predict(self, w, b, x):
        return w * x + b

    def plotter(self, path=None):
        w, b = self.gradient_descent()
        print("The predicted value is")
        f_wb = self.fwb(w, b)
        plt.plot(self.x_train, f_wb, c='b', label='Our Prediction')
        plt.scatter(self.x_train, self.y_train, marker='x', c='r', label='Actual Values')
        if path:
            plt.savefig("C:/Users/shara/OneDrive/Desktop/Machine Learning/plot")
        plt.show()
