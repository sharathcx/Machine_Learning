import random

import pandas as pd
import random

# x = []
# y = []
# a = 200
# for i in range(2, 20):
#     x.append(i)
#     y.append(a)
#     a += 100
#
# data = {'Square Feet': x,
#         'Price': y}
# df = pd.DataFrame(data)
#
# print(df)
path = "C:/Users/shara/OneDrive/Desktop/Machine Learning/cleanedHouseData.xlsx"
# df.to_excel(path, index=False)

read_df = pd.read_excel(path)
mean_x = read_df.iloc[:, 0].mean()
mean_y = read_df.iloc[:, 1].mean()
x = read_df.iloc[:, 0].values
y = read_df.iloc[:, 5].values
data = {'Square Feet': x/mean_x,
        'Price': y/mean_y}
df = pd.DataFrame(data)
df.to_excel("C:/Users/shara/OneDrive/Desktop/Machine Learning/new_data.xlsx")


