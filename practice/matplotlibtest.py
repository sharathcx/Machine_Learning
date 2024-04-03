import matplotlib.pyplot as plt


# to plot x and y
x = [x for x in range(100)]
y = [x*x for x in range(100)]

plt.plot(x, y, marker="x")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Test")
plt.show()

# to plot a line
x1, y1 = 0, 0
x2, y2 = 4, 3

plt.plot([x1, x2], [y1, y2])
plt.show()


