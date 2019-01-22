# %%
from benchmark.simple import rosenbrock, rastrigin

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sys

# %% rosenbrock
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Xs = np.array([X, Y])
Z = rosenbrock(Xs)

fig = plt.figure()

ax = fig.gca(projection='3d')
ax.view_init(30, 130)

surf = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False
)

plt.show()

# %% rastrigin
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Xs = np.array([X, Y])
Z = rosenbrock(Xs)
Z = rastrigin(Xs)

fig2 = plt.figure()

ax = fig2.gca(projection='3d')
ax.view_init(45, 130)

surf = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False
)

plt.show()
