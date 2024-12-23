"""
Example Title
=============

This is an example script that demonstrates XYZ.
"""

import matplotlib.pyplot as plt
import numpy as np

#%%
# This is a test cell.

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel(r"$x$")
plt.ylabel(r"$\sin(x)$")
# To avoid matplotlib text output
plt.show()