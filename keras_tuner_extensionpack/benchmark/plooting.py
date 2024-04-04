from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from keras_tuner_extensionpack.benchmark import functions as func


# Define the range of the input space
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = func.ackley(np.array([x, y]))

# Calculate the output value for each pair of input
plt.figure(figsize=(7, 7))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Plot the 3D surface
ax = plt.subplot(1, 1, 1, projection="3d")
ax.plot_surface(x, y, z, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Ackley Function")

plt.show()
