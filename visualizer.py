import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

def save_pickle(obj, file_name):
  with open(file_name, 'wb') as f:
    pickle.dump(obj, f)
def load_pickle(file_name):
  with open(file_name, 'rb') as f:
    d = pickle.load(f)
  return d


# Global Plot Settings
plt.rcParams.update({
    'figure.figsize': (6.5, 4.5),  # Double column figure
    'figure.dpi': 300,             # DPI for high-resolution figures
    'font.size': 10,               # Font size for labels and legends
    'font.family': 'Arial',        # Font family
    'lines.linewidth': 1.5,        # Line width
    'axes.grid': True,             # Enable grid
    'grid.alpha': 0.3,             # Light grid lines
    'legend.framealpha': 0.5,      # Transparent legend background
})

# Sample Data for Algorithm Comparison
np.random.seed(42)
x = np.linspace(0, 10, 100)
algorithm_1 = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)
algorithm_2 = np.cos(x) + np.random.normal(0, 0.1, size=x.shape)
algorithm_3 = np.sin(0.5 * x) + np.random.normal(0, 0.1, size=x.shape)

# Plotting the Algorithms
plt.plot(x, algorithm_1, label="Algorithm 1", marker="o", linestyle="-")
plt.plot(x, algorithm_2, label="Algorithm 2", marker="s", linestyle="--")
plt.plot(x, algorithm_3, label="Algorithm 3", marker="^", linestyle="-.")

# Customizing Plot
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Algorithm Comparison')
plt.legend(loc='upper left', frameon=False)

# Save Figure as PDF and PNG
plt.savefig('algorithm_comparison.pdf', format='pdf')  # Vector image
plt.savefig('algorithm_comparison.png', format='png')  # High-DPI raster image

# Display the plot
plt.show()
