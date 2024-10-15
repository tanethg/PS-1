import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the step function
def step_function(x):
    return np.where(x >= -4, 17, 3)  # 1 for x >= 0, 0 otherwise

# Step 2: Generate the input data for the simulation
x = np.linspace(-10, 10, 500)  # Generate 500 points from -10 to 10

# Step 3: Apply the step function to the input data
y = step_function(x)

# Step 4: Plot the step function
plt.plot(x, y, label="Step Function")
plt.title("Step Function Simulation")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True)
plt.legend()
plt.show()
