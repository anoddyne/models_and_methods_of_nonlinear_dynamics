import numpy as np
import matplotlib.pyplot as plt

# Setting the range for the parameter r
ParameterRange = (2.8, 4)

# Specifying the step size for iteration by r
ParameterStep = 0.0001

# Setting the number of iterations for each value r
IterationsNum = 600

# Choosing how many endpoints to build for each r
PlotPointsNum = 200

# An array for storing calculated values 
Values = np.zeros(IterationsNum)

fig, ax = plt.subplots(figsize=(16, 9))

# Going through the r values with a given step
for r in np.arange(*ParameterRange, ParameterStep):

    # Setting the initial value randomly
    Values[0] = np.random.rand()

    # Going through each r value
    for i in range(IterationsNum - 1):

        # Applying the logistic mapping equation
        Values[i + 1] = r * Values[i] * (1 - Values[i])

    # Plotting the graph
    ax.plot(r * np.ones(PlotPointsNum), Values[-PlotPointsNum:], "b.", markersize=0.02)

# Labels and title of the chart
ax.set(xlabel="r", ylabel="x", title="Логистическое отображение")

# Displaying the graph
plt.show()