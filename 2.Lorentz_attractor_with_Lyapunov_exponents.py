import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def compute_Lorenz_derivatives(state_vector):
    """
    This function computes the derivatives of the Lorenz system and its Jacobian.

    Parameters:
    state_vector (numpy array): The state vector of the Lorenz system.

    Returns:
    lorenz_derivatives (numpy array): The derivatives of the Lorenz system.
    jacobian_matrix (numpy array): The Jacobian matrix of the Lorenz system.
    """
    x_val, y_val, z_val = state_vector
    lorenz_derivatives = [
        sigma * (y_val - x_val), r * x_val - y_val - x_val * z_val, x_val * y_val - b * z_val]
    jacobian_matrix = [[-sigma, sigma, 0],
                       [r - z_val, -1, -x_val], [y_val, x_val, -b]]
    return np.array(lorenz_derivatives), np.array(jacobian_matrix)


def compute_LEC_system(state_vector):
    """
    This function computes the system that includes the Lorenz system and Lyapunov exponents.

    Parameters:
    state_vector (numpy array): The state vector of the Lorenz system and Lyapunov exponents.

    Returns:
    numpy array: The derivatives of the Lorenz system, Lyapunov exponents, and the U matrix.
    """
    U_matrix = state_vector[3:12].reshape([3, 3])
    lorenz_derivatives, jacobian_matrix = compute_Lorenz_derivatives(
        state_vector[:3])
    A_matrix = U_matrix.T.dot(jacobian_matrix.dot(U_matrix))
    lyapunov_derivatives = np.diag(A_matrix).copy()
    for i in range(3):
        A_matrix[i, i] = 0
        for j in range(i + 1, 3):
            A_matrix[i, j] = -A_matrix[j, i]
    U_derivatives = U_matrix.dot(A_matrix)
    return np.concatenate([lorenz_derivatives, U_derivatives.flatten(), lyapunov_derivatives])


# Initial parameters
sigma = 10
r = 28
b = 8 / 3

U_matrix = np.eye(3)
initial_state_vector = np.ones(3)
lyapunov_exponents = []

iterations = 10 ** 3
time_step = 0.1
final_time = iterations * time_step

# Initialization of the initial state for the system with Lyapunov exponents
initial_state_vector = np.ones(3)
initial_U_matrix = np.identity(3)
initial_L_vector = np.zeros(3)
initial_state_vector = np.concatenate(
    [initial_state_vector, initial_U_matrix.flatten(), initial_L_vector])

# Time interval for integration
time_interval = np.linspace(0, 200, 501)
# Integration of the system with Lyapunov exponents
state_vector = odeint(lambda state_vector, t: compute_LEC_system(
    state_vector), initial_state_vector, time_interval, hmax=0.05)
# Calculation of Lyapunov exponents
L_vector = state_vector[5:, 12:15].T / time_interval[5:]

average_lyapunov_exponents = np.mean(L_vector, axis=1)

# Plotting of Lyapunov exponents
plt.plot(time_interval[5:], L_vector.T)
plt.legend([f'Lyapunov exponent {
           i+1}: {average_lyapunov_exponents[i]}' for i in range(3)])
plt.show()
