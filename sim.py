import opengen as og
import matplotlib.pyplot as plt
import numpy as np

# Create a TCP connection manager
mng = og.tcp.OptimizerTcpManager("OpEn/python_build/ball_and_plate")

# Start the TCP server
mng.start()

# Run simulations
x_state_0 = [0.1, -0.05, 0, 0.0]
simulation_steps = 4000

state_sequence = []
input_sequence = []

mass_ball = 1
moment_inertia = 0.0005
gravity_acceleration = 9.8044
sampling_time_sim = 0.001
sampling_time_control = 0.05
nx = 4

def dynamics_ct(x, u):
    dx1 = x[1]
    dx2 = (5/7)*(x[0] * x[3]**2 - gravity_acceleration * np.sin(x[2]))
    dx3 = x[3]
    dx4 = (u - mass_ball*gravity_acceleration*x[0]*np.cos(x[2]) 
           - 2*mass_ball*x[0]*x[1]*x[3]) \
          / (mass_ball * x[0]**2 + moment_inertia)
    return [dx1, dx2, dx3, dx4]

# RK4
def dynamics_dt(x, u):
    k1 = dynamics_ct(x, u)

    x_k2 = [x[i] + 0.5 * sampling_time_sim * k1[i] for i in range(nx)]
    k2 = dynamics_ct(x_k2, u)

    x_k3 = [x[i] + 0.5 * sampling_time_sim * k2[i] for i in range(nx)]
    k3 = dynamics_ct(x_k3, u)

    x_k4 = [x[i] + sampling_time_sim * k3[i] for i in range(nx)]
    k4 = dynamics_ct(x_k4, u)

    x_next = [x[i] + (sampling_time_sim / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(nx)]
    return x_next

x = x_state_0
u = 0  # Initial control input
control_interval = int(sampling_time_control / sampling_time_sim)  # How many simulation steps per control step

for k in range(simulation_steps):
    if k % control_interval == 0:
        # Update control input at the slower control rate
        solver_status = mng.call(x)
        us = solver_status['solution']
        u = us[0]

    # Update state using RK4
    x_next = dynamics_dt(x, u)

    # Store the state and input
    state_sequence.append(x_next)
    input_sequence.append(u)

    # Update the state for the next iteration
    x = x_next

mng.kill()

# Convert state_sequence into a flattened list for plotting
state_sequence_flat = [item for sublist in state_sequence for item in sublist]

# Generate time vector for plotting
time = np.arange(0, sampling_time_sim * simulation_steps, sampling_time_sim)

# Plot position and angle
plt.plot(time, state_sequence_flat[0::4], '-', label="position")
plt.plot(time, state_sequence_flat[2::4], '-', label="angle")
plt.grid()
plt.ylabel('states')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(0.7, 0.85), loc='upper left', borderaxespad=0.)
plt.show()