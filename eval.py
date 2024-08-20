import opengen as og
import matplotlib.pyplot as plt
import numpy as np

# Create a TCP connection manager
mng = og.tcp.OptimizerTcpManager("OpEn/python_build/ball_and_plate", port=8333)
mng2 = og.tcp.OptimizerTcpManager("OpEn/python_build_lifted_slow/ball_and_plate_lifted_slow", port=8334)
# Start the TCP servers
mng.start()
mng2.start()

# Define parameters
x_state_0 = [0.1, -0.05, 0, 0.0]
simulation_steps = 4000 * 10

mass_ball = 1
moment_inertia = 0.0005
gravity_acceleration = 9.8044
sampling_time_sim = 0.001
sampling_time_control = 0.05
nx = 4

def dynamics_ct(x, u):
    dx1 = x[1]
    dx2 = (5 / 7) * (x[0] * x[3] ** 2 - gravity_acceleration * np.sin(x[2]))
    dx3 = x[3]
    dx4 = (u - mass_ball * gravity_acceleration * x[0] * np.cos(x[2])
           - 2 * mass_ball * x[0] * x[1] * x[3]) \
          / (mass_ball * x[0] ** 2 + moment_inertia)
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

# Simulate with the first controller
x1 = x_state_0
state_sequence_1 = []
u_sequence_1 = []
u1 = 0  # Initial control input for controller 1
control_interval = int(sampling_time_control / sampling_time_sim)

for k in range(simulation_steps):
    if k % control_interval == 0:
        # Update control input at the slower control rate
        solver_status = mng.call(x1)
        us = solver_status['solution']
        u1 = us[0]

    # Update state using RK4
    x_next_1 = dynamics_dt(x1, u1)

    # Store the state and input
    state_sequence_1.append(x_next_1)
    u_sequence_1.append(u1)

    # Update the state for the next iteration
    x1 = x_next_1

# Simulate with the second controller
x2 = x_state_0
state_sequence_2 = []
u_sequence_2 = []
u2 = 0  # Initial control input for controller 2

for k in range(simulation_steps):
    if k % control_interval == 0:
        # Update control input at the slower control rate
        solver_status = mng2.call(x2)
        us = solver_status['solution']
        u2 = us[0]

    # Update state using RK4
    x_next_2 = dynamics_dt(x2, u2)

    # Store the state and input
    state_sequence_2.append(x_next_2)
    u_sequence_2.append(u2)

    # Update the state for the next iteration
    x2 = x_next_2

# Convert state_sequence into flattened lists for plotting
state_sequence_flat_1 = [item for sublist in state_sequence_1 for item in sublist]
state_sequence_flat_2 = [item for sublist in state_sequence_2 for item in sublist]

# Generate time vector for plotting
time = np.arange(0, sampling_time_sim * simulation_steps, sampling_time_sim)
control_time = np.arange(0, sampling_time_control * len(u_sequence_1), sampling_time_control)

# Determine the right limit based on the simulation time
right_limit = time[-1]  # The last value of the time array
control_right_limit = control_time[-1]  # The last value of the control_time array

# Plot positions, angles, velocities, and control inputs for both controllers
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Main title
fig.suptitle('Controller period 0.05', fontsize=16)

# Plot Position (x[0])
axs[0].plot(time, state_sequence_flat_1[0::4], '-', label="NMPC")
axs[0].plot(time, state_sequence_flat_2[0::4], '-', label="Lifted NMPC")
axs[0].set_ylabel('Position (m)')
axs[0].grid(True)
axs[0].legend(loc='upper right')
axs[0].set_xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis

# Plot Angle (x[2])
axs[1].plot(time, state_sequence_flat_1[2::4], '-', label="NMPC")
axs[1].plot(time, state_sequence_flat_2[2::4], '-', label="Lifted NMPC")
axs[1].set_ylabel('Angle (rad)')
axs[1].grid(True)
axs[1].legend(loc='upper right')
axs[1].set_xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis

# Plot Velocity (x[1])
axs[2].plot(time, state_sequence_flat_1[1::4], '-', label="NMPC")
axs[2].plot(time, state_sequence_flat_2[1::4], '-', label="Lifted NMPC")
axs[2].set_ylabel('Velocity (m/s)')
axs[2].grid(True)
axs[2].legend(loc='upper right')
axs[2].set_xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis

# Plot Control Input (u)
axs[3].plot(control_time, u_sequence_1, '-', label="NMPC")
axs[3].plot(control_time, u_sequence_2, '-', label="Lifted NMPC")
axs[3].set_ylabel('Control Input (u)')
axs[3].set_xlabel('Time (s)')
axs[3].grid(True)
axs[3].legend(loc='upper right')
axs[3].set_xlim(left=0, right=control_right_limit)  # Set both left and right limits for the x-axis

plt.tight_layout()
plt.show()

# Stop the TCP servers
mng.kill()
mng2.kill()
