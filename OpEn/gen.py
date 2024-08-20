import casadi.casadi as cs
import opengen as og

mass_ball = 1
moment_inertia = 0.0005
gravity_acceleration = 9.8044
sampling_time = 0.05
nx = 4
T = 15

def dynamics_ct(x, u):
    dx1 = x[1]
    dx2 = (5/7)*(x[0] * x[3]**2 - gravity_acceleration * cs.sin(x[2]))
    dx3 = x[3]
    dx4 = (u - mass_ball*gravity_acceleration*x[0]*cs.cos(x[2]) 
           - 2*mass_ball*x[0]*x[1]*x[3]) \
          / (mass_ball * x[0]**2 + moment_inertia)
    return [dx1, dx2, dx3, dx4]

def dynamics_dt_euler(x, u):
    dx = dynamics_ct(x, u)
    return [x[i] + sampling_time * dx[i] for i in range(nx)]

def dynamics_dt(x, u):
    k1 = dynamics_ct(x, u)

    x_k2 = [x[i] + 0.5 * sampling_time * k1[i] for i in range(nx)]
    k2 = dynamics_ct(x_k2, u)

    x_k3 = [x[i] + 0.5 * sampling_time * k2[i] for i in range(nx)]
    k3 = dynamics_ct(x_k3, u)

    x_k4 = [x[i] + sampling_time * k3[i] for i in range(nx)]
    k4 = dynamics_ct(x_k4, u)

    x_next = [x[i] + (sampling_time / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(nx)]
    return x_next

def stage_cost(x, u):
    cost = 5*x[0]**2 + 0.01*x[1]**2 + 0.01*x[2]**2 + 0.05*x[3]**2 + 2.2*u**2
    return cost

def terminal_cost(x):
    cost = 100*x[0]**2 + 50*x[2]**2 + 20*x[1]**2 + 0.8*x[3]**2
    return cost

u_seq = cs.MX.sym("u", T)  # sequence of all u's
x0 = cs.MX.sym("x0", nx)   # initial state

x_t = x0
total_cost = 0
for t in range(0, T):
    total_cost += stage_cost(x_t, u_seq[t])  # update cost
    x_t = dynamics_dt(x_t, u_seq[t])         # update state

total_cost += terminal_cost(x_t)  # terminal cost

U = og.constraints.BallInf(None, 0.95)

problem = og.builder.Problem(u_seq, x0, total_cost)  \
            .with_constraints(U)

build_config = og.config.BuildConfiguration()  \
    .with_build_directory("python_build")      \
    .with_tcp_interface_config()

meta = og.config.OptimizerMeta().with_optimizer_name("ball_and_plate")

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-6)\
    .with_initial_tolerance(1e-6)

builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                          build_config, solver_config)
builder.build()