import matplotlib.pyplot as plt
import numpy as np

import time
import pickle
import math

from sympy import symbols
from sympy.physics import mechanics

from sympy import Dummy, lambdify
from scipy.integrate import odeint

from matplotlib import animation

def generateODEingredients(n, lengths=None, masses=1):
    """Integrate a multi-pendulum with `n` sections"""
    #-------------------------------------------------
    # Step 1: construct the pendulum model
    print("generating differential equations")
    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass)
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbolsg
    b, g, t = symbols('b,g,t')

    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    P.set_vel(A, 0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Set forces & compute kinematic ODE
        forces.append((Pi, m[i] * g * A.x - b * u[i] / l[i] * Ai.y))
        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(forces, particles)

    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.ones(n) / n

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [b] + [g] + list(l) + list(m) ##
    parameter_vals = [0.09/n] + [9.81] + list(lengths) + list(masses) ##

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u] ##
    unknown_dict = dict(zip(q + u, unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict) ##
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict) ##

    # save the odes in a pickle
    with open("ode"+str(n)+".pickle", "wb") as f:
        pickle.dump((parameters,parameter_vals,unknowns,mm_sym,fo_sym), f)

def solveODEs(n, times, func,
                       initial_positions=135,
                       initial_velocities=0):

    # initial positions and velocities – assumed to be given in degrees
    y0 = np.deg2rad(np.concatenate([np.broadcast_to(initial_positions, n),
                                    np.broadcast_to(initial_velocities, n)]))

    # load odes from pickle
    with open("ode"+str(n)+".pickle", "rb") as f:
        parameters,parameter_vals,unknowns,mm_sym,fo_sym = pickle.load(f)

    # create functions for numerical calculation
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)
    print("generating derivatives of parameters")
    # function which computes the derivatives of parameters
    def gradient(y, t, args):
        vals = np.concatenate((y, args))
        sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
        return np.array(sol).T[0]

    print("integrating")
    # ODE integration

    return func(gradient, y0, times, args=(parameter_vals,))

# euler's method
def euler(func, y0, t, args=()):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        derivative = func(y[i-1], t[i-1], args[0])
        y[i] = derivative*(t[i]-t[i-1]) + y[i-1]
    return y

# trapezoid method
def trapezoid(func, y0, t, args=()):
    y =np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        eulerApprox = dt*func(y[i-1], t[i-1], args[0]) + y[i-1]
        y[i] = 0.5 * dt * (func(y[i-1], t[i-1], args[0]) + func(eulerApprox, t[i], args[0])) + y[i-1]
    return y

# RK4 method
def RK4(func, y0, t, args=()):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = func(y[i-1], t[i-1] , args[0])
        k2 = func(y[i-1] + h*(k1/2), t[i-1] + (h/2), args[0])
        k3 = func(y[i-1] + h*(k2/2), t[i-1] + (h/2), args[0])
        k4 = func(y[i-1] + (h*k3), t[i-1] + h, args[0])
        y[i] = y[i-1] + ((h/6)*(k1+(2*k2)+(2*k3)+k4))
    return y

# get xy coords from relative polar coordinates in p
def get_xy_coords(p, lengths=None):
    p = np.atleast_2d(p)
    n = p.shape[1] // 2
    if lengths is None:
        lengths = np.ones(n) / n
    zeros = np.zeros(p.shape[0])[:, None]
    x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
    return np.cumsum(x, 1), np.cumsum(y, 1)

# animate the pendulum, and write to file
def animate_pendulum2(n, func, stepsize=.05):
    t = np.linspace(0, 10 - stepsize, 10/stepsize)
    x, y = simulate(n, stepsize, func)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))

    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x[i*int(.05/stepsize)], y[i*int(.05/stepsize)])
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=200,
                                   interval=50,
                                   blit=True, init_func=init)
    plt.close(fig)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(str(func.__name__)+'_'+str(n)+'_'+str(stepsize)+'.mp4', writer=writer)

# run the specified solver with the specified step size
def simulate(n, stepsize, func):
    t = np.linspace(0, 10 - stepsize, 10/stepsize)
    p = solveODEs(n, t, func)
    return get_xy_coords(p)

# compare with LSODA
def baseComparison(n):
    return simulate(n, .0001, odeint)

# generate the comparison plots
def generateData(n, func):
    x_base, y_base = baseComparison(n)
    # multipliers to account for different numbers of function evaluations
    multiplier = {"euler": 1, "trapezoid": 2, "RK4": 4}
    stepsizes = [.02,.01,.005, .0025]
    stepsizes = [x*multiplier[str(func.__name__)] for x in stepsizes]
    for i in range(len(stepsizes)):
        step = stepsizes[i]
        x, y = simulate(n, step, func)
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.title("stepsize: " + str(stepsizes[i]) + " function: " + str(func.__name__))
        plt.plot(np.linspace(0, 10 - step, 10/step),((x[:,n]-x_base[::int(step/.0001),n])**2 + (y[:,n]-y_base[::int(step/.0001),n])**2)**.5,'b.')
        plt.xlabel('time')
        plt.ylabel('error (distance away)')
        plt.show()
#generateODEingredients(4)
#anim = animate_pendulum2(4, odeint)

# deprecated create files with the plots
def generateFiles(n, func):
    x_base, y_base = baseComparison(n)
    stepsizes = [.04,.02,.01, .005]
    for i in range(len(stepsizes)):
        step = stepsizes[i]
        x, y = simulate(n, step, func)
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.title("stepsize: " + str(stepsizes[i]) + ", function: " + str(func.__name__) + ", segments: " + str(n))
        plt.plot(np.linspace(0, 10 - step, 10/step),((x[:,n]-x_base[::int(step/.0001),n])**2 + (y[:,n]-y_base[::int(step/.0001),n])**2)**.5,'b.')
        plt.savefig(str(n) + "-" + str(step) + "-" + func.__name__ + ".png")
        plt.show()

# test function to generate a single graph
def generate1Data(n, func):
    x_base, y_base = baseComparison(n)
    step = 0.02
    x, y = simulate(n, step, func)
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title("stepsize: " + str(step) + ", function: " + str(func.__name__))
    plt.plot(np.linspace(0, 10 - step, 10/step),((x[:,n]-x_base[::int(step/.0001),n])**2 + (y[:,n]-y_base[::int(step/.0001),n])**2)**.5,'b.')

    plt.show()

# generateData(4, RK4)
# generateData(4, euler)
# generateData(4, trapezoid)


#Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)


#anim.save('odeint4.mp4', writer=writer)
