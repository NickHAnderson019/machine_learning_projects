import sympy as sp, utils
from math import pi
import pyomo
from pyomo.environ import (
    ConcreteModel, RangeSet, Set, Param, Var, Objective, Constraint
)
from pyomo.opt import SolverFactory


def abs_to_rel(th):
    out = [th[0]]
    # thp = previous, thc = current
    for thp, thc in zip(th[:-1], th[1:]):
        out.append(thc - thp)
    return out


def rel_to_abs(th):
    out = [th[0]]
    for i in range(1, len(th)):
        out.append(out[i-1] + th[i])
    return out


def deriv(expr, q, dq):
    return (sp.diff(expr, q).T * dq)[0]


def get_N_link_EOM(num_links: int, angles: str, lambdify: bool,
                   debug: bool = False, latex_syms: bool = True):
    """angles = 'abs' or 'rel'"""
    X0, Y0, g = sp.symbols(['X0', 'Y0', 'g']) if debug else [0, 0, 9.81]
    _s = str(num_links)
    m  = sp.symbols('m_:'  + _s) if debug else [1]*num_links
    r  = sp.symbols('r_:'  + _s) if debug else [1]*num_links
    In = sp.symbols('In_:' + _s) if debug else [m[i]*r[i]**2 / 12 for i in range(num_links)]
    Tc = sp.symbols('T_c:' + _s)
    if latex_syms:
        # could be relative or absolute!
        th = sp.symbols('\\theta_:' + _s)
        dth  = sp.symbols('\\dot{\\theta_:'  + _s + '}')
        ddth = sp.symbols('\\ddot{\\theta_:' + _s + '}')
    else:
        th = sp.symbols('th_:' + _s)
        dth  = sp.symbols('dth_:'  + _s)
        ddth = sp.symbols('ddth_:' + _s)

    # create matrices of generalized coordinates
    q = sp.Matrix(th)
    dq = sp.Matrix(dth)
    ddq = sp.Matrix(ddth)

    # torque in base
    u = sp.Matrix(list(Tc[0:]))

    # calc the absolute position of the end and middle of each link
    if angles == 'abs':
        th_abs = th
        dth_abs = dth
        ddth_abs = ddth
    else:
        th_abs = rel_to_abs(th)
        dth_abs = rel_to_abs(dth)
        ddth_abs = rel_to_abs(ddth)

    xend, yend = [X0], [Y0]
    for link in range(1, num_links):
        xend.append(xend[link-1] + r[link-1]*sp.sin(th_abs[link-1]))
        yend.append(yend[link-1] - r[link-1]*sp.cos(th_abs[link-1]))

    x, y = [], []
    for link in range(num_links):
        x.append(xend[link] + 0.5*r[link]*sp.sin(th_abs[link]))
        y.append(yend[link] - 0.5*r[link]*sp.cos(th_abs[link]))

    x, y, xend, yend = sp.trigsimp([x,y,xend,yend])

    dx = sp.trigsimp([deriv(x[link], q, dq) for link in range(num_links)])
    dy = sp.trigsimp([deriv(y[link], q, dq) for link in range(num_links)])

    # calculate the system's kinetic and potential energy
    Ek = sp.Matrix([
        sum(0.5*m[l]*(dx[l]**2 + dy[l]**2) + 0.5*In[l]*dth_abs[l]**2
            for l in range(num_links)
    )])

    Ep = sp.trigsimp(sp.Matrix([
        sum(m[link]*g*y[link] for link in range(num_links))
    ]))
    
    M, C, G = utils.manipulator_equation(Ek, Ep, q, dq)
    M, G = sp.trigsimp([M, G])

    if angles == 'abs':
        u_prev = sp.Matrix(u[1:] + [0])
        torques = u - u_prev
    else:
        torques = u
    
    EOM   = M * ddq + G + C - torques
    EOM_C = M * ddq + G     - torques

    if lambdify is False:
        return sp.Matrix(EOM), sp.Matrix(EOM_C) #might want to simply them first.
    else:
        vars_in_EOM = [*q, *dq, *ddq, *Tc]
        return utils.lambdify_EOM(EOM, vars_in_EOM, display_vars=False)


def make_pyomo_model(EOM: list, nfe: int, angles: str,
                     collocation: str, total_time: float,
                     seed: int = None):
    num_links = len(EOM)
    # assert num_links > 1
    
    ncp = 1 if collocation == 'euler' else 3
    
    m = ConcreteModel(name=f'{num_links}-link pendulum')

    # sets
    m.fe = RangeSet(nfe)
    m.cp = RangeSet(ncp)
    m.links = RangeSet(num_links)

    # master timestep
    m.hm0 = Param(initialize=total_time/nfe)
    m.hm  = Param(m.fe, initialize=1.0)

    # variables
    m.q   = Var(m.fe, m.cp, m.links, bounds=(-2*pi, 2*pi)) # θ = angle
    m.dq  = Var(m.fe, m.cp, m.links)    # d/dt θ
    m.ddq = Var(m.fe, m.cp, m.links)    # d/dt d/dt θ

    g = 9.81
    Tc_bounds = (-g, g) if num_links > 2 else (-2*g, 2*g)
    m.Tc = Var(m.fe, m.links, bounds=Tc_bounds)  # control torque

    # objective - min energy
    m.cost = Objective(expr = sum(m.Tc[fe, link]**2
                                       for fe in m.fe
                                       for link in m.links))

    # equations of motion (EOM)
    def EOM_fun(m, fe, cp, link):
        var_list = [*m.q[fe,cp,:], *m.dq[fe,cp,:], *m.ddq[fe,cp,:], *m.Tc[fe,:]]
        return EOM[link-1](*var_list) == 0
    m.EOM = Constraint(m.fe, m.cp, m.links, rule=EOM_fun)

    # collocation
    if collocation == 'euler':
        m.interp_q  = Constraint(m.fe, m.cp, m.links, rule=utils.implicit_euler(m.q,  m.dq))
        m.interp_dq = Constraint(m.fe, m.cp, m.links, rule=utils.implicit_euler(m.dq, m.ddq))
    else:
        m.interp_q  = Constraint(m.fe, m.cp, m.links, rule=utils.radau_3(m.q,  m.dq))
        m.interp_dq = Constraint(m.fe, m.cp, m.links, rule=utils.radau_3(m.dq, m.ddq))
    
    # set initial values
    import random
    if seed is not None:
        random.seed(seed)
    f = lambda x: random.uniform(-x, x)
    for fe in m.fe:
        for cp in m.cp:
            for link in m.links:
                m.q[fe,cp,link].value = f(pi)
                m.dq[fe,cp,link].value = f(0.1)
                m.ddq[fe,cp,link].value = f(0.1)
                m.Tc[fe,link].value = f(0.1)
                    
    # set starting point
    for link in m.links:
        m.q[1, ncp, link].fixed = True
        m.dq[1, ncp, link].fixed = True
    
    # set the final position - inverted with small movement
    for link in m.links:
        if (link == m.links[1]) or (angles == 'abs'):
            m.q[nfe, ncp, link].fix(pi)
        else:
            m.q[nfe,ncp,link].fix(0)

        m.dq[nfe,ncp,link].setlb(-0.1)
        m.dq[nfe,ncp,link].setub( 0.1)
    
    return m

def default_solver(output_file: str, max_mins: int, solver: str = 'ma86'):
    import os, platform
    from datetime import datetime
    
    if platform.system() == 'Linux':
        IPOPT_PATH = '/home/alex/CoinIpopt/build/bin/ipopt'
    else:
        IPOPT_PATH = 'C:/cygwin64/home/Nick/CoinIpopt/build/bin/ipopt.exe'

    opt = SolverFactory('ipopt', executable=IPOPT_PATH)
    opt.options['print_level'] = 5
    opt.options['max_iter'] = 30_000
    opt.options['max_cpu_time'] = max_mins * 60
    opt.options['Tol'] = 1e-6
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['halt_on_ampl_error'] = 'yes'
    opt.options['OF_acceptable_tol'] = 1e-6
    # opt.options['output_file'] = os.getcwd() + '/' + output_file

    if platform.system() == 'Linux':
        if solver == 'ma86':
            opt.options['linear_solver'] = 'ma86'
            opt.options['OF_ma86_scaling'] = 'none'
        else:
            opt.options['linear_solver'] = solver  # other options: 'ma77', 'ma97'

    # print('optimization start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return opt

def make_animation(th, lengths, masses, Tc, num_links: int, nfe: int,
                   h_m: float, title=None):
    """Make an animation of an N link pendulum
    th = in absolute coords"""
    from matplotlib import pyplot as plt
    import matplotlib.animation
    from IPython.display import HTML
    from numpy import sin, cos
    import sys
    import platform

    # if platform.system() == 'Linux':
        # plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    # else:
        # plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\alexa\alknemeyer-msc\windows\ffmpeg\ffmpeg-win32-v3.2.4.exe'

    # some plotting admin
    fig = plt.figure(figsize=(5, 5), dpi=60)
    L = sum(lengths)
    ax = plt.axes(xlim=(-L, L), ylim=(-L, L))
    if title is not None:
        plt.title(title + '\nRed when T > 0, blue when T < 0',
                  fontdict={'fontsize': 18})
    ax.set_aspect('equal')
    ax.axis('off')

    # Define the different elements in the animation
    rods = [ax.plot([], [], color='black', linewidth=2)[0]
            for i in range(num_links)]

    #TC_SCALE = 1/20
    #Tc_circs = [plt.Circle((0, 0), radius=Tc[i, 0]*TC_SCALE, fc='red')
    #           for i in range(num_links)]
    #for circ in Tc_circs:
    #    ax.add_patch(circ)

    def animate(i):
        θ = th[i,0]#th[0, i]

        x = [ sin(θ) * lengths[0]]
        y = [-cos(θ) * lengths[0]]

        rods[0].set_data([0, x[0]], [0, y[0]])

        #Tc_circs[0].set_radius(Tc[i, 0] * TC_SCALE)

        for j in range(1, num_links):
            θ = th[i,j]#th[j, i]

            x.append(x[j-1] + sin(θ) * lengths[j])
            y.append(y[j-1] - cos(θ) * lengths[j])

            rods[j].set_data([x[j], x[j-1]], [y[j], y[j-1]])

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=nfe,
                                              interval=1000*h_m,
                                              repeat_delay=1000)

    if 'ipykernel' in sys.modules:  # or 'IPython'?
        from IPython.core.display import display, HTML
        plt.close(anim._fig)
        display(HTML(anim.to_html5_video()))
    else:
        plt.show(anim._fig)
