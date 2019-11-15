# python 2/3 compatibility stuff:
from __future__ import print_function, division

import sympy as sp, numpy as np
from sympy import Matrix as Mat
sp.init_printing()

from typing import List, Tuple#, Iterable, Literal

# derivatives ################################################################
def deriv(expr, q, dq):
    """Take the time derivative of an expression `expr` with respect to time,
    handling the chain rule semi-correctly"""
    return (sp.diff(expr, q).T * dq)[0]

def full_deriv(var, q, dq, ddq):
    return deriv(var, q, dq) + deriv(var, dq, ddq)


# rotations ##################################################################
def rot_x(θ) -> Mat:
    return sp.Matrix([
        [ 1,        0,           0],
        [ 0, sp.cos(θ),  sp.sin(θ)],
        [ 0,-sp.sin(θ),  sp.cos(θ)],
    ])

def rot_y(θ) -> Mat:
    return sp.Matrix([
        [ sp.cos(θ), 0,-sp.sin(θ)],
        [         0, 1,         0],
        [ sp.sin(θ), 0, sp.cos(θ)],
    ])

def rot_z(θ) -> Mat:
    return sp.Matrix([
        [ sp.cos(θ), sp.sin(θ), 0],
        [-sp.sin(θ), sp.cos(θ), 0],
        [        0,         0, 1],
    ])

def euler_321(phi, theta, psi) -> Mat:
    return rot_x(phi) @ rot_y(theta) @ rot_z(psi)


# utils to find equations of motion ##########################################
def skew_symmetric(Rx_I, q, dq) -> Mat:
    """Rx_I is the 3x3 rotation from body frame `x` to the inertial `I`
    `q` is a vector of symbolic variables
    `dq` is a vector of the derivates of `q`
    
    The return matrix can be simplified fairly quickly with `trigsimp` if
    it's made up of ~3 or fewer rotations
    """
    dRx_I = sp.zeros(3, 3)
    
    for i in range(3):
        dRx_I[:, i] = Rx_I[:, i].jacobian(q) * dq
    
    omega_Rx = Rx_I.T @ dRx_I
    return Mat([
        omega_Rx[2,1],
        omega_Rx[0,2],
        omega_Rx[1,0]
    ])

def manipulator_equation(Ek, Ep, q, dq):
    """Ek and Ep are the kinetic and potential energy of the system.
    They must be passed in as sp.Matrix types (or at least have a
    `.jacobian` method defined on them)"""
    M = sp.hessian(Ek, dq)
    N = M.shape[0]

    dM = sp.zeros(N, N)
    for i in range(N):
        for j in range(N):
            dM[i,j] = Mat([M[i,j]]).jacobian(q) @ dq

    C = dM @ dq - Ek.jacobian(q).T

    G = Ep.jacobian(q).T
    
    return M, C, G

def calc_velocities_and_energies(
        positions: List[Mat], rotations: List[Mat],
        masses: List[float], inertias: List[Mat],
        q, dq, g: float = 9.81):
    """Calculate and return  the kinetic and potential energies of
    a system, given lists of:
        - positions of each body (in inertial)
        - rotations for each body (from body to inertial)
        - mass of each body
        - inertia of each body (given as a 3x3 matrix)
        - q (each state, such as q = [x, y, theta]
        - dq (derivative of each state, such as dq = [dx, dy, dtheta]
    """
    from functools import reduce
    
    dPs = [Mat(Px_I.jacobian(q) * dq) for Px_I in positions]
    ang_vels = [
        skew_symmetric(Rx_I, q, dq) for Rx_I in rotations
    ]
    # this should be sum(, but it inits the sum with the int 0, which can't
    # be added to matrices
    Ek = reduce(lambda a,b: a + b, [
        dPx_I.T * mx * dPx_I / 2 + dωx_I.T * Ix * dωx_I / 2
           for (dPx_I, mx, dωx_I, Ix) in zip(dPs, masses, ang_vels, inertias)
    ])
    Ep = Mat([sum(m * Mat([0, 0, g]).dot(p) for (m,p) in zip(masses, positions))])
    
    return Ek, Ep, dPs, ang_vels

def friction_polygon(nsides: int = 8):
    assert nsides == 8, "Only 8-sided polygon implemented at the moment"
    D = np.array([
         1, 0, 0,
         1, 1, 0,
         0, 1, 0,
        -1, 1, 0,
        -1, 0, 0,
        -1,-1, 0,
         0,-1, 0,
         1,-1, 0,
    ]).reshape(8, 3)
    
    # normalize and return
    return D / np.linalg.norm(D, axis=1).reshape(8,1)

def lambdify_EOM(EOM, vars_in_EOM, display_vars: bool = True,
                 test_func: bool = True):
    """ Returns a list of functions which, when called with arguments which match
    `vars_in_EOM`, will evaluate the equations of motion specified in `EOM`.
    `display_vars` specifies whether to print out `vars_in_EOM` as a sanity check.
    `test_func` specifies whether to test that the function returns a float.

        >>> lambdify_EOM(EOM, vars_in_EOM)
        [<function _lambdifygenerated(_Dummy_833, _Dummy_834, ...
         <function _lambdifygenerated(_Dummy_851, _Dummy_852, ... 
         <function _lambdifygenerated(_Dummy_923, _Dummy_924, ...]

        >>> lambdify_EOM(EOM, vars_in_EOM[:-3])  # not all vars in EOM
        AssertionError: The function didn't return a float - it's likely ...
        """
    import pyomo.environ
    import random

    if display_vars is True:
        try:
            from IPython.core.display import display
            display(vars_in_EOM)
        except:
            print(vars_in_EOM)
    
    func_map = [{'sin': pyomo.environ.sin, 'cos': pyomo.environ.cos}]
    
    if EOM.is_Matrix is False:
        EOM = sp.Matrix([EOM])
    
    funcs = [sp.lambdify(vars_in_EOM, eqn, modules=func_map) for eqn in EOM]
    
    # replace with set(EOM.free_symbols).difference(set(vars_in_EOM))?
    if test_func is True:
        vals = [random.random() for _ in range(len(vars_in_EOM))]
        for func in funcs:
            ret = func(*vals)
            assert type(ret) == float, "The function didn't return a float - it's likely "\
                                        "because there are symbolic variables in the EOM "\
                                        "which weren't specified in `vars_in_EOM`. Got: " + str(ret)
    
    return funcs[0] if len(funcs) == 1 else funcs


# simplification #################################################################
import multiprocessing
def parsimp_worker(_arg, allow_recur: bool = True):
    expr, simp_func, idx = _arg
    args = []

    for (i, arg) in enumerate(expr.args):
        if allow_recur is True and sp.count_ops(arg) > 200:
            arg = parsimp_worker((arg, simp_func, -1),
                                 allow_recur=False)
        
        args.append(simp_func(arg))

    return expr.func(*args)

def parsimp(mat: sp.Matrix, nprocs: int, f = sp.trigsimp):
    with multiprocessing.Pool(nprocs) as p:
        return sp.Matrix(
           p.map(parsimp_worker, [(v,f,i) for (i,v) in enumerate(mat)])
        ).reshape(*mat.shape)

#     import sys
#     outvals = []
#         for i, val in enumerate(p.imap_unordered(parsimp_worker,
#                                 [(v,f,i) for (i,v) in enumerate(mat)]), 1):
#             outvals.append(val)
#             if disp_progress is True:
#                 sys.stdout.write('\rSimplifying.... {0:%} done'.format(i/len(vec)))
#     return sp.Matrix(outvals).reshape(vec.shape)


def prune_small(expr, eta=1e-13):
    """Prunes small elements (ie, less than `eta=1e-15`) from an expression"""
    new_args = []
    for idx, val in enumerate(expr.args):
        if val.is_number and abs(val) < eta:
            new_args.append(sp.Integer(0))
        else:
            new_args.append(prune_small(val))
    
    return expr.func(*new_args) if len(new_args) > 0 else expr


# interpolation ##############################################################
def implicit_euler(q, dq):  # q = θ or dθ
    from pyomo.environ import Constraint
    def func(m, fe, cp, var):
        # assert cp == 1 and m.cp[-1] == 1
        if fe > 1:
            return q[fe,cp,var] == q[fe-1,cp,var] + m.hm0 * m.hm[fe] * dq[fe,cp,var]
        else:
            return Constraint.Skip
    return func

def radau_3(q, dq):  # q = θ or dθ
    from pyomo.environ import Constraint
    R = [
        [ 0.19681547722366, -0.06553542585020,  0.02377097434822],
        [ 0.39442431473909,  0.29207341166523, -0.04154875212600],
        [ 0.37640306270047,  0.51248582618842,  0.11111111111111]
    ]
    def func(m, fe, cp, var):
        if fe > 1:
            inc = sum(R[cp-1][pp-1]*dq[fe,pp,var] for pp in m.cp)
            return q[fe,cp,var] == q[fe-1,m.cp[-1],var] + m.hm0*m.hm[fe] * inc
        else:
            return Constraint.Skip
    return func

def radau_dummy_version(m):
    from pyomo.environ import Var, Constraint
    m.radau_dummy_q  = Var(m.fe, m.cp, m.vars)
    m.radau_dummy_dq = Var(m.fe, m.cp, m.vars)
    def radau_3_fix_dummy(q, dq, dummy):  # q = θ or dθ
        R = [
            [ 0.19681547722366, -0.06553542585020,  0.02377097434822],
            [ 0.39442431473909,  0.29207341166523, -0.04154875212600],
            [ 0.37640306270047,  0.51248582618842,  0.11111111111111]
        ]
        def func(m, fe, cp, var):
            if fe > 1:
                inc = sum(R[cp-1][pp-1]*dq[fe,pp,var] for pp in m.cp)
                return dummy[fe,cp,var] == m.hm0 * m.hm[fe] * inc
            else:
                return Constraint.Skip
        return func
    m.interp_q_d  = Constraint(m.fe, m.cp, m.vars,
                               rule=radau_3_fix_dummy(m.q,  m.dq,  m.radau_dummy_q))
    m.interp_dq_d = Constraint(m.fe, m.cp, m.vars,
                               rule=radau_3_fix_dummy(m.dq, m.ddq, m.radau_dummy_dq))

    def radau_3_dummy(q, dq, dummy):  # q = θ or dθ
        def func(m, fe, cp, var):
            if fe > 1:
                return q[fe,cp,var] == q[fe-1,m.cp[-1],var] + dummy[fe,cp,var]
            else:
                return Constraint.Skip
        return func
    m.interp_q  = Constraint(m.fe, m.cp, m.vars,
                             rule=radau_3_dummy(m.q,  m.dq,  m.radau_dummy_q))
    m.interp_dq = Constraint(m.fe, m.cp, m.vars,
                             rule=radau_3_dummy(m.dq, m.ddq, m.radau_dummy_dq))

# other utils for pyomo ######################################################
def get_vals(m, q, idxs=None):
    """Always assumes first dimension is `fe`!
    
    >>> get_vals(m.q, m.var_set)
    >>> get_vals(m.Tc)
    """
    ncp = m.cp[-1]
    if idxs is None:
        if q.dim() == 1:
            return np.array([q[fe].value for fe in m.fe])
        elif q.dim() == 2:
            return np.array([q[fe,ncp].value for fe in m.fe])
        else:
            raise IndexError()
    
    else:
        if q.dim() == 2:
            return np.array([q[fe,idx].value for fe in m.fe
                                             for idx in idxs]).reshape(m.fe[-1],-1)
        elif q.dim() == 3:
            return np.array([q[fe,ncp,idx].value for fe in m.fe
                                                 for idx in idxs]).reshape(m.fe[-1],-1)
        else:
            raise IndexError()


def get_vals2(m, var, idxs):
    """Always assumes first dimension is `fe`!
    >>> get_vals(m.q, m.var_set)
    >>> get_vals(m.Tc)
    """
    nfe = m.fe[-1]
    ncp = m.cp[-1]
    arr = np.array([var[idx].value for idx in var]).astype(float)
    
    idxs = [len(i) for i in idxs]
    
    # if m.cp wasn't given as an index
    idx_dims = next(iter(var))
    if len(idx_dims) != len(idxs) + 1:
        return arr.reshape(nfe, ncp, *idxs)
    else:
        return arr.reshape(nfe, *idxs)

class MarkovBinaryWalk():
    """Generate a Markov process random walk which alternates between two states:
    `a` and `b`. If in state `a`, return 1 then stay in state `a with probability
    `a_prob`. Otherwise, go to state `b`. If in state `b`, return 0 and stay in
    state `b` with probability `b_prob"""
    def __init__(self, a_prob, b_prob):
        self.a_prob = a_prob
        self.b_prob = b_prob
        self.state = 'a'

    def step(self):
        import random
        if self.state == 'a':
            self.state = 'a' if random.random() < self.a_prob else 'b'
            return 1
        else:
            self.state = 'b' if random.random() < self.b_prob else 'a'
            return 0
    
    def walk(self, n):
        return np.array([self.step() for _ in range(n)], dtype=int)

# utils for matplotlib #######################################################
def plot3d_setup(figsize=(10, 10),
                 dpi: int = 60,
                 title: str = '',
                 lim: float = 2.0,
                 height: float = 5.0,
                 show_grid: bool = True,
                 plot_ground: bool = True,
                 ground_lims: tuple = (),
                 scale_plot_size: bool = True):
    """Set up a fairly "standard" figure for a 3D system, including:
    - setting up the title correctly
    - setting limits on and labelling axes
    - optionally removing the grid
    - plotting the ground (at z = 0m)
    - scaling the plot to reduce surrounding whitespace
    
    Returns a figure and axis
    """
    from matplotlib import pyplot as plt
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca(projection='3d')
        
    if len(title) > 0:
        fig.suptitle(title, fontsize=20)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, height)
    ax.set_xlabel('$X$ [m]'); ax.set_ylabel('$Y$ [m]'); ax.set_zlabel('$Z$ [m]')
    
    if show_grid is False:
        ax.grid(False)     # hide grid lines
        ax.set_xticks([])  # hide axes ticks
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')    # hmmm
        plt.grid(b=None)   # also, hmm
    
    if plot_ground is True:
        import numpy as np
        if ground_lims == ():
            ground_lims = [-10*lim, lim], [-lim, 10*lim]
        ax.plot_surface(*np.meshgrid(*ground_lims),#was: (-2,1) (-1, 3)
                        np.zeros((2,2)), alpha=0.5, color='green', zorder=1)
    
    if scale_plot_size is True:
        fig.subplots_adjust(left=-0.25, bottom=-0.25, right=1.25, top=1.25,
                            wspace=None, hspace=None)
    
    return fig, ax

def update_3d_line(line, pt1, pt2):
    """Update data in a 3D line, passed as two points"""
    line.set_data([[pt1[0], pt2[0]],
                   [pt1[1], pt2[1]]])
    line.set_3d_properties([pt1[2], pt2[2]])

def set_view(ax, along = (45, 150)):
    """Set the angle for the 'camera' in matplotlib"""
    if type(along) == tuple:
        ax.view_init(elev=along[0], azim=along[1])
    
    else:
        assert along in ('x', 'y', 'z')
        if along == 'x':
            ax.view_init(elev=0, azim=0)
            ax.set_xlabel('')
        elif along == 'y':
            ax.view_init(elev=0, azim=-90)
            ax.set_ylabel('')
        elif along == 'z':
            ax.view_init(elev=90, azim=0)
            ax.set_zlabel('')

def track_pt(ax, pt, lim):
    """Adjust the limits of a plot so as to track a 2D or 3D point"""
    ax.set_xlim(pt[0]-lim, pt[0]+lim)
    ax.set_ylim(pt[1]-lim, pt[1]+lim)
    if len(pt) > 2:
        ax.set_zlim(pt[2]-lim, pt[2]+lim)


# post-analysis ##############################################################
import textwrap
def check_constraints(pyomo_constraints: list, max_per_constr: int = 5,
                      tol: float = 1e-6, textwrap_width: int = 150):
    """Checks that the indexed Constraints in the list `pyomo_constraints`
    don't violate `tol`. The constraints can be equality, upper bound or
    lower bound
    
    Note: doesn't yet work with bounded/fixed variables"""
    for con in pyomo_constraints:
        counter = 0
        for idx in con:
            body = con[idx].body()
            
            # TODO: check this!
            if con[idx].equality is True:
                constr_violated = abs(body) > tol
            if con[idx].has_ub():
                constr_violated = body > con[idx].upper.value + tol
            if con[idx].has_lb():
                constr_violated = body < con[idx].lower.value - tol

            if constr_violated is True:
                print('constraint:', con, '| idx:', idx, '| body:', body)
                print(textwrap.shorten(con[idx].expr.to_string(),
                                       width=textwrap_width))
                print()
                
                counter += 1
                if counter > max_per_constr:
                    print('Too many violated', con[idx], 'constraints - skipping the rest')
                    break
