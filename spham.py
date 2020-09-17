import functools
import pickle
import os
from numbers import Number
from types import SimpleNamespace
from collections import UserDict
from collections import UserList
from time import perf_counter
from datetime import timedelta
from warnings import warn
from copy import copy
from tqdm import tqdm

# numerical libraries
import numpy as np
import scipy
from scipy import integrate
from scipy import optimize
from scipy import linalg
from scipy import signal
from scipy import sparse

from units import units
from tictoc import TicToc

class Calculation:
    def __init__(self):
        self.conf = SimpleNamespace(
            mtx_fmt_ycal="csc",
        )
        self.data_root = {}


class Spin:
    """ Single spin operators of spin S.

    Attributes
    ----------
    S : spin number (1/2, 1, 3/2, ...)
    plus: Operator $S_+$
    minus: Operator $S_-$
    x: Operator $S_x$
    y: Operator $S_y$
    z: Operator $S_z$
    dim: Dimension of the operator (dim = 2S+1)
    shape: shape of the operator (shape = (dim, dim))
    """

    def __init__(self, S: float):
        self.S = S
        dim = 2 * S + 1
        assert dim % 1 == 0.0, "S must be a multiple of 1/2"
        self.dim = int(dim)
        self.shape = (self.dim, self.dim)
        s = np.linspace(S, -S, self.dim)
        m, n = np.meshgrid(s, s)

        self.plus = δ(n, m + 1) * np.sqrt(S * (S + 1) - n * m)
        self.minus = δ(n + 1, m) * np.sqrt(S * (S + 1) - n * m)
        self.x = 0.5 * (self.plus + self.minus)
        self.y = -0.5j * (self.plus - self.minus)
        self.z = δ(n, m) * m
        self.eigv = np.linalg.eigvalsh(self.z)

        if (self.dim % 2) == 0:  # pair
            self.names = np.array(
                ["\u007c%+d/2\u27e9" % int(i * 2) for i in s])
            self.repr = "Spin(S=%d/2)" % int(self.S * 2)
        else:  # impair
            self.names = np.array(["\u007c%+d\u27e9" % int(i) for i in s])
            self.repr = "Spin(S=%d)" % self.S

    def __repr__(self):
        return self.repr

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.S == other.S
        )
    

def δ(i, j):
    """ Delta function"""
    return (1.0 + 0j) * (i == j)


class KronSpace:
    """ Kronecker space using a list of operators (matrix) or
    states (vectors).

    Can be use like a dictionary or function to lazy evaluate the
    operators in the kronecker space.

    Attributes
    ----------
    dim: the number of operators or states
    shape: the shape of the kronecker space, i.e. the product
        of the shape of all the operators (states)

    Funcions
    --------
    prod(i, j): get the product of two operators
    pow2(i): get the square of an operator
    pow4(i): get the 4th power of an operator
    __call__(i): get the ith operator
    """

    def __init__(self, operators):
        """ Create a Kronecker space using a list of operators (matrix) or
        states (vectors).

        Parameters
        ----------
        operators: a list-like of matrix or vectors
        """
        self._op = operators
        self.dim = len(self._op)
        self.shape = tuple(np.prod([np.asarray(op.shape)
                                    for op in self._op], axis=0))
        self.operators = [self.get_operator(i) for i in range(self.dim)]
        self._prod = {}

    def get_operator(self, n):
        """ Get a specific operator. This is not lazy evaluated."""
        op = sparse.coo_matrix(np.eye(1))
        for i, o in enumerate(self._op):
            if i == n:
                op = sparse.kron(op, o, format="csc")
            else:
                op = sparse.kron(op, np.eye(o.shape[0]), format="csc")
        return op

    def __call__(self, i):
        return self.operators[i]

    def __repr__(self):
        return "KronSpace(dim=%d, shape=%s)" % (self.dim, self.shape)

    def prod(self, i, j):
        if (i, j) in self._prod:
            return self._prod[(i, j)]
        else:
            prod = np.dot(self.operators[i], self.operators[j])
            self._prod[(i, j)] = prod
            self._prod[(j, i)] = prod
            return prod

    def pow2(self, i):
        return np.dot(self.operators[i], self.operators[i])

    def pow4(self, i):
        pow2 = self.pow2(i)
        return np.dot(pow2, pow2)

    @property
    def space(self):
        op = sparse.eye(1)
        for i, o in enumerate(self._op):
            op = sparse.kron(op, o)
        return op

    def __mul__(self, other):
        return KronSpace(self._op + other._op)


class KronSpinSpace:
    """ Kronecker space for spins (Sx, Sy, Sz, S+, S-)"""
    cache = {}

    def __init__(self, spins):
        self.spins = [s if isinstance(s, Spin) else Spin(s) for s in spins]
        self.dim = len(self.spins)
        self.shape = tuple(np.prod([s.shape for s in self.spins], axis=0))
        self.x = KronSpace([s.x for s in self.spins])
        self.y = KronSpace([s.y for s in self.spins])
        self.z = KronSpace([s.z for s in self.spins])
        self.plus = KronSpace([s.plus for s in self.spins])
        self.minus = KronSpace([s.minus for s in self.spins])

    def __repr__(self):
        return "KronSpinSpace(dim=%d, shape=%s)" % (self.dim, self.shape)

    def __add__(self, k):
        if isinstance(k, KronSpinSpace):
            return KronSpinSpace([s for s in self.spins] + [s for s in k.spins])
        else:
            return KronSpinSpace([s for s in self.spins] + [k, ])


class KronAtomSpace:
    """ Kronecker space for atoms (Sx, Sy, Sz, S+, S-)"""
    cache = {}

    def __init__(self, atoms, with_electron=False, precalc=False):
        self.atoms = atoms
        spins = []
        indexes = []
        i = 0
        for atom in atoms:
            if isinstance(atom, SpinAtom):
                spins.append(atom.S)
                indexes.append(i)
                i = i + 1
            elif isinstance(atom, SpinOrbitAtom):
                spins.append(atom.S)
                spins.append(atom.L)
                indexes.append(i)
                i = i + 2
            else:
                warn("Unknow atom type %s at pos %d" % (atom, len(indexes)))

        if with_electron:
            spins.append(electron)
        self.indexes = tuple(indexes)
        self.spins = [s if isinstance(s, Spin) else Spin(s) for s in spins]
        self.dim = len(self.spins)
        self.shape = tuple(np.prod([s.shape for s in self.spins], axis=0))
        self.x = KronSpace([s.x for s in self.spins])
        self.y = KronSpace([s.y for s in self.spins])
        self.z = KronSpace([s.z for s in self.spins])
        self.plus = KronSpace([s.plus for s in self.spins])
        self.minus = KronSpace([s.minus for s in self.spins])

        if precalc is not False:
            N = len(spins)
            if (precalc is True) or (precalc == "all"):
                for i in range(N):
                    for j in range(N):
                        self.x.prod(i, j)
                        self.y.prod(i, j)
                        self.z.prod(i, j)
                        self.plus.prod(i, j)
                        self.minus.prod(i, j)
            elif (precalc == "loop") or (precalc == "line"):
                for i in range(N-1):
                    self.x.prod(i, i+1)
                    self.y.prod(i, i+1)
                    self.z.prod(i, i+1)
                    self.plus.prod(i, i+1)
                    self.minus.prod(i, i+1)
                if precalc == "loop":
                    self.x.prod(0, N-1)
                    self.y.prod(0, N-1)
                    self.z.prod(0, N-1)
                    self.plus.prod(0, N-1)
                    self.minus.prod(0, N-1)

    def __repr__(self):
        return "KronAtomSpace(dim=%d, shape=%s)" % (self.dim, self.shape)

    def __add__(self, k):
        if isinstance(k, KronSpinSpace):
            return KronSpinSpace([s
                                  for s in self.spins] + [s for s in k.spins])
        else:
            return KronSpinSpace([s for s in self.spins] + [
                k,
            ])


electron = Spin(1/2)


class SpinAtom:
    """ A single atom description with standard anisotropy"""

    def __init__(
            self,
            S,  # spin (ex.: 5/2, 2)
            g=0.0,  # g-factor
            D=0.0,  # axial anisotropy
            E=0.0,  # plane anisotropy
    ):
        self.S = S if isinstance(S, Spin) else Spin(S)
        self.g = g
        self.D = D
        self.E = E

    def __repr__(self):
        return "SpinAtom(S=%d, g=%g, D=%g, E=%g)" % (
            self.S.S, self.g, self.D, self.E)

    def ham(self, i: int, k: KronSpinSpace, Bx, By, Bz):
        """ Calculate the hamiltonian for a single spin atom.

        Parameters
        ----------
        i: int
            The position of the atom in the Kronecker space
        k: KronSpinSpace
            The Kronecker space
        Bx, By, Bz: Float
            The magnetic field
        """
        # zeeman
        gish = -self.g * units.μB
        zx = gish * Bx * k.x(i) if Bx != 0.0 else 0.0
        zy = gish * By * k.y(i) if By != 0.0 else 0.0
        zz = gish * Bz * k.z(i) if Bz != 0.0 else 0.0

        # anisotropy
        ae = self.E * (k.x.pow2(i) - k.y.pow2(i))
        ad = self.D * k.z.pow2(i)

        return zx + zy + zz + ae + ad

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.S == other.S and
            self.g == other.g and
            self.D == other.D and
            self.E == other.E
        )


class SpinOrbitAtom:
    """ A single and orbit atom description with Stevens Operator anisotropy"""

    def __init__(
            self,
            S,  # Spin
            L,  # Orbital momentum
            B20=0.0,  # Stevens operator B_2^4
            B40=0.0,  # Stevens operator B_0^4
            B44=0.0,  # Stevens operator B_4^4
            λso=0.0,  # Spin orbit coupling
    ):
        self.S = S if isinstance(S, Spin) else Spin(S)
        self.L = L if isinstance(L, Spin) else Spin(L)
        self.B20 = B20
        self.B40 = B40
        self.B44 = B44
        self.λso = λso

    def __repr__(self):
        return "SpinOrbitAtom(S=%d, L=%d, B20=%g, B40=%g, B44=%g, λso=%g)" % (
            self.S.S, self.L.S, self.B20, self.B40, self.B44, self.λso)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.S == other.S and
            self.L == other.L and
            self.B20 == other.B20 and
            self.B40 == other.B40 and
            self.B44 == other.B44 and
            self.λso == other.λso
        )

    def ham(self, i: int, k: KronSpinSpace, Bx, By, Bz):
        """ Calculate the hamiltonian for a single spin atom.

        Parameters
        ----------
        i: int
            The position of the atom in the Kronecker space
        k: KronSpinSpace
            The Kronecker space
        Bx, By, Bz: Float
            The magnetic field
        """
        N = k.shape[0]
        Id = sparse.identity(N)

        S = self.L.S
        s = S * (S + 1)

        j = i + 1

        # Stevens operators
        s20 = self.B20 * (3 * k.z.pow2(j) - s * Id)
        s40 = self.B40 * (35 * k.z.pow4(j) - (30 * s - 25) * k.z.pow2(j) +
                          (3 * s**2 - 6 * s) * Id)
        s44 = self.B44 * 0.5 * (k.plus.pow4(j) + k.minus.pow4(j))

        # Spin-orbit coupling
        soc = self.λso * (k.x.prod(j, i) + k.y.prod(j, i) + k.z.prod(j, i))

        # Zeeman
        zx = units.μB * (2 * k.x(i) + k.x(j)) * Bx if Bx != 0.0 else 0.0
        zy = units.μB * (2 * k.y(i) + k.y(j)) * By if By != 0.0 else 0.0
        zz = units.μB * (2 * k.z(i) + k.z(j)) * Bz if Bz != 0.0 else 0.0

        return s20 + s40 + s44 + soc + zx + zy + zz


def hamcalc(
    k,  # the kronecker space of the atoms object list
    Bx,  # field in x
    By,  # field in y
    Bz,  # field in z
    Ji,  # Heisenberg coupling
):
    """ Calculating the Hamiltonian."""

    # k = KronAtomSpace(atoms) if k is None else k
    atoms = k.atoms
    indexes = k.indexes
    Natoms = len(indexes)
    h = sparse.coo_matrix(k.x.shape)

    Bx = np.full(Natoms, Bx) if isinstance(Bx, Number) else np.asarray(Bx)
    By = np.full(Natoms, By) if isinstance(By, Number) else np.asarray(By)
    Bz = np.full(Natoms, Bz) if isinstance(Bz, Number) else np.asarray(Bz)

    for b, n in [(Bx, "Bx"), (By, "By"), (Bz, "Bz")]:
        if b.size != Natoms:
            warn("%s field array len not egal to the number of atoms" % n)

    for i in range(Natoms):
        h += atoms[i].ham(indexes[i], k, Bx[i], By[i], Bz[i])

    for (ii, jj), J in Ji.items():
        i = indexes[ii]
        j = indexes[jj]
        h += J * (k.x.prod(i, j) + k.y.prod(i, j) + k.z.prod(i, j))

    return h


def eigencalc(ham, maxstates):
    N = ham.shape[0]

    if (N - 1) > maxstates:
        v, w = sparse.linalg.eigsh(ham, k=maxstates)
    else:
        v, w = linalg.eigh(ham.todense())
    argsort = np.argsort(v)
    return v[argsort], w[:, argsort]


def Ycalc(tippos: int,  # tunneling spin
          #u: float,  # elastic tunneling
          ke: KronSpinSpace,  # The kronecker space with the electron
          eigvec,
          u = 0.0,
          mxt_fmt="csc",
          use_sparse=False,
          with_progress=False,
          ):
    
    t = TicToc()
    
    indexes = ke.indexes
    fmt_Sσ = mxt_fmt
    fmt_ϕσ = mxt_fmt

    j = ke.dim - 1  # electron position

    t.tic("Sσ_s")
    Sσ_s = sparse.coo_matrix(ke.x.shape)
    for i in range(j):
        Sσ_s += (ke.x.prod(i, j) + ke.y.prod(i, j) + ke.z.prod(i, j))
    Sσ_s = Sσ_s.asformat(fmt_Sσ)
    t.toc("Sσ_s")

    t.tic("Sσ_t")
    tip = indexes[tippos]
    
    Sσ_t = (ke.x.prod(tip, j) + ke.y.prod(tip, j) + ke.z.prod(tip, j))
    
    if u != 0.0:
        Sσ_t += u * sparse.eye(Sσ_t.shape[0])
        Sσ_s += u * sparse.eye(Sσ_s.shape[0])
    Sσ_t = Sσ_t.asformat(fmt_Sσ)
    t.toc("Sσ_t")

    t.tic("ϕσ")
    
    ϕσ = sparse.kron(eigvec, np.array([[1,0], [0, 1]]), format="csc")
    t.toc("ϕσ")

    N = eigvec.shape[1]

    Ys = np.empty((2*N, 2*N), dtype=np.complex)
    Yt = np.empty((2*N, 2*N), dtype=np.complex)

    if use_sparse:
        def _get(k):
            return ϕσ.getcol(k)
    else:
        if sparse.issparse(ϕσ):
            ϕσ = ϕσ.todense()

        def _get(k):
            return ϕσ[:, k]

    lst = list(range(2*N))
    if with_progress:
        lst = tqdm(lst)
        
    for i in lst:
        m = _get(i)
        right_s = Sσ_s.dot(m)
        right_t = Sσ_t.dot(m)
        for j in range(i+1):
            m = _get(j).getH()
            Ys[i, j] = m.dot(right_s)[0, 0]
            Yt[i, j] = m.dot(right_t)[0, 0]
            if i != j:
                Ys[j, i] = Ys[i, j]
                Yt[j, i] = Yt[i, j]
                
    return SimpleNamespace(s = np.square(np.abs(Ys)), 
                           t = np.square(np.abs(Yt)))

def Pcalc(Ys,  # transition matrix (sample)
          Yt,  # transition matrix (tip)
          η,  # tip polarization
         ):
    ηp  = 0.5 + η/2
    ηm  = 0.5 - η/2
    
    Ys = Ys.copy()
    Yt = Yt.copy()
    
    Ys /= np.sum(Ys, axis=0)
    Yt /= np.sum(Yt, axis=0)
    
    
    N = int(Ys.shape[0]/2)
    ts = np.empty((N, N))
    st = np.empty((N, N))
    ss = np.empty((N, N))
            
    for i in range(N):
        for j in range(N):
            Y00 = Yt[2*i  , 2*j  ]
            Y01 = Yt[2*i  , 2*j+1]
            Y10 = Yt[2*i+1, 2*j  ]
            Y11 = Yt[2*i+1, 2*j+1]
            
            ts[i, j] = Y00*ηp + Y01*ηm + Y10*ηp + Y11*ηm
            st[i, j] = Y00*ηp + Y01*ηp + Y10*ηm + Y11*ηm
            
            Y00 = Ys[2*i  , 2*j  ]
            Y01 = Ys[2*i  , 2*j+1]
            Y10 = Ys[2*i+1, 2*j  ]
            Y11 = Ys[2*i+1, 2*j+1]
            
            ss[i, j] = Y00 + Y01 + Y10 + Y11
            
    #for i in range(N):
    #    for m in [ts, st, ss]:
    #        mi = m[:, i]
    #        m[:, i] = mi / np.sum(mi) 
            
    return SimpleNamespace(ts=ts, st=st, ss=ss)

_maxexp = np.log(np.finfo(np.float64).max)

def fermish(dE, β):
    dEβ = dE * β
    out = np.empty(dE.shape)
    
    dE0 = dE == 0.0
    dEinf = dEβ > _maxexp
    dEexp = np.logical_not(np.logical_or(dE0, dEinf))
    
    out[dE0] = 1 / β
    out[dEinf] = 0.0
    out[dEexp] = dE[dEexp] / np.expm1(dEβ[dEexp])
    
    return out


def ratescalc(G0,  # conductance at 0V
              b0,  # Fraction of conduction electron
              Gs,  # sample conductance
              eigs,  # Eigen value of the hamiltonian (the states' energy)
              bias, # bias
              β,  # beta = 1/(kB*T) 
              Pss,  # transition matrix
              Pts,  # transition matrix
              Pst,  # transition matrix
): 
    Gp = (1 - b0) * G0
    
    N = eigs.shape[0]
    
    j, i = np.meshgrid(range(N), range(N))
    dE = eigs[j] - eigs[i] 

    ts = fermish(dE - bias, β)
    st = fermish(dE + bias, β)
    ss = fermish(dE       , β)
            
    return SimpleNamespace(
        ts=ts*Gp*Pts, 
        st=st*Gp*Pst,
        ss=ss*Gs*Pss )


def steadystatescalc_nnls(
    rij,  # rates matrixes
):
    N = rij.shape[0]
    #A = np.full((N+1, N), 1.0)
    #r = rij.T
    #rjsum = np.sum(r, axis=0)
    #for i in range(N):
    #    for j in range(N):
    #        A[i, j] = r[i, j] - (rjsum[i] if i == j else 0.0)
    #b = np.zeros((N + 1, ))
    #b[-1] = 1.0
    
    A = np.full((N+1, N), 1.0)
    A[:N, :N] = getA(rij)
    #A[-1, -1] = 0.0
    b = np.zeros(N+1)
    b[-1] = 1.0
    
    n, residuals = optimize.nnls(A, b, maxiter=100*A.shape[1])
    #n, residuals = optimize.nnls(rij, np.sum(rij, axis=0), maxiter=10*A.shape[1])
    
    return n[:N] / np.sum(n[:N])


def steadystatescalc_rk23(
    rij,  # rates matrixes
    init_states = None,  # if None, create a [1, 0, 0...., 0] array
):
    #rij = rates.ts + rates.st + rates.ss
    N = rij.shape[0]
    if init_states is None:
        init_states = np.zeros(N) 
        init_states[0] = 1.0
    A = getA(rij, ext=True)
    # RK45 init
    func = lambda t, x: A @ x
    rk = scipy.integrate.RK23(
        #lambda t, x: x @ A,
        func,
        0.0, 
        init_states, 
        1e11,
        rtol = 0.000001,
        atol = 1e-10,
    )
    #y = [rk.y, ]
    #t = [rk.t, ]
    for i in range(1000000):
        rk.step()
        y = rk.y
        #t.append(rk.t)
        if ((i > 1000) and np.all(np.abs(func(0, y)) < 5e-11)):
            break
        if rk.status in ["finished", "failed"]:
            print('failed')
            break
    else:
        warn("Maximum number of steps for RK45 reached")
    return y #, np.array(y), np.array(t)


def steadystatescalc_firstorder(
    rij,  # rates matrixes
    init_states = None,  # if None, create a [1, 0, 0...., 0] array
    dt0 = 0.1*units["ps"],
    dt1 = 10*units["ps"],
    max_steps = 100000,
    max_error = 1e-10,
):
    #rij = rates.ts + rates.st + rates.ss
    N = rij.shape[0]
    if init_states is None:
        init_states = np.zeros(N) 
        init_states[0] = 1.0
    r = rij.T
    A = r - np.diag(np.sum(r, axis=0))
    Δt = np.linspace(dt0, dt1, max_steps)
    x = [init_states, ]
    for i in range(1, max_steps):
        Ax = A @ x[-1]
        x.append(x[-1] + Δt[i]*Ax)
        if np.all(np.abs(Ax) < max_error):
            t = Δt[:i+1]
            break
        else:
            warn("Maximum number of steps reached")
            t = Δt
    return x[-1]

def steadystatescalc_firstorder_old(
    rij,  # rates matrixes
    init_states = None,  # if None, create a [1, 0, 0...., 0] array
    dt0 = 0.1*units["ps"],
    dt1 = 10*units["ps"],
    max_steps = 100000,
    max_error = 1e-10,
):
    #rij = rates.ts + rates.st + rates.ss
    N = rij.shape[0]
    if init_states is None:
        init_states = np.zeros(N) 
        init_states[0] = 1.0
    r = rij.T
    A = r - np.diag(np.sum(r, axis=0))
    Δt = np.linspace(dt0, dt1, max_steps)
    x = [init_states, ]
    for i in range(1, max_steps):
        Ax = A @ x[-1]
        x.append(x[-1] + Δt[i]*Ax)
        if np.all(np.abs(Ax) < max_error):
            t = Δt[:i+1]
            break
        else:
            warn("Maximum number of steps reached")
            t = Δt
    return x[-1]

def getA(rij):
    return (rij - np.diag(np.sum(rij, axis=1))).T


def steadystatescalc_null_space(
    rij,  # rates matrixes
):
    #rij = rates.ts + rates.st + rates.ss
    A = getA(rij)
    y = scipy.linalg.null_space(A, rcond=1e-15)[:,0]
    #init_states = np.zeros(A.shape[0]) 
    #init_states[0] = 1.0
    #return init_states
    return y / np.sum(y)

def steadystatescalc_root(
    rij,  # rates matrixes
    init_states = None,
):
    #rij = rates.ts + rates.st + rates.ss
    N = rij.shape[0]
    if init_states is None:
        init_states = np.zeros(N) 
        init_states[0] = 1.0
    r = rij.T / rij.max()
    A = r - np.diag(np.sum(r, axis=0))
    #y = scipy.optimize.root(lambda x: A @ (np.abs(x) / np.sum(np.abs(x))), init_states, tol=1e-20).x
    y = scipy.optimize.fmin(lambda x: np.sum(np.square(A @ x)), init_states, ftol=1e-20, xtol=1e-11, disp=False)
    return y

def steadystatescalc_randomwalk(
    rij,  # rates matrixes
    init_states = None,
):
    #rij = rates.ts + rates.st + rates.ss
    A = getA(rij)
    p = np.cumsum(rij.T, axis=0)
    p /= p[-1, :]
    
    N = A.shape[0]
    #steps = 2**20
    steps = 2**23
    
    occ = np.empty(steps, dtype=int)
    occ[0] = 0
    rand = np.random.rand(steps)
    _random_walk(steps, p, occ, rand)
    x = np.bincount(occ, minlength=N)
    return x / np.sum(x)

        
meV = units.meV
Fe_CuN = SpinAtom(S=2, g=2.11, D=-1.57 * meV, E=0.31 * meV)

def D7(g=2.11,
       D=-1.57 * meV,
       E=0.31 * meV,
       Jz=0.7 * meV,
       Jx=0.3 * meV,
       Jxz=-0.69 * meV,
       ):
    N = 7
    atoms = [SpinAtom(S=2, g=g, D=D, E=E) for i in range(N)]
    _Ji = [Jz, Jz, Jx, Jxz, Jz, Jxz, Jx]
    Ji = {(i, i + 1): _Ji[i] for i in range(N - 1)}
    Ji[(0, N - 1)] = _Ji[N - 1]

    return atoms, Ji

def trimer(
       g=2.11,
       D=-1.57 * meV,
       E=0.31 * meV,
       Jz=0.7 * meV,
       ):
    N = 3
    atoms = [SpinAtom(S=2, g=g, D=D, E=E) for i in range(N)]
    Ji = {(i, i + 1): _Jz for i in range(N - 1)}

    return atoms, Ji

def FeOnCuDE(
    g = 2.11,
    D = -1.57 * meV,
    E = 0.31 * meV,
):
    N = 1
    atoms = [SpinAtom(S=2, g=g, D=D, E=E) for i in range(N)]
    Ji = {}
    
    return atoms, Ji

def FeOnNDE(
    g = 2.0,
    D = -6.15 * meV,
    E = -0.11 * meV,
):
    N = 1
    atoms = [SpinAtom(S=2, g=g, D=D, E=E) for i in range(N)]
    Ji = {}
    
    return atoms, Ji

def FeOnNStevens(
    S = 2, 
    L = 2,
    B20 = -6.35 * meV,
    B40 = -0.637 * meV,
    B44 =  0.194 * meV,
    λso = -49.8 * meV,
    
):
    N = 1
    atoms = [SpinOrbitAtom(S=S, L=L, B20=B20, B40=B40, B44=B44, λso=λso)
             for i in range(N)]
    Ji = {}
    return atoms, Ji
