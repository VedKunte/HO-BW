import numpy as np
import sys 
from fractions import Fraction
from math import lcm
from itertools import permutations, product, compress
from operator import add

o = sys.stdout


def generate_g_states(num_in, num_out, method = None, reduce = True):
    """Generates the vertices of the state space for a given g-system.
    
    Parameters
    ----------
    num_in : int64
        The number of settings/ fiducial measurements.
    num_out : int64
        Number of outcomes for each setting/ fiducial measurement. 
        Currently only defined for the case where each of the settings 
        has the same number of outcomees.
    method : str. Either 'Pauli' or 'Probability
        Chooses which representation to use.
    reduce : bool
        If using the Probability representation then choose whether 
        to write in the reduced/compact form or the full probability table.
    """

    num_states = (num_out)**(num_in)
    
    if method == None:
        if num_in <= 3 and num_out == 2: 
            method = 'Pauli'
        else: 
            method = 'Probability'

    if method == 'Pauli':
        if not(num_in <= 3 or num_out == 2):
            raise ValueError(" The Pauli coefficient method is only supported for bit, gbit and g3bit states.")
        else:
            states =[]
            for i in range(num_states):
                states += [np.concatenate([np.zeros(num_in*(num_out-1)), [1]])]
            f = '{0:0' + str(num_in)+'b}'
            for i in range(len(states)):
                for k in range(np.shape(states)[1]-1):
                    states[i][k] = 2*int(f.format(i)[k])-1
        return np.array(states).astype(np.int64)

    elif method == 'Probability':
        states = []
        temp = np.concatenate([[1],np.zeros(num_out-1)])
        if reduce == True:
            perms = [p[:-1] for p in set(permutations(temp))]
            states = [np.concatenate([np.array(p).flatten(),[1]]) for p in product(perms, repeat=num_in)]
            return np.array(states).astype(np.int64)
        if reduce == False:
            perms = [p for p in set(permutations(temp))]
            states = [np.array(p).flatten() for p in product(perms, repeat=num_in)]
            return np.array(states).astype(np.int64)
            

def effects_ve_in(st_in, destination='./polytopes/effects_ve_in.txt'):
    """"Generates a text file in the provided destination to act as the input
    for vertex enumeration to find the polytope of effects.

    Parameters
    ----------
    st_in : numpy.array or list
        The set of vertices of the state space
    destination : str
        Path to the file where the output should be saved.
    """

    o = sys.stdout

    if not(st_in.dtype == 'int64'):
        st_in = np.array(st_in).astype(np.int64)
    list1 = []
    for i in range(len(st_in)):
        list1 += [np.concatenate([-1*st_in[i],[0]])]
        list1 += [np.concatenate([st_in[i],[-1]])]
    with open(destination, 'w') as f:
        sys.stdout = f
        print("Inequalities" + '\n')
        for i in range(len(list1)):
            print(*list1[i])
    sys.stdout = o



def import_verts(ve_out):
    """Import and interpret the panda vertex enumeration output.

    Parameters
    ---------
    ve_out : str
        Path to the panda vertex enumeration output file.
    """

    verts_raw = np.loadtxt(ve_out, skiprows=1)

    verts = []
    for i in range(len(verts_raw)):
        if verts_raw[i][-1] != 0:
            temp = verts_raw[i][0:-1]/verts_raw[i][-1]
            verts += [temp]
        else:
            raise ValueError("The input polyhedron is unbounded.")
    return verts




def transf_ve_in(st_in, eff_out, destination='./polytopes/transf_ve_in.txt'):
    """Generate a text file to act as the input to panda for the vertex enumeration
    to find the set of all transformations between two given systems.

    Parameters
    ----------
    st_in : np.ndarray or list
        The vertices of the state space of the input system
    eff_out: np.ndarray or list
        The vertices of the effect space of the output system.
    destination: str
        Path to the file where the output must be saved.
    """

    o = sys.stdout

    eff_out2 = [e for e in eff_out if not np.array_equal(e,np.zeros(np.shape(eff_out)[1]))]


    testers = []
    for i in range(len(st_in)):
        for j in range(len(eff_out2)):
            testers += [np.kron(eff_out2[j],st_in[i]).flatten()]

    list1 = []
    for i in range(len(testers)):
        list1 += [np.concatenate([-1*testers[i],[0]])]
        list1 += [np.concatenate([testers[i],[-1]])]

    list1_rational = []
    for i in range(len(list1)):
        list1_rational += [[Fraction(j) for j in list1[i]]]

    list1_int = []
    for i in range(len(list1_rational)):
        list1_int += [lcm(*[j.denominator for j in list1_rational[i]])*np.array(list1_rational[i])]

        
    with open(destination, 'w') as f:
        sys.stdout = f
        print("Inequalities" + '\n')
        for i in range(len(list1_rational)):
            print(*[round(j) for j in list1_int[i]])
    sys.stdout = o
    
                


def find_instruments(transf, dims):
    """From a set of normalization non-increasing transformations find the instruments.
    Currently only restricted to 2 elements per instrument.

    Parameters
    ----------
    transf : np.ndarray or list
        A set of normalization non-increasing transformations. 
        Ideally, should be the vertices of the set of all CP maps between two systems.
    dims : list
        A list/tuple of integers corresponding to the dimensions (i.e. the size of the state vectors) 
        of the input and output systems.
        The dimension of the system here is the dimension of the linear span of the state space. 
    """

    if np.shape(transf)[1] != dims[0]*dims[1]:
        raise ValueError("Dimensions are not correct or the transformations are not flattened.")

    ins = []

    for i in range(1,len(transf)):
        for j in range(i,(len(transf))):
            if np.array_equal((transf[i]+transf[j])[dims[0]*(dims[1]-1):], np.concatenate([np.zeros(dims[0]-1),[1]])):
                ins += [[transf[i],transf[j]]]
    return ins


def channels(transf, dims):
    """From a set of normalization non-increasing transformations find the channels.

    Parameters
    ----------
    transf : np.ndarray or list
        A set of normalization non-increasing transformations. 
        Ideally, should be the vertices of the set of all CP maps between two systems.
    dims : list
        A list/tuple of integers corresponding to the dimensions (i.e. the size of the state vectors) 
        of the input and output systems.
        The dimension of the system here is the dimension of the linear span of the state space. 
    """

    if np.shape(transf)[1] != dims[0]*dims[1]:
        raise ValueError("Dimensions are not correct or the transformations are not flattened.")

    chan = []
    for ins_el in transf:
        if np.array_equal(ins_el[dims[0]*(dims[1]-1):], np.concatenate([np.zeros(dims[0]-1),[1]])):
            chan += [ins_el]
    return chan

def not_channels(transf, dims):
    """From a set of normalization non-increasing transformations find the normalization non-preserving maps.

    Parameters
    ----------
    transf : np.ndarray or list
        A set of normalization non-increasing transformations. 
        Ideally, should be the vertices of the set of all CP maps between two systems.
    dims : list
        A list/tuple of integers corresponding to the dimensions (i.e. the size of the state vectors) 
        of the input and output systems.
        The dimension of the system here is the dimension of the linear span of the state space. 
    """
    if np.shape(transf)[1] != dims[0]*dims[1]:
        raise ValueError("Dimensions are not correct or the transformations are not flattened.")
    ins_els = []
    for el in transf:
        if not np.array_equal(el[dims[0]*(dims[1]-1):], np.concatenate([np.zeros(dims[0]-1),[1]])):
            ins_els += [el]
    return ins_els
    

def allowed_terms(dims):
    """Find the allowed terms of a process matrix for the given systems.

    Parameters
    ---------
    dims : list of ints
        List of dimensions of the input and output systems of each of the parties.
        In the order AIAOBIBOCICO..... 
    """

    num_parties = int(len(dims)/2)
    chan_allowed = [1]
    for i in range(num_parties):
        chan_allowed = np.kron(chan_allowed, np.concatenate([np.ones(dims[2*i-2]*(dims[2*i-1]-1)),np.concatenate([np.zeros(dims[2*i-2]-1),[1]])]))
    allowed = np.ones(len(chan_allowed)) - chan_allowed
    allowed[-1] = 1
    return allowed

    
def pick_only_allowed(vec, dims):
    """Given a vectorized process matrix returns a vector with only the allowed terms.
    
    Parameters
    ----------
    vec : np.ndarray
        The vectorized process matrix with all the terms.
    dims : list of ints
        List of dimensions of the input and output systems of each of the parties.
        In the order AIAOBIBOCICO.....  
    """

    allowed = allowed_terms(dims)
    return np.array(list(compress(vec, allowed.tolist())))




def process_ve_in(transf, dims, destination='./polytopes/process_ve_in.txt'):

    """Generate a text file to act as the input for PANDA vertex enumeration to find 
    the set of vertices of the polytope of all processes between the given systems.
    Currently only works for bipartite processes.
    
    Parameters
    ----------
    transf : list of np.ndarrays or np.ndarray
        A tuple/list of list of all extremal normalization non-preserving and non-increasing transformations
        for each party.
    dims : list
        A list/tuple of integers corresponding to the dimensions (i.e. the size of the state vectors) 
        of the input and output systems.
    destination :  str
        Path to the file where the output must be saved.
    """

    o = sys.stdout
    transf = np.array(transf)

    for i in range(len(transf)):
        if np.array_equal(transf[i], np.zeros([transf.shape[1], transf.shape[1]])):
            del transf[i] 


    transf_prod = transf
    temp =[]

    for j in range(len(transf_prod)):
        for k in range(1,len(transf)):
            temp += [np.kron(transf_prod[j], transf[k])]
    transf_prod = temp
    temp =[]

    transf_prod_red = [pick_only_allowed(i,dims) for i in transf_prod]

    list1 = []
    for i in range(len(transf_prod_red)):
        list1 += [np.concatenate([-1*transf_prod_red[i],[0]])]
        list1 += [np.concatenate([transf_prod_red[i],[-1]])]
    list1 += [np.concatenate([np.concatenate([np.zeros(np.shape(transf_prod_red)[1]-1),[1]]),[-1]])]
    list1 += [np.concatenate([-np.concatenate([np.zeros(np.shape(transf_prod_red)[1]-1),[1]]),[1]])]

    list1_rational = []
    for i in range(len(list1)):
        list1_rational += [[Fraction(j) for j in list1[i]]]

    list1_int = []
    for i in range(len(list1_rational)):
        list1_int += [lcm(*[j.denominator for j in list1_rational[i]])*np.array(list1_rational[i])]

        
    with open(destination, 'w') as f:
        sys.stdout = f
        print("Inequalities" + '\n')
        for i in range(len(list1_rational)):
            print(*[round(j) for j in list1_int[i]])
    sys.stdout = o




def generate_distribution(sys_info):
    """Generates all vertices of valid (num_parties, num_in, num_out) probability distributions.
    The output is in the 'Tsirelesen' representation.
    
    Parameters
    ----------
    sys_info : list of ints
        A list of integers indicating the number of parties, and  number of settings and outcomes 
        for each of the parties. 
    """

    num_parties = sys_info[0]
    num_ins = sys_info[1]
    num_out = sys_info[2]

    vec = np.concatenate([[1], np.zeros(num_out-1)])
    perms = [p for p in set(permutations(vec))]
    perms_prod = [p for p in product(perms, repeat = num_parties)]
    blocks_list = []
    block = [1]
    for p in perms_prod:
        for i in range(num_parties):
            block = np.kron(block, p[i])
        blocks_list += [block.reshape([q for q in product([num_out], repeat=num_parties)][0])]
        block = [1]
    num_blocks = num_ins**num_parties
    dist_unravel = np.array([p for p in product(blocks_list, repeat = num_blocks)])    
    dists = dist_unravel.reshape(np.concatenate([[dist_unravel.shape[0]],
                                                 [q for q in product([num_ins], repeat=num_parties)][0], 
                                                 [q for q in product([num_out], repeat=num_parties)][0]]))
    return np.transpose(dists, axes = np.concatenate([[0],
    np.concatenate([[i,num_parties+i] for i in range(1,num_parties+1)])])).reshape(np.concatenate([[dists.shape[0]], 
    list(np.array([q for q in product([num_ins], repeat=num_parties)][0])*np.array([q for q in product([num_out], repeat=num_parties)][0]))]))

