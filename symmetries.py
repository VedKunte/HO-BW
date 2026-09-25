import numpy as np
from polytopes import *


def permute_rows(mat,order):
    """Permute the rows of a matrix according to a given rule.
    
    Parameters
    ----------
    mat : np.matrix or np.ndarray
        The matrix whose rows are to be permuted.
    order : list of ints
        The order in which the rows should be rearranged.
    """

    if not len(order) == np.shape(mat)[1]:
        arr = np.arange(np.shape(mat)[1])
        for i in range(len(order)):
            arr[np.sort(order)[i]] = order[i].copy()
        order = arr.copy()

    return np.array([mat[i] for i in order])



def relabellings(io, method = None, reduce = True):
    """Find all relabellings (reversible channels) of the states of a system. 

    Parameters
    ----------
    io : list of ints
        A pair of integers that indicate the number of fiducial measuremenets and outcomes of a system.
    method : str. Can be 'Pauli' or 'Probability'
        Dictates which representation is used for the states.
    reduce : bool
        If using the Probability representation then choose whether 
        to write in the reduced/compact form or the full probability table.
    """

    num_states = io[1]**io[0]
        
    if method == None:
        if io[0] <= 3 and io[1] == 2: 
            method = 'Pauli'
        else: 
            method = 'Probability'

    if method == 'Pauli':

        rel_out = []
        f = '{0:0' + str(io[0])+'b}'
        for i in range(io[1]**io[0]):
            rel_out += [np.diag(np.concatenate([[2*int(k)-1 for k in f.format(i)], [1]]))]
        perms = np.array([p for p in permutations(np.arange(io[0]))])
        rel_in =[]
        for p in perms:
            rel_in += [permute_rows(m, p) for m in rel_out]
            
        return rel_in

    if method == 'Probability':

        id0 = np.eye(io[0])
        id1 = np.eye(io[1])

        perm0 = np.array([p for p in permutations(np.arange(io[0]))])
        perm1 = np.array([p for p in permutations(np.arange(io[1]))])
        perms_list = np.array([p for p in product(perm1, repeat = io[0])])

        all_rel = []
        rel = []
        for p in perm0:
            t = permute_rows(id0, p)
            for q in perms_list:
                rel += [[np.kron(t[i], permute_rows(id1,q[i])) for i in range(io[0])]]
                rel = np.array(rel).reshape(-1,np.array(rel).shape[-1])
                all_rel += [np.matrix(rel)]
                rel = []

        if reduce == False:
            return all_rel
        else:
            block1 = np.vstack([np.eye(io[1]-1), -np.ones(io[1]-1)])
            proj1 = []
            for i in range(io[0]):
                proj1 += [np.hstack([np.kron(np.eye(io[0])[i], block1), np.concatenate([np.zeros(io[1]-1),[1]]).reshape(io[1],1)])]
            proj1 = np.array(proj1).reshape(-1,np.array(proj1).shape[-1])

            block2 = np.hstack([np.eye(io[1]-1), np.zeros([io[1]-1,1])])
            proj2 = []
            for i in range(io[0]):
                proj2 += [np.kron(np.eye(io[0])[i], block2)]
            proj2 = np.array(proj2).reshape(-1, np.array(proj2).shape[-1])
            proj2 = np.vstack([proj2, np.concatenate([np.zeros(io[0]*io[1]-1),[1]])])

            return [proj2@m@proj1 for m in all_rel]


        

def remove_duplicates(list1):
    """Remove duplicate elements in a list. 
    Normally in python one would use set() for this, but it doesn't support lists where
    the elements are np.ndarrays so I need to use a custom implementation.
    
    Parameters
    ----------
    list1 : list of np.ndarrays or nd.array
        The list with (potential) duplicate elements.
    """

    list1 = np.array(list1)
    indices = []

    i=0
    while i < len(list1):
        for j in range(i+1, len(list1)):
            if np.array_equal(list1[i], list1[j]):
                indices += [j]
        list1 = np.delete(list1, indices, axis = 0)
        indices = []
        i += 1
                
    return list1




def eq_classes_channels(chan, sys_info):
    """Find the equivalence classes of processes that are invariant under relabellings of the 
    states of the input and ouput systems of the parties.
    This version doesn't take into account party relabellings. 

    Parameters
    ----------
    chan : list of np.ndarrays
        The list of channels. Ideally, this is the set of all vertices 
        of the channels between two systems.
    sys_info : list of lists of ints
        The list of number of fiducial measurements and outcomes that identify
        the input and output systems.
    """

    dims = [i[0]**(i[1]-1)+1 for i in sys_info]
    num_sys = len(sys_info)

    rel_sys = [relabellings(s) for s in sys_info]

    rel_prod = rel_sys[0]
    temp = []
    for i in range(1,num_sys):
        for j in range(len(rel_prod)):
            for k in range(len(rel_sys[i])):
                temp += [np.kron(rel_prod[j], rel_sys[i][k])]
        rel_prod = temp
        temp =[]    

    class_sets = []
    eq_class = []
    indices = []
    remaining_chans = np.array(chan.copy())
    while len(remaining_chans) > 0:
        eq_class = remove_duplicates([rel@remaining_chans[0] for rel in rel_prod])
        for eq in eq_class:
            for j in range(len(remaining_chans)):
                if np.array_equal(eq, remaining_chans[j]):
                    indices += [j]
            remaining_chans = np.delete(remaining_chans, indices, axis =0)
            indices = []
        class_sets += [eq_class]
        eq_class = []

    return class_sets




def eq_classes_proc(proc_set, sys_info):
    """Find the equivalence classes of processes that are invariant under relabellings of the 
    states of the input and ouput systems of the parties.
    This version doesn't take into account party relabellings. 

    Parameters
    ----------
    proc_set : list of np.ndarrays
        The list of processes. Ideally, this is the set of all vertices 
        of the process polytope between the parties.
    sys_info : list of lists of ints
        The list of number of fiducial measurements and outcomes that identify
        the input and output systems of the parties.


    Note: This function doesn't seem to be very optimized. It is not reccommended to use it for large sets
    in its current state.
    """

    dims = [i[0]**(i[1]-1)+1 for i in sys_info]
    num_sys = len(sys_info)

    rel_sys = [relabellings(s) for s in sys_info]

    rel_prod = rel_sys[0]
    temp = []
    for i in range(1,num_sys):
        for j in range(len(rel_prod)):
            for k in range(len(rel_sys[i])):
                temp += [np.kron(rel_prod[j], rel_sys[i][k])]
        rel_prod = temp
        temp =[]    
    rel_prod_red = [pick_only_allowed(pick_only_allowed(m, dims).transpose(), dims) for m in rel_prod]

    class_sets = []
    eq_class = []
    indices = []
    remaining_procs = np.array(proc_set.copy())
    while len(remaining_procs) > 0:
        eq_class = remove_duplicates([rel@remaining_procs[0] for rel in rel_prod_red])
        for eq in eq_class:
            for j in range(len(remaining_procs)):
                if np.array_equal(eq, remaining_procs[j]):
                    indices += [j]
            remaining_procs = np.delete(remaining_procs, indices, axis =0)
            indices = []
        class_sets += [eq_class]
        eq_class = []

    return class_sets

        


