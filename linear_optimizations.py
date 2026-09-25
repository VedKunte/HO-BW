import numpy as np
import cvxpy as cp
from polytopes import *



def linear_opt_proc(ins, dims, transf, corr = np.array([1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1])/4):
    """"The linear optimization to find the maximum value of the causal game 
    given by corr for a given pair of instruments. 
    The current version of the linear optimization only works for the case where
    all parties have only 2 instruments with 2 elements each.
    
    Parameters
    ----------
    ins : list of np.ndarray or np.ndarray
        A list with the choice of instruments for each of the parties.
    dims : list of ints
        List of dimensions of the input and output systems of each of the parties.
        In the order AIAOBIBOCICO..... 
    transf : list of np.ndarrays
        A tuple/list of list of the extremal (vertices of the set of) normalization non-increasing and 
        non-preserving transformations for each of the parties.
    corr : np.ndarray
        The vectorized causal game that we need to maximize.
    """


    num_parties = int(len(dims)/2)

    W = cp.Variable(int(sum(allowed_terms(dims))))


    transf_prod = transf[0]
    temp =[]
    for i in range(1,num_parties):
        for j in range(len(transf_prod)):
            for k in range(len(transf[i])):
                temp += [np.kron(transf_prod[j], transf[i][k])]
        transf_prod = temp
        temp =[]

    transf_prod_red = [pick_only_allowed(i,dims) for i in transf_prod]

    constraints = [W[-1] == 1]
    constraints += [W@i >= 0 for i in transf_prod_red]
    #constraints += [W@i <= 1 for i in transf_prod_red]

    ins_set = []
    ins_set2 = []
    for ins_A in ins[0]:
        ins_set += [ins_el for ins_el in ins_A]
    for j in range(1, num_parties):
        for i in range(len(ins_set)):
            for ins_A2 in ins[j]:
                for ins_el2 in ins_A2:
                    ins_set2 += [np.kron(ins_set[i],ins_el2)]
        ins_set = ins_set2
        ins_set2 = []

    ins_prod = [pick_only_allowed(i, dims) for i in ins_set]

    obj = cp.Maximize(corr@[W@i for i in ins_prod])

    prob1 = cp.Problem(obj, constraints)

    prob1.solve()

    return [prob1.value, W.value]