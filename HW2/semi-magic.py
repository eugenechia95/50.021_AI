import pdb
from csp import *
from random import shuffle

def solve_semi_magic(algorithm=backtracking_search, **args):
    """ From CSP class in csp.py
        vars        A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b
                    """
    # Use the variable names in the figure
    csp_vars = ['V%d'%d for d in range(1,10)]

    #########################################
    # Fill in these definitions

    csp_domains = UniversalDict([1,2,3])
    # for i in csp_vars:
    #     csp_domains[i] = [1,2,3]
        
    csp_neighbors = {}
    csp_neighbors["V1"]=["V2","V3"]
    csp_neighbors['V2']=['V8','V5']
    csp_neighbors['V3']=['V5','V7']
    csp_neighbors['V4']=['V6','V5']
    csp_neighbors['V5']=['V1','V9']
    csp_neighbors['V6']=['V3','V9']
    csp_neighbors['V7']=['V9','V8']
    csp_neighbors['V8']=['V2','V5']
    csp_neighbors['V9']=['V3','V6']


    def csp_constraints(A, a, B, b):
        "A constraint saying two neighboring variables must differ in value."
        a != b
        # pass

    #########################################
    
    # define the CSP instance
    csp = CSP(csp_vars, csp_domains, csp_neighbors,
              csp_constraints)

    # run the specified algorithm to get an answer (or None)
    ans = algorithm(csp, **args)
    
    print 'number of assignments', csp.nassigns
    assign = csp.infer_assignment()
    if assign:
        for x in sorted(assign.items()):
            print x
    return csp

print("Normal backtracking search:")
solve_semi_magic()

print("WITH VARIABLE search")
print("Backtracking serach with minimum remaining values heuristic:")
solve_semi_magic(select_unassigned_variable=mrv)

print("WITH VALUE ORDERING")
print("Backtracking search with least constraining values:")
solve_semi_magic(order_domain_values=lcv)

print("WITH INFERENCE")
print("Backtrackingsearch with forward checking:")
solve_semi_magic(inference=forward_checking)

print("Backtrackingsearch with arc consistency:")
solve_semi_magic(inference=mac)

print("WITH MULTIPLE HEURISTICS")
print("Backtracking search with minimum remaining values heuristic, least constraining values and forward checking:")
solve_semi_magic(select_unassigned_variable=mrv, order_domain_values=lcv, inference=forward_checking)

print("Backtracking search with minimum remaining values heuristic, least constraining values and arc consistency:")
solve_semi_magic(select_unassigned_variable=mrv, order_domain_values=lcv, inference=mac)
