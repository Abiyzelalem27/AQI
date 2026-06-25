

from qutip import mesolve, sesolve, Qobj
from Adva_Quant_Inform.hamiltonians import H0, H1
from Adva_Quant_Inform.operators import psi0 
# define the function for the time-evolution of the state
def psi_te(N, tlist): # 'te' stays for time-evolved (with Schrödinger's equation)
    args = {'t_max': max(tlist)}
    
    # construct the time-dependent Hamiltonian
    # in list-function format (which can be used as input for mesolve())
    h_t_flag = [[H0(N), lambda t, args : (args['t_max']-t)/args['t_max']],
       [H1(N), lambda t, args : t/args['t_max']]]
    
    # evolve the system, request the solver to call process_rho at each time step.
    result_flag = mesolve(h_t_flag, psi0(N), tlist, [], [], args)
    
    # resolve time-dependent Schrödinger's equation
    psi_te_flag = result_flag.states 
    return(psi_te_flag) 