from functools import reduce
import pickle
import numpy as np
from scipy.linalg import sqrtm, expm
np.set_printoptions(precision=2, suppress=True)

pickle_in = open('ibm_perth_gates.pickle', 'rb')
exact_cx_top, noisy_cx_top, exact_cx_bottom, noisy_cx_bottom, exact_x, noisy_x, exact_sx, noisy_sx = pickle.load(pickle_in)
pickle_in.close()

def exact_rz(theta): 
    rz = np.array([[np.exp(-1.j*theta/2), 0], [0, np.exp(1.j*theta/2)]])
    return np.kron(rz,rz.conj().T)

def noisy_rz(theta): 
    ''' rz gates have no noise since IBM applies them by simply changing the basis in software '''
    return exact_rz(theta)

def gate_action(gate, state_tensor): 
    ''' 
    gate is a tuple: (superoperator, qubit_list)
    for single qubit gates: qubit_list = [qubit_index,]
    for two qubit gates: qubit_list = [qubit_index, qubit_index+1]
    if you want to apply cx(1,0) you'll have to use noisy_cx_bottom as superoperator, qubit_list would still be [0,1]
    state_tensor is an n qubit density matrix reshaped as a rank 2n tensor: (2,2,2,2,...) 
    '''
    num_qubits = len(state_tensor.shape)//2    
    if len(gate[1]) == 1: 
        # single qubit gate 
        superop, q = gate[0].reshape([2]*4), gate[1][0]
        order = [i for i in range(2*num_qubits-2)]
        order.insert(q, 2*num_qubits-2)
        order.insert(q+num_qubits, 2*num_qubits-1)
        return np.tensordot(state_tensor, superop, axes=([q+num_qubits,q],[1,2])).transpose(*order) 
        
    else:
        # two qubit gate
        superop, q = gate[0].reshape([2]*8), gate[1][0]
        order = [i for i in range(2*num_qubits-4)]
        order.insert(q, 2*num_qubits-4)
        order.insert(q+1, 2*num_qubits-3)
        order.insert(q+num_qubits, 2*num_qubits-2)
        order.insert(q+num_qubits+1, 2*num_qubits-1)
        return np.tensordot(state_tensor, superop, axes=([q,q+1,q+num_qubits,q+num_qubits+1],[4,5,2,3])).transpose(*order)
    
def circuit_action(circ, init_state): 
    ''' circ is a list of gates, each gate is a tuple as described above, init_state can be a vector or a matrix '''
    num_qubits = int(np.log2(init_state.shape[0]))
    if len(init_state.shape) == 1: 
        init_state = np.outer(init_state, init_state.conj().T) # turns state into density matrix if provided as a vector
    state_tensor = init_state.reshape([2]*num_qubits*2)
    for gate in circ: 
        state_tensor = gate_action(gate, state_tensor)
    return state_tensor.reshape(2**num_qubits, 2**num_qubits)

def trace_distance(dm1, dm2): 
    return (1/2)*np.trace(sqrtm((dm1-dm2).conj().T @ (dm1-dm2)))

def fidelity(dm1, dm2): 
    return np.trace(sqrtm(sqrtm(dm1) @ dm2 @ sqrtm(dm2)))