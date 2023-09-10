from simulator import *
from tqdm import tqdm

def givens_gate(theta): 
    ''' exact givens gate unitary '''
    return np.array([[1,0,0,0],
                     [0, np.cos(theta/2), -np.sin(theta/2), 0], 
                     [0, np.sin(theta/2), np.cos(theta/2), 0], 
                     [0,0,0,1]])

def givens_gate_noisy(theta, q0, q1, dagger=False): 
    ''' compiles two qubit Givens rotation as a circuit of noisy IBM superoperators '''
    gate_list = [
        (noisy_rz(np.pi/2), [q0,]), 
        (noisy_sx, [q0,]),
        (noisy_rz(np.pi/2), [q0,]), 
        (noisy_cx_top, [q0,q1]),
        (noisy_sx, [q0,]), 
        (noisy_sx, [q1,]), 
        (noisy_rz(np.pi-(theta/2)), [q0,]),
        (noisy_rz(np.pi-(theta/2)), [q1,]),
        (noisy_sx, [q0,]), 
        (noisy_sx, [q1,]), 
        (noisy_rz(-np.pi), [q0,]),
        (noisy_rz(-np.pi), [q1,]),
        (noisy_cx_top, [q0,q1]),
        (noisy_rz(np.pi/2), [q0,]),
        (noisy_sx, [q0,]),
        (noisy_rz(np.pi/2), [q0,])
    ]
    if not dagger: 
        return gate_list
    else: 
        gate_list.reverse()
        return gate_list
    
def vff_circuit(angles, params): 
    ''' construct noisy circuit for WDW^d where W are built from Givens rotation with params and D is built from Rz gates with angles '''
    num_qubits = params.shape[1] + 1
    circ = []
    
    # lay down layers of W^d
    for layer_params in np.flip(params,0): 
        # lay down the short stack of the layer
        for i in range(num_qubits//2, num_qubits-1): 
            q0 = 2*(i - num_qubits//2) + 1
            circ = circ + givens_gate_noisy(layer_params[i], q0, q0+1, dagger=True)
        # lay down the long stack of the layer
        for i in range(num_qubits//2): 
            q0 = 2*i
            circ = circ + givens_gate_noisy(layer_params[i], q0, q0+1, dagger=True)
    
    # lay down D
    for i in range(num_qubits): 
        circ.append((noisy_rz(angles[i]), [i,]))
    
    for layer_params in params: 
        # lay down the long stack of the layer
        for i in range(num_qubits//2): 
            q0 = 2*i
            circ = circ + givens_gate_noisy(layer_params[i], q0, q0+1, dagger=False)
        # lay down the short stack of the layer
        for i in range(num_qubits//2, num_qubits-1): 
            q0 = 2*(i - num_qubits//2) + 1
            circ = circ + givens_gate_noisy(layer_params[i], q0, q0+1, dagger=False)
            
    return circ 

def givens_gate_exact(theta, q0, q1, dagger=False): 
    ''' compiles two qubit Givens rotation as a circuit of exact IBM superoperators, useful for testing '''
    gate_list = [
        (exact_rz(np.pi/2), [q0,]), 
        (exact_sx, [q0,]),
        (exact_rz(np.pi/2), [q0,]), 
        (exact_cx_top, [q0,q1]),
        (exact_sx, [q0,]), 
        (exact_sx, [q1,]), 
        (exact_rz(np.pi-(theta/2)), [q0,]),
        (exact_rz(np.pi-(theta/2)), [q1,]),
        (exact_sx, [q0,]), 
        (exact_sx, [q1,]), 
        (exact_rz(-np.pi), [q0,]),
        (exact_rz(-np.pi), [q1,]),
        (exact_cx_top, [q0,q1]),
        (exact_rz(np.pi/2), [q0,]),
        (exact_sx, [q0,]),
        (exact_rz(np.pi/2), [q0,])
    ]
    if not dagger: 
        return gate_list
    else: 
        gate_list.reverse()
        return gate_list
    
def vff_circuit_exact(angles, params): 
    ''' construct noiseless circuit for WDW^d, useful for testing '''
    num_qubits = params.shape[1] + 1
    circ = []
    
    # lay down layers of W^d
    for layer_params in np.flip(params, 0): 
        # lay down the short stack of the layer
        for i in range(num_qubits//2, num_qubits-1): 
            q0 = 2*(i - num_qubits//2) + 1
            circ = circ + givens_gate_exact(layer_params[i], q0, q0+1, dagger=True)
        # lay down the long stack of the layer
        for i in range(num_qubits//2): 
            q0 = 2*i
            circ = circ + givens_gate_exact(layer_params[i], q0, q0+1, dagger=True)
        
    for i in range(num_qubits): 
        circ.append((exact_rz(angles[i]), [i,]))
    
    for layer_params in params: 
        # lay down the long stack of the layer
        for i in range(num_qubits//2): 
            q0 = 2*i
            circ = circ + givens_gate_exact(layer_params[i], q0, q0+1, dagger=False)
        # lay down the short stack of the layer
        for i in range(num_qubits//2, num_qubits-1): 
            q0 = 2*(i - num_qubits//2) + 1
            circ = circ + givens_gate_exact(layer_params[i], q0, q0+1, dagger=False)
            
    return circ 

def vff_circuit_draw(angles, params):
    ''' returns list of gates representing the vff circuit, useful for testing ''' 
    num_qubits = params.shape[1] + 1
    circ = []
    
    # lay down layers of W^d
    for layer_params in np.flip(params, 0): 
        # lay down the short stack of the layer
        for i in range(num_qubits//2, num_qubits-1): 
            q0 = 2*(i - num_qubits//2) + 1
            circ.append(('givens gate with angle '+str(-layer_params[i]), [q0,q0+1])) 
        # lay down the long stack of the layer
        for i in range(num_qubits//2): 
            q0 = 2*i
            circ.append(('givens gate with angle '+str(-layer_params[i]), [q0,q0+1])) 
        
    for i in range(num_qubits): 
        circ.append(('rz gate with angle '+str(angles[i]), [i,]))
    
    for layer_params in params: 
        # lay down the long stack of the layer
        for i in range(num_qubits//2): 
            q0 = 2*i
            circ.append(('givens gate with angle '+str(layer_params[i]), [q0,q0+1]))
        # lay down the short stack of the layer
        for i in range(num_qubits//2, num_qubits-1): 
            q0 = 2*(i - num_qubits//2) + 1
            circ.append(('givens gate with angle '+str(layer_params[i]), [q0,q0+1])) 
         
    return circ 