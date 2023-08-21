from simulator import *
from tqdm import tqdm

pauli = np.array([np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]), np.array([[0,-1.j],[1.j,0]]), np.array([[1,0],[0,-1]])])
pauli_tensor = np.array([[np.kron(pauli[i], pauli[j]) for i in range(4)] for j in range(4)])

def kronecker_pad(matrix, num_qubits, starting_site): 
    ''' takes a local gate described as a matrix and pads it with identity matrices to create a global operator '''
    kron_list = [np.eye(2) for i in range(num_qubits)]    
    kron_list[starting_site] = matrix
    if matrix.shape[0] == 4: 
        del kron_list[starting_site+1]
    
    padded_matrix = kron_list[0]
    for i in range(1, len(kron_list)):
        padded_matrix = np.kron(kron_list[i], padded_matrix)    
    return padded_matrix

def xy_ham(num_qubits): 
    ''' generates the XY Hamiltonian as a 2^n by 2^n matrix '''
    terms = []        
    for i in range(num_qubits-1): 
        y_hop = kronecker_pad(pauli_tensor[2,2], num_qubits, i)
        terms.append(y_hop)
        x_hop = kronecker_pad(pauli_tensor[1,1], num_qubits, i)
        terms.append(x_hop)
    return sum(terms) 

def xy_gate_noisy(t, q0, q1):
    ''' compiles the 2 qubit XY interaction as a circuit of noisy IBM gates '''
    return [
        (noisy_rz(np.pi/2), [q0,]), 
        (noisy_rz(-np.pi/2), [q1,]),
        (noisy_sx, [q1,]),
        (noisy_rz(np.pi/2), [q1,]), 
        (noisy_cx_bottom, [q0,q1]),
        (noisy_sx, [q0,]),
        (noisy_sx, [q1,]), 
        (noisy_rz(np.pi-2*t), [q0,]),
        (noisy_rz(np.pi-2*t), [q1,]), 
        (noisy_sx, [q0,]),
        (noisy_sx, [q1,]), 
        (noisy_rz(-np.pi), [q0,]), 
        (noisy_rz(-np.pi), [q1,]),
        (noisy_cx_bottom, [q0,q1]),
        (noisy_rz(-np.pi/2), [q0,]), 
        (noisy_rz(np.pi/2), [q1,]),
        (noisy_sx, [q1,]),
        (noisy_rz(-np.pi/2), [q1,]),
    ]

def xy_trotter(num_qubits, total_time, num_steps): 
    ''' generates the first order Trotter circuit for the XY Hamiltonian '''
    step_size = total_time/num_steps
    circ = []
    for i in range(num_steps):
        for q in range(0,num_qubits,2): 
            circ += xy_gate_noisy(step_size, q, q+1)
        for q in range(1,num_qubits-1,2): 
            circ += xy_gate_noisy(step_size, q, q+1)
    return circ 

def optimal_step(num_qubits, init_state, total_time, num_step_opts, output_file): 
    ''' 
    runs noisy Trotter evolution for total_time with each choice of num_step in num_step_opts
    init_state can be vector or density matrix 
    records list of fidelity with exact evolution for each choice of num_step
    ''' 
    ham = xy_ham(num_qubits)
    exact_state = expm(-1.j*total_time*ham) @ init_state
    fidelity_list = []
    for num_step in tqdm(num_step_opts):
        circ = xy_trotter(num_qubits, total_time, num_step)
        noisy_state = circuit_action(circ, init_state)
        fidelity_list.append((exact_state.conj() @ noisy_state @ exact_state).real)
    
    with open(output_file, 'wb') as f:
        pickle.dump(fidelity_list, f)