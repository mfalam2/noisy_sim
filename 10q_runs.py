from noisy_trotter import *

num_qubits = 10
init_state = np.zeros(2**num_qubits)
init_state[int('10'*(num_qubits//2),2)] = 1.0 # Neel state
runs = [1,2,3,4,5]

for total_time in runs:
    num_step_opts = np.arange(1,10)
    output_file = '10q_' + str(total_time) + '.pickle'
    optimal_step(num_qubits, init_state, total_time, num_step_opts, output_file)