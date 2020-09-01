# Main file for the simulator
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from math import isclose
import itertools


def binary_ittorator(control_bit, target_bit, n):
    """
    create all possible binary integers of bit length n with the known control_bit,
    target_bit positions held constant. Return list of all binary variations (as strs)

    :param control_bit: int
    :param target_bit: int
    :param n: int
    :return: list of str
    """
    try:
        assert(n > control_bit)
        assert(n > target_bit)
    except AssertionError as e:
        print(f"Error: n ({n}) must be larger than the index of each bit")
        print(e)
    else:
        # itertools creates all possible binary combinations for n bites
        binary_combinations = [list(i) for i in itertools.product([0, 1], repeat=n)]
        return_list = []  # Create an empty list to be populated

        # For all elements, replace the bit1 element to bit1 value, same for bit2
        for i in range(len(binary_combinations)):
            binary_combinations[i][control_bit] = 1
            binary_combinations[i][target_bit] = 0

            # Convert each list of ints (1s & 0s) to a list of strings
            strings = [str(i) for i in binary_combinations[i]]
            bin_string = "".join(strings)

            # Only add to the return list if it's unique
            if bin_string not in return_list:
                return_list.append(bin_string)

        return return_list


def get_binary(x, n):
    """
    determine the binary representation of an int x (base 10) given that
    x <= 2^n where n is an int representing the total bit length.

    :param x: base 10 int of interest
    :param n: base 10 int, bit length
    :return: str x as written in base 2
    """
    binary_result = ''
    target = x

    for i in np.arange(n-1, -1, -1):
        if 2**i > target:
            binary_result += '0'
        else:
            binary_result += '1'
            target -= 2**i

    return binary_result


class Qbit:

    def __init__(self, c1, c2):
        """
        construct a qbit but specifying two complex coefficients c1, c2 such that
        c1^2 + c2^2 = 1 (normalized state).

        :param c1: complex float for state |0>
        :param c2: complex float for state |1>
        """
        try:
            assert isclose((np.sqrt(c1**2 + c2**2)), 1.0, abs_tol=1e-5)
        except AssertionError as error: # must be normalized
            print(c1, c2)
            print(error)
        else:
            self.state = np.array([c1, c2])

        return

    def __repr__(self):  # plots the qbit state in dirac notation (|psi> = c0|0> + c1|1>)
        return "{0}|0> + {1}|1>".format(self.state[0], self.state[1])

    def measure(self, shots=1):
        """
        measure the desired state multiple times (shots = # of measurements).

        :param shots: int, # of independant measurements
        :return: numpy array of ints, results of each measurement on state space (0,1)
        """
        prob_1 = la.norm(self.state[1])
        result = np.random.binomial(1, prob_1, size=shots)
        return result


class Circuit:
    # Quantum Gates of Interest:
    h_mat = 1 / np.sqrt(2) * np.array([[1,  1],
                                       [1, -1]])

    x_mat = np.array([[0, 1],
                      [1, 0]])

    y_mat = np.array([[0, complex(0, -1)],
                      [complex(0, 1), 0]])

    z_mat = np.array([[1, 0],
                      [0, -1]])

    phase_mat = np.array([[1, 0],
                          [0, complex(0, 1)]])

    gate_to_mat = {'H': h_mat,
                   'X': x_mat,
                   'Y': y_mat,
                   'Z': z_mat,
                   'S': phase_mat}

    def __init__(self, num_qbits):
        assert(num_qbits > 0)  # can't have a circuit without qbits

        circuit_state = np.zeros(2**num_qbits, dtype=float)  # an array of coefficients for the measureable states,
        circuit_state[0] = 1.0                               # initialized to the 0 state for all qbits

        self.circuit_state = circuit_state  # the 2^n dim state vector for the circuit with n qbits
        self.num_qbits = num_qbits
        self.gate_array = []  # we will store the specifics of our circuit in the gate array

        return

    def __repr__(self):  # creates an ASCII display of the circuit for visual confirmation
        display = ''
        circuit_rows = []

        for i in range(self.num_qbits):
            circuit_rows.append(['q{}: |0>--'.format(i)])   # each qbit begins in the 0 state. q1: |0>--

        for meta_tuple in self.gate_array:   # we iterate through the gate_array and plot each element of the circuit
            if len(meta_tuple) == 3:
                control_index = meta_tuple[1][0]
                target_index = meta_tuple[1][1]

                circuit_rows[control_index].append('CQ-')                          # here we plot the control-unitary
                circuit_rows[target_index].append('T{}-'.format(meta_tuple[2]))    # and target gate

            else:
                for index in meta_tuple[1]:
                    circuit_rows[index].append(meta_tuple[0] + '--')    # here we plot the single qbit gates

            max_len = max([len(row) for row in circuit_rows])
            for row in circuit_rows:
                while len(row) < max_len:
                    row.append('---')         # this is to ensure the circuit rows are the same length for allignment

        for row in circuit_rows:
            display += "".join(row) + '--M\n'

        return display  # a str which displays the circuit to the user

    def add_sq_gate(self, qbit, gate_str):
        """
        abstract function to add a single qbit gate to the circuit.
        Takes the index/indicies of the qbits and the gate string for the gate
        we wish to apply to those qbits.

        :param qbit: int or array of ints, refers to qbit index (starts at 0) for applying gate
        :param gate_str: str, the string corresponding to the gate you wish to add
        :return: None
        """
        if not type(qbit) is list:
            qbit = [qbit]

        meta_tuple = (gate_str, qbit)         # a meta_tuple stores the meta data required for the circuit operation
        self.gate_array.append(meta_tuple)    # meta_tuple s are stored in the gate_array for the circuit
        return

    def add_dq_gate(self, control_ind, target_ind, gate_str):
        """
        abstract function to add control-unitary gates to the circuit.
        Takes an index for the control qbit, an index for the target qbit and a string for the gate.
        Applies gate to target qbit if control qbit is in the state |1>.

        :param control_ind: int, control qbit index
        :param target_ind: int, target qbit index
        :param gate_str: str, the string corresponding to the gate you wish to apply
        :return: None
        """
        meta_tuple = ('CQ', [control_ind, target_ind], gate_str)  # meta_tuple stores data for circuit operation
        self.gate_array.append(meta_tuple)                        # meta_tuple s stored in gate_array
        return

    def apply_controlgate(self, control_ind, target_ind, unitary_mat):
        """
        updates the coefficients of the circuit_state based on the control-unitary gate applied.

        :param control_ind: int, control qbit index
        :param target_ind: int, target qbit index
        :param unitary_mat: numpy array of ints (shape = 2x2), normalized unitary matrix
        :return: None
        """
        control = (self.num_qbits - 1) - control_ind  # due to common notation, the elements of the state vector
        target = (self.num_qbits - 1) - target_ind  # are represented as |q_n> x |q_n-1> x ... x |q_1> x |q_0>
        unique_qbit_combinations = binary_ittorator(control, target, self.num_qbits)  # thus we adjust the indices ^^

        for combination in unique_qbit_combinations:  # iterate over unique combinations of the other qbits
            q0_comb = combination
            q1_comb = combination[:target] + '1' + combination[target + 1:]
            index_0 = int(q0_comb, 2)
            index_1 = int(q1_comb, 2)

            substate_vect = np.array([self.circuit_state[index_0], self.circuit_state[index_1]])
            substate_vect = unitary_mat @ substate_vect

            self.circuit_state[index_0] = substate_vect[0]  # update components of the circuit state vector as required
            self.circuit_state[index_1] = substate_vect[1]
        return

    def apply_gate(self, gate_matrix):
        """
        applies the gate_matrix to circuit_state to transform it. Since single qbit gates
        are tensor decomposable we can combine them outside of the tensor product and apply
        them directly to the state vector.

        :param gate_matrix: numpy array of ints (shape = nxn), normalized unitary matrix
        :return: None
        """
        self.circuit_state = gate_matrix @ self.circuit_state
        return

    def run(self, status=True):
        """
        once the circuit has been defined, circuit.run() will update the circuit_state vector to reflect the
        results of the circuit.

        :param status: bool (default is True), prints messages to give status update while processing circuit
        :return: None
        """
        total_layers = len(self.gate_array)

        for layer, meta_tuple in enumerate(self.gate_array):
            if status:
                print('Processing layer {}/{} ...'.format(layer + 1, total_layers))

            gate_str = meta_tuple[0]

            if gate_str == 'CQ':                      # a layer will never have both a control-gate and an independant
                control_indx = meta_tuple[1][0]       # single qbit gate
                target_indx = meta_tuple[1][1]
                unitary_mat = self.gate_to_mat[meta_tuple[2]]   # for more info on the control gate read the pdf
                self.apply_controlgate(control_indx, target_indx, unitary_mat)  # implementingCUgates.pdf

            else:
                gate = np.eye(2)  # the gate that will be applied to the state vector is initialized as identity
                if self.num_qbits - 1 in meta_tuple[1]:  # if a single qbit gate is applied to the 'first' (nth) qbit
                    gate = self.gate_to_mat[gate_str]    # then we initialize gate to that particular gate

                for index in np.arange(self.num_qbits - 2, -1, -1):   # iterate backwards through the rest of the qbits
                    unitary_mat = self.gate_to_mat[gate_str]

                    qbit_gate = np.eye(2)
                    if index in meta_tuple[1]:        # since single qbit gates are tensor decomposable we can
                        qbit_gate = unitary_mat

                    gate = np.kron(gate, qbit_gate)   # combine all of the gates in a layer with the tensor/kroneker
                self.apply_gate(gate)                 # product and apply it directly to the circuit state vector

        if status:
            print('Done!')
        self.gate_array = []
        return

    def simulate(self, shots=100):
        """
        once a circuit has been run, the circuit_state vector will be determined, we can then simulate repeated
        measurements on this prepared state.

        :param shots: int (default = 100), number of independant measurements of the circuit
        :return: dict, a dictionary with key is a str representing measureable state, value is int of observations
        """
        probabilities = self.circuit_state * self.circuit_state        # the coefficients are probabiltiy amplitudes
        results = np.random.multinomial(shots, probabilities, size=1)
        counts = {}

        for index, value in enumerate(results[0]):
            key = get_binary(index, self.num_qbits)    # determines the str representation of the measureable
            counts[key] = value

        return counts

    @staticmethod
    def plot_counts(counts):
        """
        quick method to plot histogram of results from dictionary returned by circuit.simulate().

        :param counts: dict, a dictionary with key is str of measureables, value is int of observations
        :return: None
        """
        plt.bar(range(len(counts)), counts.values())
        plt.xticks(range(len(counts)), list(counts.keys()))
        plt.show()
        return

    # Quick Methods for building circuits:
    def h(self, qbit):
        self.add_sq_gate(qbit, 'H')
        return

    def ch(self, control_qbit, target_qbit):
        self.add_dq_gate(control_qbit, target_qbit, 'H')
        return

    def x(self, qbit):
        self.add_sq_gate(qbit, 'X')
        return

    def cx(self, control_qbit, target_qbit):
        self.add_dq_gate(control_qbit, target_qbit, 'X')
        return

    def y(self, qbit):
        self.add_sq_gate(qbit, 'Y')
        return

    def cy(self, control_qbit, target_qbit):
        self.add_dq_gate(control_qbit, target_qbit, 'Y')
        return

    def z(self, qbit):
        self.add_sq_gate(qbit, 'Z')
        return

    def cz(self, control_qbit, target_qbit):
        self.add_dq_gate(control_qbit, target_qbit, 'Z')
        return

    def s(self, qbit):
        self.add_sq_gate(qbit, 'S')
        return

    def cs(self, control_qbit, target_qbit):
        self.add_dq_gate(control_qbit, target_qbit, 'S')
        return


def main():
    return


if __name__ == "__main__":
    main()
