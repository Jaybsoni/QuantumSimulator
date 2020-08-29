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
        try:
            assert isclose((np.sqrt(c1**2 + c2**2)), 1.0, abs_tol=1e-5)
        except AssertionError as error:
            print(np.sqrt(c1**2 + c2**2))
            print(c1, c2)
            print(error)
        else:
            self.state = np.array([c1, c2])

        return

    def __repr__(self):
        return "{0}|0> + {1}|1>".format(self.state[0], self.state[1])

    def measure(self, shots=1):
        prob_1 = la.norm(self.state[1])
        result = np.random.binomial(1, prob_1, size=shots)
        return result


class Circuit:

    h_mat = 1 / np.sqrt(2) * np.array([[1,  1],
                                       [1, -1]])

    x_mat = np.array([[0, 1],
                      [1, 0]])

    y_mat = np.array([[0, complex(0, 1)],
                      [complex(0, 1), 0]])

    z_mat = np.array([[1, 0],
                      [0, -1]])

    gate_to_mat = {'H': h_mat,
                   'X': x_mat,
                   'Y': y_mat,
                   'Z': z_mat}

    def __init__(self, num_qbits):
        assert(num_qbits > 0)

        circuit_state = np.zeros(2**num_qbits, dtype=float)
        circuit_state[0] = 1.0

        self.circuit_state = circuit_state
        self.num_qbits = num_qbits
        self.gate_array = []

        return

    def __repr__(self):
        display = ''
        circuit_rows = []

        for i in range(self.num_qbits):
            circuit_rows.append(['q{}: |0>--'.format(i)])

        for meta_tuple in self.gate_array:
            if len(meta_tuple) == 3:
                control_index = meta_tuple[1][0]
                target_index = meta_tuple[1][1]

                circuit_rows[control_index].append('CQ-')
                circuit_rows[target_index].append('T{}-'.format(meta_tuple[2]))

            else:
                for index in meta_tuple[1]:
                    circuit_rows[index].append(meta_tuple[0] + '--')

            max_len = max([len(row) for row in circuit_rows])
            for row in circuit_rows:
                while len(row) < max_len:
                    row.append('---')

        for row in circuit_rows:
            display += "".join(row) + '--M\n'

        return display

    def add_sq_gate(self, qbit, gate_str):
        if not type(qbit) is list:
            qbit = [qbit]

        meta_tuple = (gate_str, qbit)
        self.gate_array.append(meta_tuple)
        return

    def add_dq_gate(self, control_ind, target_ind, gate_str):
        meta_tuple = ('CQ', [control_ind, target_ind], gate_str)
        self.gate_array.append(meta_tuple)
        return

    def apply_controlgate(self, control_ind, target_ind, unitary_mat):
        control = (self.num_qbits - 1) - control_ind
        target = (self.num_qbits - 1) - target_ind
        unique_qbit_combinations = binary_ittorator(control, target, self.num_qbits)

        for combination in unique_qbit_combinations:
            q0_comb = combination
            q1_comb = combination[:target] + '1' + combination[target + 1:]
            index_0 = int(q0_comb, 2)
            index_1 = int(q1_comb, 2)

            substate_vect = np.array([self.circuit_state[index_0], self.circuit_state[index_1]])
            substate_vect = unitary_mat @ substate_vect

            self.circuit_state[index_0] = substate_vect[0]
            self.circuit_state[index_1] = substate_vect[1]
        return

    def apply_gate(self, gate_matrix):
        self.circuit_state = gate_matrix @ self.circuit_state
        return

    def run(self, status=True):
        total_layers = len(self.gate_array)

        for layer, meta_tuple in enumerate(self.gate_array):
            if status:
                print('Processing layer {}/{} ...'.format(layer + 1, total_layers))

            gate_str = meta_tuple[0]

            if gate_str == 'CQ':
                control_indx = meta_tuple[1][0]
                target_indx = meta_tuple[1][1]
                unitary_mat = self.gate_to_mat[meta_tuple[2]]
                self.apply_controlgate(control_indx, target_indx, unitary_mat)

            else:
                gate = np.eye(2)  # the gate that will be applied to the state vector
                if self.num_qbits - 1 in meta_tuple[1]:
                    gate = self.gate_to_mat[gate_str]

                for index in np.arange(self.num_qbits - 2, -1, -1):
                    unitary_mat = self.gate_to_mat[gate_str]

                    qbit_gate = np.eye(2)
                    if index in meta_tuple[1]:
                        qbit_gate = unitary_mat

                    gate = np.kron(gate, qbit_gate)
                self.apply_gate(gate)

        if status:
            print('Done!')
        return

    def simulate(self, shots=100):
        probabilities = self.circuit_state * self.circuit_state
        results = np.random.multinomial(shots, probabilities, size=1)
        counts = {}

        for index, value in enumerate(results[0]):
            key = get_binary(index, self.num_qbits)
            counts[key] = value

        return counts

    @staticmethod
    def plot_counts(counts):
        plt.bar(range(len(counts)), counts.values())
        plt.xticks(range(len(counts)), list(counts.keys()))
        plt.show()
        return

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


def main():
    return


if __name__ == "__main__":
    main()
