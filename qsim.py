# Main file for the simulator
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from math import isclose


def split_state(vect):
    assert (len(vect) == 4)
    qbit1 = Qbit(vect[0] + vect[1], vect[2] + vect[3])
    qbit2 = Qbit(vect[0] + vect[2], vect[1] + vect[3])
    return qbit1.state, qbit2.state


def lst_to_str(lst):
    result = ''
    for i in lst:
        result += str(int(i))  # sometimes i is a float
    return result


class Qbit:

    def __init__(self, c1, c2):
        try:
            assert isclose((np.sqrt(c1**2 + c2**2)), 1.0, abs_tol=1e-5)
        except AssertionError:
            print(np.sqrt(c1**2 + c2**2))
            print(c1, c2)

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

    def __init__(self, num_qbits, num_bits):
        assert(num_qbits > 0)
        self.lst_qbits = []
        self.num_qbits = num_qbits
        self.num_bits = num_bits
        self.gate_array = []

        for i in range(num_qbits):
            self.lst_qbits.append(Qbit(1, 0))
            self.gate_array.append([])
        return

    def __repr__(self):
        disp_circuit = ''
        for i in range(self.num_qbits):
            row = 'q{}: |0>'.format(i)
            for meta_tuple in self.gate_array[i]:
                gate_str = meta_tuple[0]
                row += '--{0}'.format(gate_str)
            row += '--M\n'
            disp_circuit += row
        return disp_circuit

    def add_sq_gate(self, qbit_ind, gate_str):
        if type(qbit_ind) is list:
            pass
        else:
            qbit_ind = [qbit_ind]

        for index, gate_lst in enumerate(self.gate_array):
            if index in qbit_ind:
                meta_tuple = (gate_str, index)
            else:
                meta_tuple = ('-', index)
            gate_lst.append(meta_tuple)
        return

    def add_dq_gate(self, control_ind, target_ind, gate_str):
        for index, gate_lst in enumerate(self.gate_array):
            if index == control_ind:
                meta_tuple = ('CQ', control_ind, target_ind, gate_str)
            elif index == target_ind:
                meta_tuple = ('T' + gate_str, control_ind, target_ind, gate_str)
            else:
                meta_tuple = ('-', index)
            gate_lst.append(meta_tuple)
        return

    @staticmethod
    def apply_controlgate(control_qbit, target_qbit, unitary_mat):
        a = np.array([[0, 0],
                      [0, 1]])

        b = np.array([[1, 0],
                      [0, 0]])

        cntrl_mat = np.kron(a, unitary_mat) + np.kron(b, np.identity(2))
        print(cntrl_mat)
        print(control_qbit)
        print(target_qbit)
        combined_state = np.kron(control_qbit.state, target_qbit.state)
        print(combined_state)
        resultant_state = cntrl_mat @ combined_state
        print(resultant_state)

        control_qbit.state, target_qbit.state = split_state(resultant_state)
        return

    @staticmethod
    def apply_gate(qbit, unitary_mat):
        qbit.state = unitary_mat @ qbit.state
        return

    def measure(self, qbit_lst, bit_lst, trails=100):
        assert len(qbit_lst) == len(bit_lst)
        assert len(bit_lst) == self.num_bits

        circ_depth = len(self.gate_array[0])
        for layer in range(circ_depth):
            print('layer {}'.format(layer))
            for index, qbit in enumerate(self.lst_qbits):
                meta_tuple = self.gate_array[index][layer]
                gate_str = meta_tuple[0]

                if gate_str == '-' or gate_str[0] == 'T':
                    pass

                elif gate_str == 'CQ':
                    target_mat = self.gate_to_mat[meta_tuple[3]]
                    target_qbit = self.lst_qbits[meta_tuple[2]]
                    control_qbit = qbit

                    self.apply_controlgate(control_qbit, target_qbit, target_mat)

                else:
                    unitary_mat = self.gate_to_mat[gate_str]
                    self.apply_gate(qbit, unitary_mat)

                print('q{0}: {1}|0> + {2}|1>'.format(index, qbit.state[0], qbit.state[1]))

        results_lst = []
        for qbit in self.lst_qbits:
            result = qbit.measure(shots=trails)
            results_lst.append(result)

        final_results_lst = np.zeros((self.num_bits, trails))
        for i, j in zip(qbit_lst, bit_lst):
            final_results_lst[j] = results_lst[i]

        final_results_lst = np.transpose(final_results_lst)
        counts_dict = {}
        for row in final_results_lst:
            state_label = lst_to_str(row)
            if state_label in counts_dict:
                counts_dict[state_label] += 1
            else:
                counts_dict[state_label] = 1

        return counts_dict

    @staticmethod
    def plot_counts(counts):
        plt.bar(range(len(counts)), counts.values())
        plt.xticks(range(len(counts)), list(counts.keys()))
        plt.show()
        return

    def h(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'H')
        return

    def ch(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'H')
        return

    def x(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'X')
        return

    def cx(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'X')
        return

    def y(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'Y')
        return

    def cy(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'Y')
        return

    def z(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'Z')
        return

    def cz(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'Z')
        return


def main():
    circ = Circuit(2, 2)
    circ.h(0)
    circ.cx(1, 0)
    print(circ)    # constructed the bell state
    counts = circ.measure([0, 1], [0, 1], trails=1024)
    circ.plot_counts(counts)
    return


if __name__ == "__main__":
    main()
