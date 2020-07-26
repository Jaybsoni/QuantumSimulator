# Main file for the simulator
import numpy as np
from numpy import linalg as la


def split_state(vect):
    assert (len(vect) == 4)
    qbit1 = Qbit(vect[0] + vect[1], vect[2] + vect[3])
    qbit2 = Qbit(vect[0] + vect[2], vect[1] + vect[3])
    return qbit1, qbit2


class Qbit:

    def __init__(self, c1, c2):
        try:
            assert (la.norm(c1)**2 + la.norm(c2)**2 == 1)
        except AssertionError:
            c1 = 1
            c2 = 0
            pass

        self.state = np.array([c1, c2])

    def __repr__(self):
        return "{0}|0> + {1}|1>".format(self.state[0], self.state[1])

    def measure(self, shots=1):
        prob_0 = la.norm(self.state[0])
        result = np.random.binomial(1, prob_0, size=shots)
        return result


class Circuit:

    def __init__(self, num_qbits):
        assert(num_qbits > 0)
        self.lst_qbits = []
        self.gate_array = []

        for i in range(num_qbits):
            self.lst_qbits.append(Qbit(1, 0))
            self.gate_array.append([])
        return

    def __repr__(self):
        disp_circuit = ''
        for i in range(self.num_qbits):
            row = 'q{}: |0>'.format(i)
            for gate_element in self.gate_array[i]:
                row += '--{}'.format(gate_element)
            row += '\n'
            disp_circuit += row
        return disp_circuit

    def measure_circ(self, shots=100):
        return

    @staticmethod
    def apply_h(qbit):
        hadamard_mat = 1 / np.sqrt(2) * np.array([[1,  1],
                                                  [1, -1]])
        qbit.state = hadamard_mat @ qbit.state
        return

    @staticmethod
    def apply_cnot(control_qbit, target_qbit):
        combined_state = np.kron(control_qbit.state, target_qbit.state)
        cnot_mat = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]])

        resultant_state = cnot_mat @ combined_state
        return split_state(resultant_state)

    @staticmethod
    def apply_x(qbit):
        x_mat = np.array([[0, 1],
                          [1, 0]])

        qbit.state = x_mat @ qbit.state
        return

    @staticmethod
    def apply_y(qbit):
        y_mat = np.array([[0, -j],
                          [j, 0]])

        qbit.state = y_mat @ qbit.state
        return

    @staticmethod
    def apply_z(qbit):
        z_mat = np.array([[1, 0],
                          [0, -1]])

        qbit.state = z_mat @ qbit.state
        return


def main():
    return


if __name__ == "__main__":
    main()
