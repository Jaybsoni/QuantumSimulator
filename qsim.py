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
            for gate_element in self.gate_array[i]:
                row += '--{0}'.format(gate_element)
            row += '\n'
            disp_circuit += row
        return disp_circuit

    def add_sq_gate(self, qbit_ind, gate_str):
        if type(qbit_ind) is list:
            pass
        else:
            qbit_ind = [qbit_ind]

        for index, gate_lst in enumerate(self.gate_array):
            if index in qbit_ind:
                gate_lst.append(gate_str)
            else:
                gate_lst.append('-')
        return

    def add_dq_gate(self, control_ind, target_ind, gate_str):
        for index, gate_lst in enumerate(self.gate_array):
            if index == control_ind:
                gate_lst.append('CQ')
            elif index == target_ind:
                gate_lst.append('T' + gate_str)
            else:
                gate_lst.append('-')
        return

    @staticmethod
    def apply_controlgate(control_qbit, target_qbit, unitary_mat):
        a = np.array([[0, 0],
                      [0, 1]])

        b = np.array([[1, 0],
                      [0, 0]])

        cntrl_mat = np.kron(a, unitary_mat) + np.kron(b, np.identity(2))
        combined_state = np.kron(control_qbit.state, target_qbit.state)
        resultant_state = cntrl_mat @ combined_state

        control_qbit.state, target_qbit.state = split_state(resultant_state)
        return

    def n(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'N')
        return

    @staticmethod
    def apply_n(qbit=Qbit(1, 0)):
        not_mat = np.array([[0, 1],
                            [1, 0]])
        qbit.state = not_mat @ qbit.state
        return not_mat

    def cn(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'N')
        return

    def apply_cn(self, control_qbit, target_qbit):
        self.apply_controlgate(control_qbit, target_qbit, self.apply_n())
        return

    def h(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'H')
        return

    @staticmethod
    def apply_h(qbit=Qbit(1, 0)):
        hadamard_mat = 1 / np.sqrt(2) * np.array([[1,  1],
                                                  [1, -1]])
        qbit.state = hadamard_mat @ qbit.state
        return hadamard_mat

    def ch(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'H')
        return

    def apply_ch(self, control_qbit, target_qbit):
        self.apply_controlgate(control_qbit, target_qbit, self.apply_h())
        return

    def x(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'X')
        return

    @staticmethod
    def apply_x(qbit=Qbit(1, 0)):
        x_mat = np.array([[0, 1],
                          [1, 0]])

        qbit.state = x_mat @ qbit.state
        return x_mat

    def cx(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'X')
        return

    def apply_cx(self, control_qbit, target_qbit):
        self.apply_controlgate(control_qbit, target_qbit, self.apply_x())
        return

    def y(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'Y')
        return

    @staticmethod
    def apply_y(qbit=Qbit(1, 0)):
        y_mat = np.array([[0, -j],
                          [j, 0]])

        qbit.state = y_mat @ qbit.state
        return y_mat

    def cy(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'Y')
        return

    def apply_cy(self, control_qbit, target_qbit):
        self.apply_controlgate(control_qbit, target_qbit, self.apply_y())
        return

    def z(self, qbit_ind):
        self.add_sq_gate(qbit_ind, 'Z')
        return

    @staticmethod
    def apply_z(qbit=Qbit(1, 0)):
        z_mat = np.array([[1, 0],
                          [0, -1]])

        qbit.state = z_mat @ qbit.state
        return z_mat

    def cz(self, control_ind, target_ind):
        self.add_dq_gate(control_ind, target_ind, 'Z')
        return

    def apply_cz(self, control_qbit, target_qbit):
        self.apply_controlgate(control_qbit, target_qbit, self.apply_z())
        return


def main():
    return


if __name__ == "__main__":
    main()
