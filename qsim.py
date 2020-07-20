# Main file for the simulator
import numpy as np
from numpy import linalg as la


class Qbit:

    def __init__(self, c1, c2):
        try:
            assert (la.norm(c1)**2 + la.norm(c2)**2 == 1)
        except AssertionError:
            # print(la.norm(c1)**2 + la.norm(c2)**2)
            pass

        self.state = np.array([c1, c2])

    def __repr__(self):
        return "{0}|0> + {1}|1>".format(self.state[0], self.state[1])

    def measure(self):
        prob_0 = la.norm(self.state[0])
        result = np.random.binomial(1, prob_0, size=1)
        if result:
            self.state = np.array([0, 1])
        else:
            self.state = np.array([0, 1])
        print(self)


class Circuit:

    def __init__(self, num_qbits, circuit_depth):
        assert(num_qbits > 0)
        self.circuit_depth = circuit_depth
        self.num_qbits = num_qbits
        self.lst_qbits = []
        self.gate_array = []

        for i in range(num_qbits):
            self.lst_qbits.append(Qbit(1, 0))

            gates = []
            for j in range(circuit_depth):
                gates.append('-')

            self.gate_array.append(gates)
        return

    def __repr__(self):
        disp_circuit = ''
        for i in range(self.num_qbits):
            row = '|q{}>: '.format(i)
            for gate_element in self.gate_array[i]:
                row += '--{}'.format(gate_element)
            row += '--M \n'
            disp_circuit += row
        return disp_circuit

    def set_gates(self):
        print('circuit depth, num of qbits = {0}, {1}'.format(self.circuit_depth, self.num_qbits))
        print('hadamard gate: H \n' +
              'pauli x gate: X \n' +
              'pauli y gate: Y \n' +
              'pauli z gate: Z \n ' +
              'cnot gate: CN \n')

        for i in range(self.circuit_depth):
            add_gate = input('input gate,qbit_index at circuit depth {} ' +
                             '(for cnot: CN,control_index,target_index): '.format(i + 1))

            while add_gate != '':
                print(add_gate)
                meta = add_gate.split(',')
                print(meta)
                meta_gate = meta[0]  # gate
                meta_qbit1 = int(meta[1])  # for cnot gate its control index, else its qbit index
                meta_qbit2 = int(meta[-1])  # for cnot gate its target index, else its qbit index

                if meta_gate == 'CN':
                    self.gate_array[meta_qbit1][i] = 'C'
                    self.gate_array[meta_qbit2][i] = 'T'

                else:
                    self.gate_array[meta_qbit1][i] = meta_gate

                print(self)
                add_gate = input('input gate,qbit_index at circuit depth {} ' +
                                 '(for cnot: CN,control_index,target_index): '.format(i + 1))
        return

    @staticmethod
    def apply_gate(gate_str, *qbits):
        if gate_str == 'H':
            h_gate(qbits[0])

        elif gate_str == 'X':
            pauli_x(qbits[0])

        elif gate_str == 'Y':
            pauli_y(qbits[0])

        elif gate_str == 'Z':
            pauli_z(qbits[0])

        elif gate_str == 'CN':
            cnot(qbits[0], qbits[1])

        else:
            print('Invalid gate!')
        return

    def measure_circ(self):
        for i in range(self.circuit_depth):
            for j in range(self.num_qbits):
                curr_qbit = self.lst_qbits[j]


def split_state(vect):
    assert(len(vect) == 4)
    qbit1 = Qbit(vect[0] + vect[1], vect[2] + vect[3])
    qbit2 = Qbit(vect[0] + vect[2], vect[1] + vect[3])
    return qbit1, qbit2


def h_gate(qbit):
    hadamard_mat = 1 / np.sqrt(2) * np.array([[1,  1],
                                              [1, -1]])
    qbit.state = hadamard_mat @ qbit.state
    return


def cnot(control_qbit, target_qbit):
    combined_state = np.kron(control_qbit.state, target_qbit.state)
    cnot_mat = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]])

    resultant_state = cnot_mat @ combined_state
    return split_state(resultant_state)


def pauli_x(qbit):
    x_mat = np.array([[0, 1],
                      [1, 0]])

    qbit.state = x_mat @ qbit.state
    return


def pauli_y(qbit):
    y_mat = np.array([[0, -j],
                      [j, 0]])

    qbit.state = y_mat @ qbit.state
    return


def pauli_z(qbit):
    z_mat = np.array([[1, 0],
                      [0, -1]])

    qbit.state = z_mat @ qbit.state
    return


def main():
    return


if __name__ == "__main__":
    main()