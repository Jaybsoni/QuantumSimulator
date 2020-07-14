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


def h_gate(qbit):
    hadamard_mat = 1 / np.sqrt(2) * np.array([[1,  1],
                                              [1, -1]])
    qbit.state = qbit.state @ hadamard_mat
    return


def main():
    return


if __name__ == "__main__":
    main()
