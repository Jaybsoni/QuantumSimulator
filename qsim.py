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

        self.state_0 = c1
        self.state_1 = c2

    def __repr__(self):
        return "{0}|0> + {1}|1>".format(self.state_0, self.state_1)

    def measure(self):
        prob_0 = la.norm(self.state_0)
        result = np.random.binomial(1, prob_0, size=1)
        if result:
            self.state_0 = 1
            self.state_1 = 0
        else:
            self.state_0 = 0
            self.state_1 = 1
        print(self)


def main():
    for i in range(5):
        x = Qbit(1 / (np.sqrt(2)), 1 / (np.sqrt(2)))
        print('Measurement #{0}: |x> = {1}'.format(i, x))
        x.measure()
        print('\n')


if __name__ == "__main__":
    main()
