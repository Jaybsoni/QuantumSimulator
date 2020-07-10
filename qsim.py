# Main file for the simulator
import numpy as np
import random
from numpy import linalg as la


class Qbit:

    def __init__(self, c1, c2):
        assert (la.norm(c1) + la.norm(c2) == 1)
        self.state_0 = c1
        self.state_1 = c2

    def __repr__(self):
        return "{0}|0> + {1}|1>".format(self.state_0, self.state_1)


def main():
    x = Qbit(1, 0)
    print(x)


if __name__ == "__main__":
    main()
