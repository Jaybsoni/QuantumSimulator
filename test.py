import numpy as np
import itertools


def q_bit_ittorator(bit1, bit2, n):
    """
    Given two tuples with element < n, create all possible binary integers
    of bit length n with the known bit1, bit2 positions. Return list of
    ints that represent the base 10 of those binary numbers

    :param bit1: tuple - (element, bool)
    :param bit2: tuple - (element, bool)
    :param n: int
    :return: list of ints
    """
    try:
        assert(n > bit1[0])
        assert(n > bit2[0])
    except AssertionError as e:
        print(f"Error: n ({n}) must be larger than the index of each bit")
        print(e)
    else:
        # itertools creates all possible binary combinations for n bites
        binary_combinations = [list(i) for i in itertools.product([0, 1], repeat=n)]
        return_list = []  # Create an empty list to be populated with all possible ints

        print(binary_combinations)  # For debuging only
        # For all elements, replace the bit1 element to bit1 value, same for bit2
        for i in range(len(binary_combinations)):
            binary_combinations[i][bit1[0]] = int(bit1[1])
            binary_combinations[i][bit2[0]] = int(bit2[1])

            # Convert each list of ints (1s & 0s) to a list of strings
            strings = [str(i) for i in binary_combinations[i]]
            as_a_int = int("".join(strings), 2)   # Join, then turn to int

            # Only add to the return list if it's unique
            if as_a_int not in return_list:
                return_list.append(as_a_int)

        print(binary_combinations)  # For debuging only
        return return_list
