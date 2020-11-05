import random
import itertools
from neural_network.constants import BLACK_INT, WHITE_INT


def create_combinations() -> list:

    """
    Returns a list of all combination of
    black/white in a 2x2 board.
    """

    # -- Create Combinations -- #
    combinations = list(
        itertools.product(
            [BLACK_INT, WHITE_INT],
            repeat=4
        )
    )

    # -- Shuffle them for randomness -- #
    random.shuffle(combinations)

    # -- Return list -- #
    return combinations
