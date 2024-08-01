#from .Flow import *

# print current working directory
import os
print(os.getcwd())

from .utils import standardize
from .LAMINAR import LAMINAR


def add_one(x: int):
    """An example function that increases a number

    :param x: The input parameter to increase
    :return: The successor of the given number
    """
    return x + 1
