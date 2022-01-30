"""\
Test for CurveFitting

Created on January 29, 2022

@author Eric Mader
"""

# from PathLib.Transform import Matrix
from PathLib.CurveFitting import multiply, transpose

Row = list[float]
Matrix = list[Row]

def test_multiply():
    m: Matrix = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    pr: Matrix = [[100, 200, 1]]
    pc: Matrix = [[100], [200], [1]]
    mpc: Matrix = [[-100], [-200], [1]]
    prm: Matrix = [[-100, -200, 1]]

    assert multiply(m, pc) == mpc
    assert multiply(pr, m) == prm


def test_transpose():
    m: Matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    t: Matrix = [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]

    assert transpose(m) == t

