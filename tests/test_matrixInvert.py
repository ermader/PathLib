"""\
Test for MatrixInversion

Created on August 6, 2021

@author Eric Mader
"""

from PathLib.MatrixInversion import matrixInvert
from PathLib.CurveFitting import multiply

def test_matrixInversion():
    expectedInversion = [[7.0, -3.0, -3.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]
    identityMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    # test case from http://blog.acipo.com/matrix-inversion-in-javascript/
    M = [[1, 3, 3], [1, 4, 3], [1, 3, 4]]
    actualInversion = matrixInvert(M)

    assert actualInversion == expectedInversion
    assert multiply(M, actualInversion) == identityMatrix


