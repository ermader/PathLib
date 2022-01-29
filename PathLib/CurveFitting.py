"""\
Matrix inversion

Created on September 10, 2020

Much of this code translated from curve-fitter.js and others
from https://github.com/Pomax/BezierInfo-2

@author Eric Mader
"""

import typing

import math
from .PathTypes import Point
from .Transform import Matrix, Row
from .MatrixInversion import matrixInvert

binomialCoefficients = [
           [1],                # n = 0
         [1, 1],               # n = 1
        [1, 2, 1],             # n = 2
       [1, 3, 3, 1],           # n = 3
      [1, 4, 6, 4, 1],         # n = 4
     [1, 5, 10, 10, 5, 1],     # n = 5
    [1, 6, 15, 20, 15, 6, 1]]  # n = 6

def dist(p1: Point, p2: Point):
    p1x, p1y = p1
    p2x, p2y = p2
    dx = p1x - p2x
    dy = p1y - p2y

    return math.hypot(dx, dy)

def binomial(n: int, k: int):
    if n == 0: return 1
    lut = binomialCoefficients
    while n >= len(lut):
        nextRow = [1]
        prev = len(lut) - 1
        for i in range(1, len(lut)):
            nextRow.append(lut[prev][i-1] + lut[prev][i])
        nextRow.append(1)
        lut.append(nextRow)
    return lut[n][k]

def getPointValuesColumn(points: list[Point], v: int):
    return [[p[v]] for p in points]

def computeTimeValues(p: list[Point], n: typing.Optional[int] = None, polygonal: bool = True) -> list[float]:
    if not n: n = len(p)

    if polygonal:
        d: list[float] = [0]
        for i in range(1, n):
            d.append(d[i-1] + dist(p[i-1], p[i]))

        length = d[n - 1]
        return [v / length for v in d]
    else:
        return [i / (n-1) for i in range(n)]


def raiseRowPower(row: Row, i: int) -> Row:
    return [math.pow(v, i) for v in row]

def basisMatrix(n: int) -> Matrix:
    """\
    Return the (n x n) basis matrix
    We can form any basis matrix using a generative approach:

     - it's a lower triangular matrix: all the entries above the main diagonal are zero
     - the main diagonal consists of the binomial coefficients for n
     - all entries are symmetric about the antidiagonal.

    What's more, if we number rows and columns starting at 0, then
    the value at position M[r,c], with row=r and column=c, can be
    expressed as:

      M[r,c] = (r choose c) * M[r,r] * S,

      where S = 1 if r+c is even, or -1 otherwise

    That is: the values in column c are directly computed off of the
    binomial coefficients on the main diagonal, through multiplication
    by a binomial based on matrix position, with the sign of the value
    also determined by matrix position. This is actually very easy to
    write out in code:
"""
    m: Matrix = [[0 for _ in range(n)] for _ in range(n)]
    k = n - 1

    # populate the main diagonal
    for i in range(n):
        m[i][i] = binomial(k, i)

    # compute the remaining values
    for c in range(n):
        for r in range(c+1, n):
            sign = 1 if (r+c) & 1 == 0 else -1
            value = binomial(r, c) * m[r][r]
            m[r][c] = sign * value

    return m

def transpose(m: Matrix) -> Matrix:
    t: Matrix = []
    for column in range(len(m[0])):
        tc: Row = []
        for row in range(len(m)):
            tc.append(m[row][column])
        t.append(tc)

    return t

def formTMatrix(s: Row, n: typing.Optional[int] = None) -> tuple[Matrix, Matrix]:
    if not n: n = len(s)
    tp: Matrix = []

    # it's easier to generate the transposed matrix:
    for i in range(n):
        tp.append(raiseRowPower(s, i))

    return (tp, transpose(tp))

def row(m: Matrix, i: int) -> Row:
    return m[i]

def col(m: Matrix, i: int):
    return [r[i] for r in m]

# JavaScript code:
# function multiply(M1, M2) {
#   // prep
#   var M = [];
#   var dims = [M1.length, M1[0].length, M2.length, M2[0].length];
#   // work
#   for (var r=0, c; r<dims[0]; r++) {
#     M[r] = [];
#     var _row = row(M1, r);
#     for (c=0; c<dims[3]; c++) {
#       var _col = col(M2,c);
#       var reducer = (a,v,i) => a + _col[i]*v;
#       M[r][c] = _row.reduce(reducer, 0);
#     }
#   }
#   return M;
# }
def multiply(M1: Matrix, M2: Matrix) -> Matrix:
    M : Matrix= []
    for r in range(len(M1)):
        mr: Row = []
        _row: Row = row(M1, r)
        for c in range(len(M2[0])):
            _col = col(M2, c)
            # Python reduce callback doesn't include the index
            # so we have to do this by hand
            a = 0
            for i in range(len(_row)):
                a += _col[i] * _row[i]
            mr.append(a)

        M.append(mr)

    return M

def bestFit(P: list[Point], M: Matrix, S: list[float], n: typing.Optional[int] = None) -> tuple[Matrix, Matrix]:
    if not n: n = len(P)

    Tt, T = formTMatrix(S, n)
    M1 = typing.cast(Matrix, matrixInvert(M))
    TtT1 = typing.cast(Matrix, matrixInvert(multiply(Tt, T)))
    step1 = multiply(TtT1, Tt)
    step2 = multiply(M1, step1)
    X = getPointValuesColumn(P, 0)
    Cx = multiply(step2, X)
    Y = getPointValuesColumn(P, 1)
    Cy = multiply(step2, Y)

    return (Cx, Cy)

def fit(points: list[Point], polygonal: bool = True):
    n = len(points)
    m = basisMatrix(n)
    s = computeTimeValues(points, polygonal=polygonal)
    c = bestFit(points, m, s)

    return (points, m, s, c)

def test():
    m: Matrix = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    pr: Matrix = [[100, 200, 1]]
    pc: Matrix = [[100], [200], [1]]
    print(multiply(m, pc))
    print(multiply(pr, m))

    m: Matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print(transpose(m))

    # points = [(70, 120), (80, 160), (110, 170), (120, 120)]
    #
    # f = fit(points)

if __name__ == "__main__":
    test()
