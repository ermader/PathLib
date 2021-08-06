"""\
Test for PathUtilities

Created on August 6, 2021

@author Eric Mader
"""

import math
from PathLib.PathUtilities import length, slope, midpoint, intersectionPoint, pointOnLine

p0 = (0, 0)
p1 = (300, 0)
p2 = (300,400)
l0 = [p0, p1]
l1 = [p1, p2]
l2 = [p0, p2]

l3 = [(0, 0), (100, 100)]
l4 = [(50, 200), (200, 50)]

def test_length():
    assert length(l0) == 300.0
    assert length(l1) == 400.0
    assert length(l2) == 500.0

def test_slope():
    assert slope(l0) == 0
    assert slope(l1) == math.inf
    assert slope(l2) == 4.0/3.0
    assert slope([p2, p0]) == 4.0/3.0

def test_midpoint():
    assert midpoint(l0) == (150.0, 0.0)
    assert midpoint(l1) == (300.0, 200.0)
    assert midpoint(l2) == (150.0, 200.0)

def test_intersectionPoint():
    assert intersectionPoint([(0,1), (4,5)], [(4, 2), (0,4)]) == (2.0, 3.0)
    assert intersectionPoint(l0, l1) == (300.0, 0.0)
    assert intersectionPoint(l0, l2) == (0.0, 0.0)
    assert intersectionPoint(l1, l2) == (300.0, 400.0)
    assert intersectionPoint(l3, l4) == None

def test_pointOnLine():
    assert pointOnLine((150, 200), l2)
    assert not pointOnLine((-300, -400), l2)