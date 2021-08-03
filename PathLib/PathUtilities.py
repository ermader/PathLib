"""\
Utilities for manipulating outline paths and segments

Created on July 7, 2020

@author Eric Mader
"""

import math
from .Transform import Transform

from .BezierUtilities import lli

class BoundsRectangle(object):
    """\
    A bounds rectangle for a set of points.
    """

    relationEncloses = 0
    relationEnclosed = 1
    relationIntersects = 2
    relationSeparate = 3

    def __init__(self, *points):
        """\
        Initialize a bounds rectangle that encloses the given
        list of points.

        Returns an empty rectangle if the list is empty.
        """
        right = top = -32768
        left = bottom = 32768

        for point in points:
            px, py = point
            left = min(left, px)
            right = max(right, px)
            bottom = min(bottom, py)
            top = max(top, py)

        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    @staticmethod
    def fromContour(contour):
        """\
        Return a BoundsRectangle that encloses the points in contour.
        """
        bounds = BoundsRectangle()
        for segment in contour:
            bounds = bounds.union(BoundsRectangle(*segment))

        return bounds

    @staticmethod
    def fromCoutours(contours):
        """\
        Return a BoundsRectangle that encloses the points in contours.
        """
        bounds = BoundsRectangle()
        for contour in contours:
            bounds = bounds.union(BoundsRectangle.fromContour(contour))

        return bounds

    def __str__(self):
        return f"[({self.left}, {self.bottom}), ({self.right}, {self.top})]"

    @property
    def width(self):
        """\
        The width of the bounds rectangle.
        """
        return self.right - self.left

    @property
    def height(self):
        """\
        The height of the bounds rectangle.
        """
        return self.top - self.bottom

    @property
    def area(self):
        """\
        The area of the rectangle.
        """
        return self.width * self.height

    @property
    def diagonal(self):
        """\
        A line from the bottom left corner to the top right corner.
        """
        return [(self.left, self.bottom), (self.right, self.top)]

    @property
    def contour(self):
        p0 = (self.left, self.bottom)
        p1 = (self.left, self.top)
        p2 = (self.right, self.top)
        p3 = (self.right, self.bottom)
        return [[p0, p1], [p1, p2], [p2, p3], [p3, p0]]

    @property
    def centerPoint(self):
        """\
        The center point of the rectangle.
        """
        return midpoint(self.diagonal)

    @property
    def points(self):
        """\
        A list of the min, max x, y coordinates of the rectangle.
        """
        return (self.left, self.bottom, self.right, self.top)

    def yFromBottom(self, percent):
        """\
        Return the y coordinate value a percentage of the way from the bottom edge of the rectangle.
        """
        return self.bottom + self.height * percent

    def xFromLeft(self, percent):
        """\
        Return the x coordinate value a percentage of the way from the left edge of the rectangle.
        """
        return self.left + self.width * percent

    def enclosesPoint(self, point):
        """\
        Test if the given point is within the rectangle.
        """
        px, py = point

        return self.left <= px <= self.right and self.bottom <= py <= self.top

    def encloses(self, other):
        """\
        Test if this rectangle encloses the given rectangle.
        """
        otherDiagonal = other.diagonal
        return self.enclosesPoint(otherDiagonal[0]) and self.enclosesPoint(otherDiagonal[1])

    def crossesX(self, x):
        """\
        Test if the given x coordinate is within the rectangle.
        """
        return self.left <= x <= self.right

    def crossesY(self, y):
        """\
        Test if the given y coordinate is within the rectangle.
        """
        return self.bottom <= y <= self.top

    def union(self, other):
        """\
        Return a rectangle that is the union of this rectangle and other.
        """
        newLeft = min(self.left, other.left)
        newTop = max(self.top, other.top)
        newRight = max(self.right, other.right)
        newBottom = min(self.bottom, other.bottom)

        return BoundsRectangle((newLeft, newBottom), (newRight, newTop))

    def intersection(self, other):
        """\
        Return a rectangle that is the intersection of this rectangle and other.
        """
        newLeft = max(self.left, other.left)
        newTop = min(self.top, other.top)
        newRight = min(self.right, other.right)
        newBottom = max(self.bottom, other.bottom)

        if newRight < newLeft or newTop < newBottom: return None  # maybe want <=, >=?
        return BoundsRectangle((newLeft, newBottom), (newRight, newTop))

    def relationTo(self, other):
        if self.encloses(other):
            return self.relationEncloses
        elif other.encloses(self):
            return self.relationEnclosed
        elif self.intersection(other):
            return self.relationIntersects
        else:
            return self.relationSeparate

def pointXY(point):
    if isinstance(point, complex):
        return point.real, point.imag
    return point

def minMax(a, b):
    """\
    Return a tuple with the min value first, then the max value.
    """
    return (a, b) if a <= b else (b, a)

def endPoints(segment):
    """\
    Return the x, y coordinates of the start and end of the segment.
    """
    p0x, p0y = pointXY(segment[0])
    p1x, p1y = pointXY(segment[-1])

    return (p0x, p0y, p1x, p1y)

def getDeltas(segment):
    """\
    Return the x, y span of the segment.
    """
    p0x, p0y, p1x, p1y = endPoints(segment)

    return (p1x - p0x, p1y - p0y)

def isVerticalLine(segment):
    """\
    Test if the segment is a vertical line.
    """
    dx, _ = getDeltas(segment)
    return len(segment) == 2 and dx == 0

def isHorizontalLine(segment):
    """\
    Test if the segment is a horizontal line.
    """
    _, dy = getDeltas(segment)
    return len(segment) == 2 and dy == 0

def isDiagonalLine(segment):
    dx, dy = getDeltas(segment)
    return len(segment) == 2 and dx !=0 and dy != 0

def length(segment):
    """\
    Return the length of the segment. Only really makes sense for a line...
    """
    return math.hypot(*getDeltas(segment))

def slope(segment):
    """\
    Return the slope of the segment. rise / run. Returns
    math.inf if the line is vertical.
    """
    dx, dy = getDeltas(segment)

    if dx == 0: return math.inf
    return dy / dx

def slopeAngle(segment):
    """\
    Return the angle of the segment from vertical, in degrees.
    """
    dx, dy = getDeltas(segment)
    return math.degrees(math.atan2(abs(dx), abs(dy)))

def lineSlopeAngle(line):
    delta = line.end - line.start
    return math.degrees(math.atan2(abs(delta.real), abs(delta.imag)))

def rawSlopeAngle(segment):
    dx, dy = getDeltas(segment)
    return math.degrees(math.atan2(dy, dx))

def midpoint(line):
    """\
    Return the midpoint of the line.
    """
    p0x, p0y, p1x, p1y = endPoints(line)

    return ((p0x + p1x) / 2, (p0y + p1y) / 2)

def intersectionPoint(l1, l2):
    """\
    Find the intersection point of the two lines.
    """
    intersection = lli(l1, l2)

    b1 = BoundsRectangle(*l1)
    b2 = BoundsRectangle(*l2)

    # The point calculated above assumes that the two lines have
    # infinite length, so it may not be on both, or either line.
    # Make sure it's within the bounds rectangle for both lines.
    return intersection if b1.enclosesPoint(intersection) and b2.enclosesPoint(intersection) else None

# The result of this function cannot be used to create an SVG path...
def flatten(contours):
    """\
    Return a single contour that contains all the points in the given contours.
    """
    return [segment for contour in contours for segment in contour]

# There must be a better way to do this...
def pointOnLine(point, line):
    """\
    Test if a given point is on the given line.
    """
    bounds = BoundsRectangle(*line)

    # If the bounds rectangle of the line encloses the point and
    # a line from the start of the given line to the point has the
    # same slope as the line, it is on the line.
    return bounds.enclosesPoint(point) and slope(line) == slope([line[0], point])


def rotatePointAbout(point, about, degrees=90, ccw=True):
    """\
    Rotate the given point the given number of degrees about the point about
    in a clockwise or counter-clockwise direction.
    """
    rt = Transform.rotationAbout(about, degrees, ccw)

    return rt.applyToPoint(point)

def rotateSegmentAbout(segment, about, degrees=90, ccw=True):
    """\
    Rotate the given segment the given number of degrees about the point about
    in a clockwise or counter-clockwise direction.
    """
    rt = Transform.rotationAbout(about, degrees, ccw)

    return rt.applyToSegment(segment)

def rotateContourAbout(contour, about, degrees=90, ccw=True):
    """\
    Rotate the given contour the given number of degrees about the point about
    in a clockwise or counter-clockwise direction.
    """
    rt = Transform.rotationAbout(about, degrees, ccw)

    return rt.applyToContour(contour)

def rotateContoursAbout(contours, about, degrees=90, ccw=True):
    """\
    Rotate the given contours the given number of degrees about the point about
    in a clockwise or counter-clockwise direction.
    """
    rt = Transform.rotationAbout(about, degrees, ccw)

    return rt.applyToContours(contours)

def toMicros(funits, unitsPerEM):
    """\
    Convert funits into micros.
    """
    return funits * 1000 / unitsPerEM

# Helvetica Neue X
xContours = [[[(248, 367), (0, 0)], [(0, 0), (106, 0)], [(106, 0), (304, 295)], [(304, 295), (496, 0)], [(496, 0), (612, 0)], [(612, 0), (361, 367)], [(361, 367), (597, 714)], [(597, 714), (491, 714)], [(491, 714), (305, 435)], [(305, 435), (127, 714)], [(127, 714), (13, 714)], [(13, 714), (248, 367)]]]
helveticaNeueUPM = 1000

p0 = (0, 0)
p1 = (300, 0)
p2 = (300,400)
l0 = [p0, p1]
l1 = [p1, p2]
l2 = [p0, p2]

l3 = [(0, 0), (100, 100)]
l4 = [(50, 200), (200, 50)]

def test():
    print(f"length(l0) = {length(l0)}, length(l1) = {length(l1)}, length(l2) = {length(l2)}")
    print(f"slope(l0) = {slope(l0)}, slope(l1) = {slope(l1)}, slope(l2) = {slope(l2)}, slope([p2, p0]) = {slope([p2, p0])}")
    print(f"midpoint(l0) = {midpoint(l0)}, midpoint(l1) = {midpoint(l1)}, midpoint(l2) = {midpoint(l2)}")
    print(f"intersection([(0,1), (4,5)], [(4, 2), (0,4)]) = {intersectionPoint([(0,1), (4,5)], [(4, 2), (0,4)])}")
    print(f"intersectionPoint(l0, l1) = {intersectionPoint(l0, l1)}")
    print(f"intersectionPoint(l0, l2) = {intersectionPoint(l0, l2)}")
    print(f"intersectionPoint(l1, l2) = {intersectionPoint(l1, l2)}")
    print(f"intersectionPoint(l3, l4) = {intersectionPoint(l3, l4)}")

    print(pointOnLine((150, 200), l2))
    print(pointOnLine((-300, -400), l2))
    print()

    #
    # Example 2-6 from Mathematical Elements for Computer Graphics
    # Second Edition
    #
    print("Example 2-6 from Mathematical Elements for Computer Graphics:")
    m1 = [[1, 0, 0], [0, 1, 0], [-4, -3, 1]]
    m2 = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    m3 = [[1, 0, 0], [0, 1, 0], [4, 3, 1]]

    fp = Transform(m1, m2, m3)
    print(f"rotation transform = {fp.transform}")
    print(f"rotation of (8, 6) = {fp.applyToPoint((8, 6))}")

    # s1 = [(253, 239), (242, 210), (216, 136), (199, 80)]
    # s2 = [(253, 239), (242, 210), (229, 173), (216, 136), (199, 80)]
    # m1 = Transform._rotationMatrix(45)
    # transform = Transform(m1)
    # r1 = transform.applyToSegment(s1)
    # r2 = transform.applyToSegment(s2)
    # print(r1)
    # print(r2)
if __name__ == "__main__":
    test()
