"""\
Utilities for manipulating outline paths and segments

Created on July 7, 2020

@author Eric Mader
"""

from __future__ import annotations
import typing

import math
from .PathTypes import Point, Segment, Contour
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

    def __init__(self, *points: Point):
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
    def fromContour(contour: Contour):
        """\
        Return a BoundsRectangle that encloses the points in contour.
        """
        bounds = BoundsRectangle()
        for segment in contour:
            bounds = bounds.union(BoundsRectangle(*segment))

        return bounds

    @staticmethod
    def fromCoutours(contours: typing.Sequence[Contour]):
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
    def area(self) -> float:
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
    def contour(self) -> Contour:
        p0 = (self.left, self.bottom)
        p1 = (self.left, self.top)
        p2 = (self.right, self.top)
        p3 = (self.right, self.bottom)
        return [[p0, p1], [p1, p2], [p2, p3], [p3, p0]]

    @property
    def centerPoint(self) -> Point:
        """\
        The center point of the rectangle.
        """
        return midpoint(self.diagonal)

    @property
    def points(self) -> tuple[float, ...]:
        """\
        A list of the min, max x, y coordinates of the rectangle.
        """
        return (self.left, self.bottom, self.right, self.top)

    def yFromBottom(self, percent: float) -> float:
        """\
        Return the y coordinate value a percentage of the way from the bottom edge of the rectangle.
        """
        return self.bottom + self.height * percent

    def xFromLeft(self, percent: float) -> float:
        """\
        Return the x coordinate value a percentage of the way from the left edge of the rectangle.
        """
        return self.left + self.width * percent

    def enclosesPoint(self, point: Point) -> bool:
        """\
        Test if the given point is within the rectangle.
        """
        px, py = point

        return self.left <= px <= self.right and self.bottom <= py <= self.top

    def encloses(self, other: BoundsRectangle):
        """\
        Test if this rectangle encloses the given rectangle.
        """
        otherDiagonal = other.diagonal
        return self.enclosesPoint(otherDiagonal[0]) and self.enclosesPoint(otherDiagonal[1])

    def crossesX(self, x: float) -> bool:
        """\
        Test if the given x coordinate is within the rectangle.
        """
        return self.left <= x <= self.right

    def crossesY(self, y: float) -> bool:
        """\
        Test if the given y coordinate is within the rectangle.
        """
        return self.bottom <= y <= self.top

    def union(self, other: BoundsRectangle) -> BoundsRectangle:
        """\
        Return a rectangle that is the union of this rectangle and other.
        """
        newLeft = min(self.left, other.left)
        newTop = max(self.top, other.top)
        newRight = max(self.right, other.right)
        newBottom = min(self.bottom, other.bottom)

        return BoundsRectangle((newLeft, newBottom), (newRight, newTop))

    def intersection(self, other: BoundsRectangle) -> typing.Optional[BoundsRectangle]:
        """\
        Return a rectangle that is the intersection of this rectangle and other.
        """
        newLeft = max(self.left, other.left)
        newTop = min(self.top, other.top)
        newRight = min(self.right, other.right)
        newBottom = max(self.bottom, other.bottom)

        if newRight < newLeft or newTop < newBottom: return None  # maybe want <=, >=?
        return BoundsRectangle((newLeft, newBottom), (newRight, newTop))

    def relationTo(self, other: BoundsRectangle) -> int:
        if self.encloses(other):
            return self.relationEncloses
        elif other.encloses(self):
            return self.relationEnclosed
        elif self.intersection(other):
            return self.relationIntersects
        else:
            return self.relationSeparate

def pointXY(point: Point):
    if isinstance(point, complex):
        return point.real, point.imag
    return point

def minMax(a: typing.Any, b: typing.Any) -> tuple[typing.Any, typing.Any]:
    """\
    Return a tuple with the min value first, then the max value.
    """
    return (a, b) if a <= b else (b, a)

def endPoints(segment: Segment) -> tuple[float, float, float, float]:
    """\
    Return the x, y coordinates of the start and end of the segment.
    """
    p0x, p0y = pointXY(segment[0])
    p1x, p1y = pointXY(segment[-1])

    return (p0x, p0y, p1x, p1y)

def getDeltas(segment: Segment) -> tuple[float, float]:
    """\
    Return the x, y span of the segment.
    """
    p0x, p0y, p1x, p1y = endPoints(segment)

    return (p1x - p0x, p1y - p0y)

def isVerticalLine(segment: Segment) -> bool:
    """\
    Test if the segment is a vertical line.
    """
    dx, _ = getDeltas(segment)
    return len(segment) == 2 and dx == 0

def isHorizontalLine(segment: Segment) -> bool:
    """\
    Test if the segment is a horizontal line.
    """
    _, dy = getDeltas(segment)
    return len(segment) == 2 and dy == 0

def isDiagonalLine(segment: Segment) -> bool:
    dx, dy = getDeltas(segment)
    return len(segment) == 2 and dx !=0 and dy != 0

def length(segment: Segment) -> float:
    """\
    Return the length of the segment. Only really makes sense for a line...
    """
    return math.hypot(*getDeltas(segment))

def slope(segment: Segment) -> float:
    """\
    Return the slope of the segment. rise / run. Returns
    math.inf if the line is vertical.
    """
    dx, dy = getDeltas(segment)

    if dx == 0: return math.inf
    return dy / dx

def slopeAngle(segment: Segment) -> float:
    """\
    Return the angle of the segment from vertical, in degrees.
    """
    dx, dy = getDeltas(segment)
    return math.degrees(math.atan2(abs(dx), abs(dy)))

# def lineSlopeAngle(line):
#     delta = line.end - line.start
#     return math.degrees(math.atan2(abs(delta.real), abs(delta.imag)))

def rawSlopeAngle(segment: Segment) -> float:
    dx, dy = getDeltas(segment)
    return math.degrees(math.atan2(dy, dx))

def midpoint(line: Segment) -> Point:
    """\
    Return the midpoint of the line.
    """
    p0x, p0y, p1x, p1y = endPoints(line)

    return ((p0x + p1x) / 2, (p0y + p1y) / 2)

def intersectionPoint(l1: typing.Sequence[Point], l2: typing.Sequence[Point]) -> typing.Optional[Point]:
    """\
    Find the intersection point of the two lines.
    """
    intersection = lli(l1, l2)

    if intersection is None: return None

    b1 = BoundsRectangle(*l1)
    b2 = BoundsRectangle(*l2)

    # The point calculated above assumes that the two lines have
    # infinite length, so it may not be on both, or either line.
    # Make sure it's within the bounds rectangle for both lines.
    return intersection if b1.enclosesPoint(intersection) and b2.enclosesPoint(intersection) else None

# The result of this function cannot be used to create an SVG path...
def flatten(contours: list[Contour]) -> Contour:
    """\
    Return a single contour that contains all the points in the given contours.
    """
    return [segment for contour in contours for segment in contour]

# There must be a better way to do this...
def pointOnLine(point: Point, line: typing.Sequence[Point]):
    """\
    Test if a given point is on the given line.
    """
    bounds = BoundsRectangle(*line)

    # If the bounds rectangle of the line encloses the point and
    # a line from the start of the given line to the point has the
    # same slope as the line, it is on the line.
    return bounds.enclosesPoint(point) and slope(line) == slope([line[0], point])


def rotatePointAbout(point: Point, about: Point, degrees: float = 90, ccw: bool = True):
    """\
    Rotate the given point the given number of degrees about the point about
    in a clockwise or counter-clockwise direction.
    """
    rt = Transform.rotationAbout(about, degrees, ccw)

    return rt.applyToPoint(point)

def rotateSegmentAbout(segment: Segment, about: Point, degrees: float = 90, ccw: bool = True):
    """\
    Rotate the given segment the given number of degrees about the point about
    in a clockwise or counter-clockwise direction.
    """
    rt = Transform.rotationAbout(about, degrees, ccw)

    return rt.applyToSegment(segment)

def rotateContourAbout(contour: Contour, about: Point, degrees: float = 90, ccw: bool = True):
    """\
    Rotate the given contour the given number of degrees about the point about
    in a clockwise or counter-clockwise direction.
    """
    rt = Transform.rotationAbout(about, degrees, ccw)

    return rt.applyToContour(contour)

def rotateContoursAbout(contours: list[Contour], about: Point, degrees: float = 90, ccw: bool = True):
    """\
    Rotate the given contours the given number of degrees about the point about
    in a clockwise or counter-clockwise direction.
    """
    rt = Transform.rotationAbout(about, degrees, ccw)

    return rt.applyToContours(contours)

def toMicros(funits: float, unitsPerEM: int):
    """\
    Convert funits into micros.
    """
    return funits * 1000 / unitsPerEM
