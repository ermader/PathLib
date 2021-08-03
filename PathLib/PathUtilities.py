"""\
Utilities for manipulating outline paths and segments

Created on July 7, 2020

@author Eric Mader
"""

import math, random
from colorsys import hls_to_rgb
from FontDocTools.Color import Color
from .Transform import Transform

from .BezierUtilities import lli

class PUColor(Color):
    """\
    A subclass of FontDocTools.Color that adds the full set of X11 and SVG 1.0 colors
    """
    def __init__(self, red, green, blue):
        Color.__init__(self, red, green, blue)

    # X11 colors plus SVG 1.0 gray/grey variants
    # from https://www.w3.org/TR/css-color-3/#html4
    _keywords = {
        'aliceblue': (240, 248, 255),
        'antiquewhite': (250, 235, 215),
        'aqua': (0, 255, 255),
        'aquamarine': (127, 255, 212),
        'azure': (240, 255, 255),
        'beige': (245, 245, 220),
        'bisque': (255, 228, 196),
        'black': (0, 0, 0),
        'blanchedalmond': (255, 235, 205),
        'blue': (0, 0, 255),
        'blueviolet': (138, 43, 226),
        'brown': (165, 42, 42),
        'burlywood': (222, 184, 135),
        'cadetblue': (95, 158, 160),
        'chartreuse': (127, 255, 0),
        'chocolate': (210, 105, 30),
        'coral': (255, 127, 80),
        'cornflowerblue': (100, 149, 237),
        'cornsilk': (255, 248, 220),
        'crimson': (220, 20, 60),
        'cyan': (0, 255, 255),
        'darkblue': (0, 0, 139),
        'darkcyan': (0, 139, 139),
        'darkgoldenrod': (184, 134, 11),
        'darkgray': (169, 169, 169),
        'darkgreen': (0, 100, 0),
        'darkgrey': (169, 169, 169),
        'darkkhaki': (189, 183, 107),
        'darkmagenta': (139, 0, 139),
        'darkolivegreen': (85, 107, 47),
        'darkorange': (255, 140, 0),
        'darkorchid': (153, 50, 204),
        'darkred': (139, 0, 0),
        'darksalmon': (233, 150, 122),
        'darkseagreen': (143, 188, 143),
        'darkslateblue': (72, 61, 139),
        'darkslategray': (47, 79, 79),
        'darkslategrey': (47, 79, 79),
        'darkturquoise': (0, 206, 209),
        'darkviolet': (148, 0, 211),
        'deeppink': (255, 20, 147),
        'deepskyblue': (0, 191, 255),
        'dimgray': (105, 105, 105),
        'dimgrey': (105, 105, 105),
        'dodgerblue': (30, 144, 255),
        'firebrick': (178, 34, 34),
        'floralwhite': (255, 250, 240),
        'forestgreen': (34, 139, 34),
        'fuchsia': (255, 0, 255),
        'gainsboro': (220, 220, 220),
        'ghostwhite': (248, 248, 255),
        'gold': (255, 215, 0),
        'goldenrod': (218, 165, 32),
        'gray': (128, 128, 128),
        'green': (0, 128, 0),
        'greenyellow': (173, 255, 47),
        'grey': (128, 128, 128),
        'honeydew': (240, 255, 240),
        'hotpink': (255, 105, 180),
        'indianred': (205, 92, 92),
        'indigo': (75, 0, 130),
        'ivory': (255, 255, 240),
        'khaki': (240, 230, 140),
        'lavender': (230, 230, 250),
        'lavenderblush': (255, 240, 245),
        'lawngreen': (124, 252, 0),
        'lemonchiffon': (255, 250, 205),
        'lightblue': (173, 216, 230),
        'lightcoral': (240, 128, 128),
        'lightcyan': (224, 255, 255),
        'lightgoldenrodyellow': (250, 250, 210),
        'lightgray': (211, 211, 211),
        'lightgreen': (144, 238, 144),
        'lightgrey': (211, 211, 211),
        'lightpink': (255, 182, 193),
        'lightsalmon': (255, 160, 122),
        'lightseagreen': (32, 178, 170),
        'lightskyblue': (135, 206, 250),
        'lightslategray': (119, 136, 153),
        'lightslategrey': (119, 136, 153),
        'lightsteelblue': (176, 196, 222),
        'lightyellow': (255, 255, 224),
        'lime': (0, 255, 0),
        'limegreen': (50, 205, 50),
        'linen': (250, 240, 230),
        'magenta': (255, 0, 255),
        'maroon': (128, 0, 0),
        'mediumaquamarine': (102, 205, 170),
        'mediumblue': (0, 0, 205),
        'mediumorchid': (186, 85, 211),
        'mediumpurple': (147, 112, 219),
        'mediumseagreen': (60, 179, 113),
        'mediumslateblue': (123, 104, 238),
        'mediumspringgreen': (0, 250, 154),
        'mediumturquoise': (72, 209, 204),
        'mediumvioletred': (199, 21, 133),
        'midnightblue': (25, 25, 112),
        'mintcream': (245, 255, 250),
        'mistyrose': (255, 228, 225),
        'moccasin': (255, 228, 181),
        'navajowhite': (255, 222, 173),
        'navy': (0, 0, 128),
        'oldlace': (253, 245, 230),
        'olive': (128, 128, 0),
        'olivedrab': (107, 142, 35),
        'orange': (255, 165, 0),
        'orangered': (255, 69, 0),
        'orchid': (218, 112, 214),
        'palegoldenrod': (238, 232, 170),
        'palegreen': (152, 251, 152),
        'paleturquoise': (175, 238, 238),
        'palevioletred': (219, 112, 147),
        'papayawhip': (255, 239, 213),
        'peachpuff': (255, 218, 185),
        'peru': (205, 133, 63),
        'pink': (255, 192, 203),
        'plum': (221, 160, 221),
        'powderblue': (176, 224, 230),
        'purple': (128, 0, 128),
        'red': (255, 0, 0),
        'rosybrown': (188, 143, 143),
        'royalblue': (65, 105, 225),
        'saddlebrown': (139, 69, 19),
        'salmon': (250, 128, 114),
        'sandybrown': (244, 164, 96),
        'seagreen': (46, 139, 87),
        'seashell': (255, 245, 238),
        'sienna': (160, 82, 45),
        'silver': (192, 192, 192),
        'skyblue': (135, 206, 235),
        'slateblue': (106, 90, 205),
        'slategray': (112, 128, 144),
        'slategrey': (112, 128, 144),
        'snow': (255, 250, 250),
        'springgreen': (0, 255, 127),
        'steelblue': (70, 130, 180),
        'tan': (210, 180, 140),
        'teal': (0, 128, 128),
        'thistle': (216, 191, 216),
        'tomato': (255, 99, 71),
        'turquoise': (64, 224, 208),
        'violet': (238, 130, 238),
        'wheat': (245, 222, 179),
        'white': (255, 255, 255),
        'whitesmoke': (245, 245, 245),
        'yellow': (255, 255, 0),
        'yellowgreen': (154, 205, 50)
    }

    @classmethod
    def _forKeyword(cls, color):
        """\
         Return a new Color object for the given color keyword.
         Return None if the given string doesn’t consist of Unicode “Letter” characters.
        """
        if not color.isalpha():
            return None
        color = color.lower()
        if color in PUColor._keywords:
            (red, green, blue) = PUColor._keywords[color]
            return PUColor(red, green, blue)
        raise ValueError(f"Unrecognized color keyword: {color}")

    @classmethod
    def fromName(cls, name):
        """\
        Return a color given the name.
        """
        return cls._forKeyword(name)

    @classmethod
    def randomRGBColor(cls):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        return PUColor(red, green, blue)

    CURRENT_HUE = 0

    @classmethod
    def setCurrentHue(cls, hue=0):
        cls.CURRENT_HUE = hue

    # Based on randomColor() from graphics-api.js from
    # from https://github.com/Pomax/BezierInfo-2
    # (not really random...)
    @classmethod
    def randomHSLColor(cls):
        cls.CURRENT_HUE = (cls.CURRENT_HUE + 73) % 360
        hue = cls.CURRENT_HUE
        saturation = 50
        lightness = 50
        red, green, blue = hls_to_rgb(hue / 360, lightness / 100, saturation / 100)
        return PUColor(round(red * 255), round(green * 255), round(blue * 255))

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
