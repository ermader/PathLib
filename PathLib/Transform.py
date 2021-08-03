"""\
Transform.py

Extracted from PathUtilities.py on August 3, 2021

@author Eric Mader
"""

import math


class Transform(object):
    """\
    A 3x3 transform.
    """
    @staticmethod
    def multiplyRowByMatrix(row, matrix):
        """\
        Multiply the given row by the given matrix. Returns a new row.
        """
        r1, r2, r3 = row
        m1, m2, m3 = matrix
        m11, m12, m13 = m1
        m21, m22, m23 = m2
        m31, m32, m33 = m3

        return [r1 * m11 + r2 * m21 + r3 * m31, r1 * m12 + r2 * m22 + r3 * m32, r1 * m13 + r2 * m23 + r3 * m33]

    @staticmethod
    def multiplyMatrixByMatrix(m1, m2):
        """\
        Multiply the two matricies.
        """
        result = []
        for row in m1:
            result.append(Transform.multiplyRowByMatrix(row, m2))

        return result

    @staticmethod
    def concatenateMatrices(*matrices):
        """\
        Multiply the given matrices together.
        """
        concatenation = matrices[0]
        for matrix in matrices[1:]:
            concatenation = Transform.multiplyMatrixByMatrix(concatenation, matrix)

        return concatenation

    @staticmethod
    def sin(degrees):
        """\
        Return the sin of the angle.
        """
        # We use round() because sin values that should be zero
        # are actually around 1e-16
        return round(math.sin(math.radians(degrees)), 15)

    @staticmethod
    def cos(degrees):
        """\
        Return the cos of the angle.
        """
        # We use round() because cos values that should be zero
        # are actually around 1e-16
        return round(math.cos(math.radians(degrees)), 15)

    def __init__(self, *matrices):
        """\
        Construct a Transform by concateniing the given matrices.
        """
        self._transform = Transform.concatenateMatrices(*matrices)

    @staticmethod
    def _matrix(a=1.0, b=0.0, c=0.0, d=1.0, m=0.0, n=0.0, p=0.0, q=0.0, s=1.0):
        """\
        Construct a 3x3 matrix from the given values.
        """
        return [
            [a, b, p],
            [c, d, q],
            [m, n, s]]

    @staticmethod
    def _identityMatrix():
        """\
        Construct the identity matrix. This is the default for _matrix().
        """
        return Transform._matrix()

    @staticmethod
    def _scaleMatrix(sx=1, sy=1):
        """\
        Construct a matrix that scales in the x, y directions by the given factor.
        """
        return Transform._matrix(a=sx, d=sy)

    @staticmethod
    def _translateMatrix(fromPoint, toPoint):
        """\
        Construct a matrix that translates from fromPoint to toPoint.
        """
        fpx, fpy = fromPoint
        tpx, tpy = toPoint
        tx = tpx - fpx
        ty = tpy - fpy

        return Transform._matrix(m=tx, n=ty)

    @staticmethod
    def _shearMatrix(sx=0, sy=0):
        """\
        Construct a matrix that shears in the x, y directions by the given amounts
        """
        return Transform._matrix(b=sy, c=sx)

    @staticmethod
    def _mirrorMatrix(xAxis=False, yAxis=False):
        """\
        Construct a matrix that mirrors around the x and or y axes.
        """
        a = -1 if yAxis else 1
        d = -1 if xAxis else 1
        return Transform._matrix(a=a, d=d)

    @staticmethod
    def _rotationMatrix(degrees, ccw=True):
        """\
        Construct a matrix that rotates by the specified number of degrees
        in a clockwise or counter-clockwise direction.
        """
        st = Transform.sin(degrees)  # sin(theta)
        ct = Transform.cos(degrees)  # cos(theta)

        return Transform._matrix(a=ct, b=st, c=-st, d=ct) if ccw else Transform._matrix(a=ct, b=-st, c=st, d=ct)

    @staticmethod
    def _perspectiveMatrix(p, q, s=1):
        """\
        Construct a matrix that does a perspective transformation.
        """
        return Transform._matrix(p=p, q=q, s=s)

    @property
    def transform(self):
        """\
        Return the transform's matrix.
        """
        return self._transform

    @classmethod
    def translate(cls, fromPoint, toPoint):
        """\
        Construct a Transform object that translates from fromPoint to toPoint.
        """
        m = Transform._translateMatrix(fromPoint, toPoint)
        return Transform(m)

    @classmethod
    def scale(cls,sx=1, sy=1):
        """\
        Construct a Transform object that scales in the x, y directions by the given factor.
        """
        m = Transform._scaleMatrix(sx, sy)
        return Transform(m)

    @classmethod
    def shear(cls, sx=0, sy=0):
        """\
        Construct a Transform object that shears in the x, y directions by the given amounts
        """
        m = Transform._shearMatrix(sx, sy)
        return Transform(m)

    @classmethod
    def mirror(cls, xAxis=False, yAxis=False):
        """\
        Construct a Transform object that mirrors around the x and or y axes.
        """
        m = Transform._mirrorMatrix(xAxis, yAxis)
        return Transform(m)

    @classmethod
    def rotation(cls, degrees=90, ccw=True):
        """\
        Construct a Transform object that rotates by the specified number of degrees
        in a clockwise or counter-clockwise direction.
        """
        m = Transform._rotationMatrix(degrees, ccw)
        return Transform(m)

    @classmethod
    def perspective(cls, p=0, q=0, s=1):
        """\
        Construct a Transform object that does a perspective transformation.
        """
        m = Transform._perspectiveMatrix(p, q, s)
        return Transform(m)

    @classmethod
    def moveAndRotate(cls, fromPoint, toPoint, degrees):
        m1 = Transform._translateMatrix(fromPoint, toPoint)
        m2 = Transform._rotationMatrix(degrees)
        return Transform(m1, m2)

    @classmethod
    def rotateAndMove(cls, fromPoint, toPoint, degrees):
        m1 = Transform._rotationMatrix(degrees)
        m2 = Transform._translateMatrix(fromPoint, toPoint)
        return Transform(m1, m2)

    @classmethod
    def rotationAbout(cls, about, degrees=90, ccw=True):
        """\
        Construct a Transform object that rotates around the point about by the specified number of degrees
        in a clockwise or counter-clockwise direction.
        """
        origin = (0, 0)
        # Translate about point to origin
        m1 = Transform._translateMatrix(about, origin)

        # rotate
        m2 = Transform._rotationMatrix(degrees, ccw)

        # translate back to about point
        m3 = Transform._translateMatrix(origin, about)

        return Transform(m1, m2, m3)

    @classmethod
    def mirrorAround(cls, centerPoint, xAxis=False, yAxis=False):
        """\
        Construct a Transform object that mirrors around the given center point
        in the x and or y directions.
        """
        tx = ty = 0
        cx, cy = centerPoint

        if xAxis:
            ty = cy

        if yAxis:
            tx = cx

        mirrorPoint = (cx - tx, cy - ty)
        m1 = Transform._translateMatrix(centerPoint, mirrorPoint)
        m2 = Transform._mirrorMatrix(xAxis, yAxis)
        m3 = Transform._translateMatrix(mirrorPoint, centerPoint)

        return Transform(m1, m2, m3)

    @classmethod
    def perspectiveFrom(cls, centerPoint, p=0, q=0):
        """\
        Construct a Transform object that does a perspective transformation
        around the given center point.
        """
        origin = (0, 0)

        # translate centerPoint to the origin
        m1 = Transform._translateMatrix(centerPoint, origin)

        # the perspective transformation
        m2 = Transform._perspectiveMatrix(p, q)

        # translate back to centerPoint
        m3 = Transform._translateMatrix(origin, centerPoint)

        return Transform(m1, m2, m3)

    def applyToPoint(self, point):
        """\
        Apply the transformation to the given point.
        """

        complexPoint = isinstance(point, complex)

        if complexPoint:
            px = point.real
            py = point.imag
        else:
            px, py = point
        rp = Transform.multiplyRowByMatrix([px, py, 1], self.transform)

        # in the general case, rp[2] may not be 1, so
        # normalize to 1.
        rx = rp[0]/rp[2]
        ry = rp[1]/rp[2]

        return complex(rx, ry) if complexPoint else (rx, ry)

    def applyToSegment(self, segment):
        """\
        Apply the transform to all points in the given segment.
        """
        transformed = []
        for point in segment:
            transformed.append(self.applyToPoint(point))

        return transformed


    def applyToContour(self, contour):
        """\
        Apply the transform to all segments in the given contour.
        """
        transformed = []
        for segment in contour:
            transformed.append(self.applyToSegment(segment))

        return transformed


    def applyToContours(self, contours):
        """\
        Apply the transform to each contour in contours.
        """
        transformed = []
        for contour in contours:
            transformed.append(self.applyToContour(contour))

        return transformed