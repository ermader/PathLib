"""\
A bézier curve object

Created on August 30, 2020

Much of this code translated from bezier.js, utils.js and others
from https://github.com/Pomax/BezierInfo-2

@author Eric Mader
"""

from __future__ import annotations
import typing

import math
import types
from decimal import Decimal, getcontext

from PathLib.PathTypes import Contour, Segment, Point

from . import BezierUtilities as butils
from . import PathUtilities
from .Transform import Transform

MAX_SAFE_INTEGER = +9007199254740991  # Number.MAX_SAFE_INTEGER
MIN_SAFE_INTEGER = -9007199254740991  # Number.MIN_SAFE_INTEGER

BBox = dict[int, tuple[float, float]]
Extrema = dict[int, list[float]]


class Bezier(object):
    dir_mixed = -1
    dir_flat = 0
    dir_up = 1
    dir_down = 2

    __slots__ = "_controlPoints", "_t1", "_t2", "_dcPoints", "_extrema", "_length", "_bbox", "_boundsRectangle", \
                "_lut", "_direction", "_linear", "_clockwise", "_start", "_end"

    def __init__(self, controlPoints: typing.Sequence[Point]):
        self._controlPoints = controlPoints
        self._t1 = 0
        self._t2 = 1
        self._dcPoints: typing.Optional[list[list[Point]]] = None
        self._extrema: typing.Optional[Extrema] = None
        self._length = None
        self._bbox: typing.Optional[BBox] = None
        self._boundsRectangle = None
        self._lut: list[Point] = []

        self._direction = self._computeDirection()

        # a bit of a hack to deal with the fact that control points
        # are sometimes Decimal values...
        def fp(p: Point): return (float(p[0]), float(p[1]))

        cp = [fp(p) for p in self.controlPoints]
        aligned = self._align(cp, [cp[0], cp[-1]])
        self._linear = len([x for x in aligned if abs(x[1]) > 0.0001]) == 0

        angle = butils.angle(cp[0], cp[-1], cp[1])
        self._clockwise = angle > 0

    def _computeDirection(self) -> int:
        _p0x, p0y = self._controlPoints[0]
        _p1x, p1y = self._controlPoints[1]

        if self.order == 1:
            if p0y == p1y:
                return Bezier.dir_flat

            if p0y < p1y:
                return Bezier.dir_up

            return Bezier.dir_down

        _p2x, p2y = self._controlPoints[2]
        if self.order == 2:
            if p0y <= p1y <= p2y:
                return Bezier.dir_up
            if p0y >= p1y >= p2y:
                return Bezier.dir_down

            # we assume that a quadratic bezier won't be flat...
            return Bezier.dir_mixed

        _p3x, p3y = self._controlPoints[3]
        if self.order == 3:
            if p0y <= p1y <= p2y <= p3y:
                return Bezier.dir_up
            if p0y >= p1y >= p2y >= p3y:
                return Bezier.dir_down

            # we assume that a cubic bezier won't be flat...
            return Bezier.dir_mixed

        # For now, just say higher-order curves are mixed...
        return Bezier.dir_mixed

    def _compute(self, t: float):
        # shortcuts...
        if t == 0: return self._controlPoints[0]

        if t == 1: return self._controlPoints[self.order]

        mt = 1 - t
        p = self._controlPoints

        # constant?
        if self.order == 0: return self._controlPoints[0]

        # linear?
        if self.order == 1:
            p0x, p0y = p[0]
            p1x, p1y = p[1]
            return (mt * p0x + t * p1x, mt * p0y + t * p1y)

        # quadratic / cubic?
        if self.order < 4:
            mt2 = mt * mt
            t2 = t * t

            
            if self.order == 2:
                p = [p[0], p[1], p[2], (0, 0)]
                a = mt2
                b = mt * t * 2
                c = t2
                d = 0
            else:  # self.order == 3:
                a = mt2 * mt
                b = mt2 * t * 3
                c = mt * t2 * 3
                d = t * t2

            p0x, p0y = p[0]
            p1x, p1y = p[1]
            p2x, p2y = p[2]
            p3x, p3y = p[3]
            rx = a * p0x + b * p1x + c * p2x + d * p3x
            ry = a * p0y + b * p1y + c * p2y + d * p3y
            return (rx, ry)
        else:
            # higher order curves: use de Casteljau's computation
            #   JavaScript code does this:
            #     const dCpts = JSON.parse(JSON.stringify(points));
            #   This is a copy operation...
            dcPoints: list[Point] = [pt for pt in p]
            while len(dcPoints) > 1:
                newPoints: list[Point] = []
                for i in range(len(dcPoints) - 1):
                    x0, y0 = dcPoints[i]
                    x1, y1 = dcPoints[i + 1]
                    nx = x0 + (x1 - x0) * t
                    ny = y0 + (y1 - y0) * t
                    newPoints.append((nx, ny))
                dcPoints = newPoints
            return dcPoints[0]

    # DecimalOrFloat = typing.TypeVar("DecimalOrFloat", float, Decimal)

    @typing.overload
    def _derivative(self, t: float) -> tuple[float, float]: ...

    @typing.overload
    def _derivative(self, t: Decimal) -> tuple[Decimal, Decimal]: ...

    #
    # We really want to use DecimalOrFloat instead of Any, but
    # type checking thinks all values can always be either, even though
    # to code flow guarantees that they're all the same type.
    #
    def _derivative(self, t: typing.Any) -> tuple[typing.Any, typing.Any]:
        p = self.dcPoints[0]
        mt = 1 - t

        if self.order == 2:
            p = [p[0], p[1], (0, 0)]
            a = mt
            b = t
            c = 0
        else:  # self.order == 3:
            a = mt * mt
            b = mt * t * 2
            c = t * t

        p0x, p0y = p[0]
        p1x, p1y = p[1]
        p2x, p2y = p[2]

        # if t is Decimal, convert the x, y coordinates to Decimal
        if isinstance(t, Decimal):
            p0x = Decimal(p0x)
            p0y = Decimal(p0y)
            p1x = Decimal(p1x)
            p1y = Decimal(p1y)
            p2x = Decimal(p2x)
            p2y = Decimal(p2y)

        rx = a * p0x + b * p1x + c * p2x
        ry = a * p0y + b * p1y + c * p2y
        return (rx, ry)

    def _tangent(self, t: float) -> Point:
        d = self._derivative(t)
        q = math.hypot(d[0], d[1])
        return (d[0] / q, d[1] / q)

    def _normal(self, t: float) -> Point:
        d = self._derivative(t)
        q = math.hypot(d[0], d[1])
        return (-d[1] / q, d[0] / q)

    def _getminmax(self, d: int, list: list[float]) -> tuple[float, float]:
        # if (!list) return { min: 0, max: 0 };
        min = +9007199254740991  # Number.MAX_SAFE_INTEGER
        max = -9007199254740991  # Number.MIN_SAFE_INTEGER

        if 0 not in list: list.insert(0, 0)
        if 1 not in list: list.append(1)

        for i in range(len(list)):
            c = self.get(list[i])
            if c[d] < min: min = c[d]
            if c[d] > max: max = c[d]

        return (min, max)

    @property
    def start(self) -> Point:
        return self._controlPoints[0]

    @property
    def startX(self) -> float:
        return self.start[0]

    @property
    def startY(self) -> float:
        return self.start[1]

    @property
    def end(self) -> Point:
        return self._controlPoints[-1]

    @property
    def endX(self) -> float:
        return self.end[0]

    @property
    def endY(self) -> float:
        return self.end[1]

    @property
    def bbox(self) -> BBox:
        if not self._bbox:
            extrema = self.extrema
            result = {}

            for dim in range(2):
                result[dim] = self._getminmax(dim, extrema[dim])

            self._bbox = result

        return self._bbox

    @property
    def tightbbox(self) -> Contour:
        aligned = self.align()
        tBounds = aligned.boundsRectangle
        angle = PathUtilities.rawSlopeAngle(self.controlPoints)
        translate = Transform.rotateAndMove((0, 0), self.controlPoints[0], angle)
        tbContour = translate.applyToContour(tBounds.contour)
        return tbContour

    @property
    def boundsRectangle(self) -> PathUtilities.BoundsRectangle:
        if not self._boundsRectangle:
            bbox = self.bbox
            minX, maxX = bbox[0]
            minY, maxY = bbox[1]
            self._boundsRectangle = PathUtilities.BoundsRectangle((minX, minY), (maxX, maxY))

        return self._boundsRectangle

    @property
    def skeletonBounds(self):
        sbounds = PathUtilities.BoundsRectangle.fromContour([self.controlPoints])
        sbounds.right += 40
        return sbounds

    def get(self, t: float) -> Point:
        return self._compute(t)

    @staticmethod
    def _align(points: typing.Sequence[Point], segment: Segment) -> Segment:
        angle = PathUtilities.rawSlopeAngle(segment)
        transform = Transform.moveAndRotate(segment[0], (0, 0), -angle)
        return transform.applyToSegment(points)

    @property
    def controlPoints(self):
        return self._controlPoints

    @property
    def order(self):
        return len(self._controlPoints) - 1

    @property
    def direction(self):
        return self._direction

    @property
    def midpoint(self):
        return self.get(0.5)

    @property
    def dcPoints(self) -> list[list[Point]]:
        if not self._dcPoints:
            dpoints: list[list[Point]] = []
            p = self._controlPoints
            d = len(p)

            while d > 1:
                dpts: list[Point] = []
                c = d - 1
                for j in range(c):
                    x0, y0 = p[j]
                    x1, y1 = p[j+1]
                    dpts.append((c*(x1-x0), c*(y1-y0)))
                dpoints.append(dpts)
                p = dpts
                d -= 1

            self._dcPoints = dpoints

        return self._dcPoints

    @property
    def extrema(self) -> Extrema:
        if not self._extrema:
            result: Extrema = {}
            roots: list[float] = []

            for dim in range(2):
                p = list(map(lambda p: p[dim], self.dcPoints[0]))
                result[dim] = list(butils.droots(p))
                if self.order == 3:
                    p = list(map(lambda p: p[dim], self.dcPoints[1]))
                    result[dim].extend(butils.droots(p))
                result[dim] = list(filter(lambda t: t >= 0 and t <= 1, result[dim]))
                roots.extend(sorted(result[dim]))

            result[2] = butils.removeDuplicates(sorted(roots))

            self._extrema = result

        return self._extrema

    def overlaps(self, curve: Bezier):
        return self.boundsRectangle.intersection(curve.boundsRectangle) is not None

    def hull(self, t: float) -> list[Point]:
        p: typing.Sequence[Point] = self.controlPoints
        q = [pt for pt in p]

        # we lerp between all points at each iteration, until we have 1 point left.
        while len(p) > 1:
            _p: list[Point] = []
            for i in range(len(p) - 1):
                pt = butils.lerp(t, p[i], p[i + 1])
                q.append(pt)
                _p.append(pt)
            p = _p

        return q

    # LUT == LookUp Table
    def getLUT(self, steps: int = 100) -> list[Point]:
        if len(self._lut) == steps: return self._lut

        self._lut = []
        # We want a range from 0 to 1 inclusive, so
        # we decrement steps and use range(steps+1)
        steps -= 1
        for t in range(steps+1):
            self._lut.append(self.get(t / steps))

        return self._lut

    def _arcfun(self, t: Decimal) -> Decimal:
        dx, dy = self._derivative(t)

        # getcontext().prec += 2
        result = (dx * dx + dy * dy).sqrt()
        # getcontext().prec -= 2
        return result

    @classmethod
    def pointXY(cls, point: Point) -> tuple[float, float]:
        return point

    @classmethod
    def xyPoint(cls, x: float, y: float) -> Point:
        return x, y

    @property
    def length(self):
        if not self._length:
            z = Decimal(0.5)
            sum = Decimal(0)

            getcontext().prec += 2
            for i in range(len(butils.tValues)):
                t = butils.tValues[i].fma(z, z)
                sum = butils.cValues[i].fma(self._arcfun(t), sum)

            length = z * sum
            getcontext().prec -= 2
            self._length = +length

        return self._length

    @typing.overload
    def split(self, t1: float) -> tuple[Bezier, Bezier, list[Point]]: ...

    @typing.overload
    def split(self, t1: float, t2: float) -> Bezier: ...

    def split(self, t1: float, t2: typing.Optional[float] = None) -> typing.Union[Bezier, tuple[Bezier, Bezier, list[Point]]]:
        # shortcuts...
        if t1 == 0 and t2: return self.split(t2)[0]
        if t2 == 1: return self.split(t1)[1]

        # no shortcut: use "de Casteljau" iteration.
        q = self.hull(t1)
        if self.order == 2:
            left = Bezier([q[0], q[3], q[5]])
            right = Bezier([q[5], q[4], q[2]])
        else:
            left = Bezier([q[0], q[4], q[7], q[9]])
            right = Bezier([q[9], q[8], q[6], q[3]])

        # make sure we bind _t1/_t2 information!
        left._t1 = butils.map(0, 0, 1, self._t1, self._t2)
        left._t2 = butils.map(t1, 0, 1, self._t1, self._t2)
        right._t1 = butils.map(t1, 0, 1, self._t1, self._t2)
        right._t2 = butils.map(1, 0, 1, self._t1, self._t2)

        # if we have no t2, we're done
        if not t2: return (left, right, q)

        t2 = butils.map(t2, t1, 1, 0, 1)
        return right.split(t2)[0]

    def roots(self, segment: typing.Optional[Segment] = None) -> list[float]:

        if segment:
            p = Bezier._align(self.controlPoints, segment)
        else:
            p = self.controlPoints

        def reduce(t: float):
            return 0 <= t <= 1 or butils.approximately(t, 0) or butils.approximately(t, 1)

        order = len(p) - 1
        if order == 2:
            a = p[0][1]
            b = p[1][1]
            c = p[2][1]
            d = a - 2 * b + c
            if d != 0:
                m1 = -butils.sqrt(b * b - a * c)
                m2 = -a + b
                v1 = -(m1 + m2) / d
                v2 = -(-m1 + m2) / d
                return list(filter(reduce, [v1, v2]))
            elif b != c and d == 0:
                return list(filter(reduce, [(2 * b - c) / (2 * b - 2 * c)]))

            return []

        # see http://www.trans4mind.com/personal_development/mathematics/polynomials/cubicAlgebra.htm
        pa = p[0][1]
        pb = p[1][1]
        pc = p[2][1]
        pd = p[3][1]

        d = -pa + 3 * pb - 3 * pc + pd
        a = 3 * pa - 6 * pb + 3 * pc
        b = -3 * pa + 3 * pb
        c = pa

        if butils.approximately(d, 0):
            # this is not a cubic curve.
            if butils.approximately(a, 0):
                # in fact, this is not a quadratic curve either.
                if butils.approximately(b, 0):
                    # in fact, there are no solutions
                    return []

                # linear solution:
                return list(filter(reduce, [-c / b]))

            # quadratic solution:
            q = butils.sqrt(b * b - 4 * a * c)
            a2 = 2 * a
            return list(filter(reduce, [(q - b) / a2, (-b - q) / a2]))

        # at this point, we know we need a cubic solution:
        a /= d
        b /= d
        c /= d

        p = (3 * b - a * a) / 3
        p3 = p / 3
        q = (2 * a * a * a - 9 * a * b + 27 * c) / 27
        q2 = q / 2
        discriminant = q2 * q2 + p3 * p3 * p3

        if discriminant < 0:
            mp3 = -p / 3
            mp33 = mp3 * mp3 * mp3
            r = butils.sqrt(mp33)
            t = -q / (2 * r)
            # cosphi = t < -1 ? -1: t > 1 ? 1: t
            cosphi = -1 if t < -1 else 1 if t > 1 else t
            phi = math.acos(cosphi)
            crtr = butils.crt(r)
            t1 = 2 * crtr
            x1 = t1 * math.cos(phi / 3) - a / 3
            x2 = t1 * math.cos((phi + butils.tau) / 3) - a / 3
            x3 = t1 * math.cos((phi + 2 * butils.tau) / 3) - a / 3
            return list(filter(reduce, [x1, x2, x3]))
        elif butils.approximately(discriminant, 0):  # discriminant == 0:
            u1 = butils.crt(-q2) if q2 < 0 else -butils.crt(q2)
            x1 = 2 * u1 - a / 3
            x2 = -u1 - a / 3
            return list(filter(reduce, [x1, x2]))
        else:
            sd = butils.sqrt(discriminant)
            u1 = butils.crt(-q2 + sd)
            v1 = butils.crt(q2 + sd)
            return list(filter(reduce, [u1 - v1 - a / 3]))

    def align(self, segment: typing.Optional[Segment] = None):
        if not segment:
            segment = self.controlPoints

        return Bezier(Bezier._align(self.controlPoints, segment))

    def normal(self, t: float) -> tuple[float, float]:
        dx, dy = self._derivative(t)
        q = butils.sqrt(dx * dx + dy * dy)
        return (-dy / q, dx / q)


    def simple(self):
        if self.order == 3:
            a1 = butils.angle(self.controlPoints[0], self.controlPoints[3], self.controlPoints[1])
            a2 = butils.angle(self.controlPoints[0], self.controlPoints[3], self.controlPoints[2])
            if (a1 > 0 and a2 < 0) or (a1 < 0 and a2 > 0): return False

        n1x, n1y = self.normal(0)
        n2x, n2y = self.normal(1)
        s = n1x * n2x + n1y * n2y
        return abs(math.acos(s)) < math.pi / 3

    def reduce(self) -> list[Bezier]:
        pass1: list[Bezier] = []
        pass2: list[Bezier] = []

        # first pass: split on extrema
        extrema = self.extrema[2]
        if not 0 in extrema: extrema.insert(0, 0)
        if not 1 in extrema: extrema.append(1)

        t1 = extrema[0]
        for i in range(1, len(extrema)):
            t2 = extrema[i]
            segment = self.split(t1, t2)
            segment._t1 = t1
            segment._t2 = t2
            pass1.append(segment)
            t1 = t2

        # second pass: further reduce these segments to simple segments
        step = 0.01
        for p1 in pass1:
            t1 = 0
            t2 = 0
            while t2 <= 1:
                t2 = t1 + step
                while t2 <= 1:
                    segment = p1.split(t1, t2)
                    if not segment.simple():
                        t2 -= step
                        if abs(t1 - t2) < step:
                            # we can never form a reduction
                            return []
                        segment = p1.split(t1, t2)
                        segment._t1 = butils.map(t1, 0, 1, p1._t1, p1._t2)
                        segment._t2 = butils.map(t2, 0, 1, p1._t1, p1._t2)
                        pass2.append(segment)
                        t1 = t2
                        break
                    t2 += step

            if t1 < 1:
                segment = p1.split(t1, 1)
                segment._t1 = butils.map(t1, 0, 1, p1._t1, p1._t2)
                segment._t2 = p1._t2
                pass2.append(segment)

        return pass2

    def lineIntersects(self, line: tuple[Point, Point]):
        p1, p2 = line
        p1x, p1y = p1
        p2x, p2y = p2
        mx = min(p1x, p2x)
        my = min(p1y, p2y)
        MX = max(p1x, p2x)
        MY = max(p1y, p2y)

        def onLine(t: float) -> bool:
            x, y = self.get(t)
            return butils.between(x, mx, MX) and butils.between(y, my, MY)

        return list(filter(onLine, self.roots(line)))

    def intersectWithLine(self, line: Bezier):
        if self.order == 1:
            return butils.lli(self.controlPoints, line.controlPoints)

        roots = self.roots(line.controlPoints)

        if roots:
            root = 0 if roots[0] < 0 else 1 if roots[0] > 1 else roots[0]

            return self.get(root)

        return None

    @staticmethod
    def pairiteration(c1: Bezier, c2: Bezier, intersectionThreshold: float = 0.5) -> list[Point]:
        c1b = c1.boundsRectangle
        c2b = c2.boundsRectangle
        r = 100000

        if c1b.height + c1b.width < intersectionThreshold and c2b.height + c2b.width < intersectionThreshold:
            return [(((r * (c1._t1 + c1._t2)) / 2) / r, ((r * (c2._t1 + c2._t2)) / 2) / r)]

        cc1left, cc1right, _ = c1.split(0.5)
        cc2left, cc2right, _ = c2.split(0.5)
        pairs = [(cc1left, cc2left), (cc1left, cc2right), (cc1right, cc2right), (cc1right, cc2left)]
        pairs = list(filter(lambda pair: pair[0].overlaps(pair[1]), pairs))

        results: list[Point] = []
        if len(pairs) == 0: return results

        for pair in pairs:
            left, right = pair
            results.extend(Bezier.pairiteration(left, right, intersectionThreshold))

        return butils.removeDuplicates(results)

    @staticmethod
    def curveIntersects(c1: list[Bezier], c2: list[Bezier], intersectionThreshold: float = 0.5) -> list[Point]:
        pairs: list[tuple[Bezier, Bezier]] = []

        # step 1: pair off any overlapping segments
        for l in c1:
            for r in c2:
                if l.overlaps(r):
                    pairs.append((l, r))

        # step 2: for each pairing, run through the convergence algorithm.
        intersections: list[Point] = []
        for pair in pairs:
            result = Bezier.pairiteration(pair[0], pair[1], intersectionThreshold)
            if len(result) > 0:
                intersections.extend(result)

        return intersections

    def selfIntersects(self, intersectionThreshold: float = 0.5) -> list[Point]:
        # "simple" curves cannot intersect with their direct
        # neighbor, so for each segment X we check whether
        # it intersects [0:x-2][x+2:last].
        reduced = self.reduce()
        length = len(reduced) - 2
        results: list[Point] = []

        for i in range(length):
            left = reduced[i:i+1]
            right = reduced[i+2:]
            result = Bezier.curveIntersects(left, right, intersectionThreshold)
            results.extend(result)

        return results

    def intersects(self, curve: typing.Optional[Bezier], intersectionThreshold: float = 0.5):
        if curve is None: return self.selfIntersects(intersectionThreshold)
        # if curve is a line: self.lineIntersects(line, intersectionThreshold)
        #if curve instanceOf Bezier: curve = curve.reduce()

        return Bezier.curveIntersects(self.reduce(), curve.reduce(), intersectionThreshold)

    def getABC(self, t: float) -> tuple[Point, Point, typing.Optional[Point], float, list[Point]]:
        hull = self.hull(t)
        points = self.controlPoints
        if self.order == 2:
            a = points[1]
            b = hull[5]
            ratio = butils.quadraticRatio(t)
        else:  # self.order == 3:
            a = hull[5]
            b = hull[9]
            ratio = butils.cubicRatio(t)

        c = butils.lli4(a, b, points[0], points[-1])

        return (a, b, c, ratio, hull)

    def raiseOrder(self):
        p = self.controlPoints
        k = len(p)
        np = [p[0]]
        for i in range(k):
            pix, piy = p[i]
            pimx, pimy = p[i - 1]
            x = ((k - i) / k) * pix + (i / k) * pimx
            y = ((k - i) / k) * piy + (i / k) * pimy
            np.append((x, y))
        np.append(p[-1])
        return Bezier(np)

    @typing.overload
    def offset(self, t: float) -> list[Bezier]: ...

    @typing.overload
    def offset(self, t: float, d: float) -> list[Point]: ...

    def offset(self, t: float, d: typing.Optional[float] = None) -> typing.Union[list[Point], list[Bezier]]:
        if d:
            cx, cy = c = self.get(t)
            nx, ny = n = self.normal(t)
            px = cx + nx * d
            py = cy + ny * d
            return [c, n, (px, py)]

        if self._linear:
            nv = self.normal(0)
            coords = list(map(lambda p: (p[0] + t * nv[0], p[1] + t * nv[1]), self.controlPoints))
            return [Bezier(coords)]

        reduced = self.reduce()
        return list(map(lambda r: r.offset(t)[0] if r._linear else r.scale(t), reduced))

    DistanceFunction = typing.Callable[[float], float]
    def scale(self, d: typing.Union[float, DistanceFunction]) -> Bezier:
        distanceFn: Bezier.DistanceFunction = d if isinstance(d, types.FunctionType) else lambda f: typing.cast(float, d)
        order = self.order

        if isinstance(d, types.FunctionType) and order == 2: return self.raiseOrder().scale(distanceFn)

        # TODO: add special handling for degenerate (=linear) curves.
        clockwise = self._clockwise
        r1 = distanceFn(0)
        r2 = distanceFn(1)

        v = [self.offset(0, 10), self.offset(1, 10)]
        np: list[Point] = [(0, 0) for _ in range(order+1)]
        o = butils.lli4(v[0][2], v[0][0], v[1][2], v[1][0])

        if not o:
            raise ValueError("Cannot scale this curve. Try reducing it first.")
        
        ox, oy = o

        # move all points by distance 'd' wrt the origin 'o'

        # move end points by fixed distance along normal.
        for t in range(2):
            px, py = self.controlPoints[t * order]
            px += (r2 if t > 0 else r1) * v[t][1][0]  # v[t].n.x
            py += (r2 if t > 0 else r1) * v[t][1][1]  # v[t].n.y
            np[t * order] = (px, py)

        if isinstance(d, types.FunctionType):
            # move control points to lie on the intersection of the offset
            # derivative vector, and the origin-through-control vector
            for t in range(2):
                if order == 2 and t > 0: break
                px, py = p = np[t * order]
                dx, dy = self._derivative(t)
                p2 = (px + dx, py + dy)
                np[t + 1] = typing.cast(Point, butils.lli4(p, p2, o, self.controlPoints[t + 1]))

            return Bezier(np)

        # move control points by "however much necessary to
        # ensure the correct tangent to endpoint".
        for t in range(2):
            if order == 2 and t > 0: break
            px, py = p = self.controlPoints[t + 1]
            ovx, ovy = (px - ox, py - oy)
            rc = distanceFn((t + 1) / order) if distanceFn else typing.cast(float, d)
            if distanceFn and not clockwise: rc = -rc
            m = math.hypot(ovx, ovy)
            ovx /= m
            ovy /= m
            np[t + 1] = (px + rc * ovx, py + rc * ovy)

        return Bezier(np)

class BContour(object):
    def __init__(self, contour: Contour):
        beziers: list[Bezier] = []
        bounds = PathUtilities.BoundsRectangle()

        for segment in contour:
            bezier = Bezier(segment)
            bounds = bounds.union(bezier.boundsRectangle)
            beziers.append(bezier)

        self._beziers = beziers
        self._bounds = bounds
        self._lut: list[Point] = []
        self._start = beziers[0].start
        self._end = beziers[-1].end

    def __getitem__(self, index: int) -> Bezier:
        return self._beziers[index]

    def __setitem__(self, index: int, value: typing.Sequence[Point]):
        self._beziers[index] = Bezier(value)
        # self._length = None
        self._start = self._beziers[0].start
        self._end = self._beziers[-1].end

    def __delitem__(self, index: int):
        del self._beziers[index]
        # self._length = None
        self._start = self._beziers[0].start
        self._end = self._beziers[-1].end

    def __iter__(self):
        return self._beziers.__iter__()

    def getLUT(self, steps: int = 100):
        if len(self._lut) == steps: return self._lut

        self._lut = []
        startPoint = 0
        for curve in self._beziers:
            self._lut.extend(curve.getLUT(steps)[startPoint:])
            # for every curve but the first, the first point is
            # the same as the last point of the last curve
            startPoint = 1

        return self._lut

    @classmethod
    def _findClosest(cls, point: Point, LUT: typing.Sequence[Point]) -> int:
        x, y = point
        closest = MAX_SAFE_INTEGER
        i = -1

        for index in range(len(LUT)):
            px, py = LUT[index]
            dist = math.hypot(px - x, py - y)
            if dist < closest:
                closest = dist
                i = index

        return i

    @classmethod
    def _refineBinary(cls, point: Point, curve: Bezier, lut: typing.Sequence[Point], i: int):
        closest = MAX_SAFE_INTEGER
        steps = len(lut)
        tt = [t / (steps - 1) for t in range(steps)]
        px, py = point
        q: Point = (0, 0)

        for _ in range(25):  # This is for safety; the loop should always break
            steps = len(lut)
            i1 = 0 if i == 0 else i - 1
            i2 = i if i == steps - 1 else i + 1
            t1 = tt[i1]
            t2 = tt[i2]
            new_lut: typing.Sequence[Point] = []
            new_tt: list[float] = []
            step = (t2 - t1) / 5

            if step < 0.001: break
            new_lut.append(lut[i1])
            new_tt.append(tt[i1])
            for j in range(1, 4):
                nt = t1 + (j * step)
                nx, ny = n = curve.get(nt)
                dist = math.hypot(nx - px, ny - py)
                if dist < closest:
                    closest = dist
                    q = n
                    i = j
                new_lut.append(n)
                new_tt.append(nt)
            new_lut.append(lut[i2])
            new_tt.append(tt[i2])

            # update the lut to be our new five point LUT, and run again.
            lut = new_lut
            tt = new_tt

        return (q, closest)

    def findClosestPoint(self, point: Point, steps: int):
        nCurves = len(self.beziers)
        curve2 = None
        i = self._findClosest(point, self.getLUT(steps))

        if i < steps:
            curve = self.beziers[0]
            if i == steps - 1 and nCurves >= 2: curve2 = self.beziers[1]
        else:
            i -= steps
            s1 = steps - 1
            ix = i // s1 + 1
            curve = self.beziers[ix]
            if ix < nCurves - 1: curve2 = self.beziers[ix + 1]
            i %= s1

        lut = curve.getLUT(steps)
        cip, closest = self._refineBinary(point, curve, lut, i)

        if curve2:
            lut = curve2.getLUT(steps)
            cip2, c2 = self._refineBinary(point, curve2, lut, 0)

            if c2 < closest:
                closest = c2
                cip = cip2

        return (closest, cip)

    def pointToString(self, point: Point) -> str:
        return ",".join([str(i) for i in point])

    def d(self, useSandT: bool = False, use_closed_attrib: bool = False, rel: bool = False):
        """Returns a path d-string for the path object.
        For an explanation of useSandT and use_closed_attrib, see the
        compatibility notes in the README."""

        commands: list[str] = []
        lastCommand = ""
        pen = firstPoint = self.start
        commands.append(f"M{self.pointToString(firstPoint)}")

        def getCommand(command: str, lastCommand: str) -> tuple[str, str]:
            return " " if lastCommand == command else command, command

        for bezier in self.beziers:
            segment = bezier.controlPoints
            if len(segment) == 2:
                # a line
                penX, penY = pen
                x, y = segment[1]

                if penX == x and penY == y:
                    continue
                elif penX == x:
                    # vertical line
                    command, lastCommand = getCommand("V", lastCommand)
                    commands.append(f"{command}{y}")
                elif penY == y:
                    # horizontal line
                    command, lastCommand = getCommand("H", lastCommand)
                    commands.append(f"{command}{x}")
                else:
                    point = self.pointToString(segment[1])
                    command, lastCommand = getCommand("L", lastCommand)
                    commands.append(f"{command}{point}")
                pen = (x, y)
            elif len(segment) == 3:
                    p1 = self.pointToString(segment[1])
                    p2 = self.pointToString(segment[2])
                    command, lastCommand = getCommand("Q", lastCommand)
                    commands.append(f"{command}{p1} {p2}")
                    pen = segment[2]
            elif len(segment) == 4:
                p1 = self.pointToString(segment[1])
                p2 = self.pointToString(segment[2])
                p3 = self.pointToString(segment[3])
                command, lastCommand = getCommand("C", lastCommand)
                commands.append(f"{command}{p1} {p2} {p3}")
                pen = segment[3]

        # if use_closed_attrib: commands.append("Z")

        return "".join(commands)

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def boundsRectangle(self):
        return self._bounds

    @property
    def beziers(self):
        return self._beziers

    @classmethod
    def pointXY(cls, point: Point) -> tuple[float, float]:
        return Bezier.pointXY(point)

    @classmethod
    def xyPoint(cls, x: float, y: float) -> Point:
        return Bezier.xyPoint(x, y)

class BOutline(object):
    __slots__ = "_bContours", "_bounds"

    def __init__(self, contours: list[Contour]):
        bounds = PathUtilities.BoundsRectangle()
        bContours: list[BContour] = []

        for contour in contours:
            bc = BContour(contour)
            bContours.append(bc)
            bounds = bounds.union(bc.boundsRectangle)

        self._bContours = bContours
        self._bounds = bounds

    def __iter__(self):
        return self._bContours.__iter__()

    @classmethod
    def pointXY(cls, point: Point) -> tuple[float, float]:
        return Bezier.pointXY(point)

    @classmethod
    def xyPoint(cls, x: float, y: float) -> Point:
        return Bezier.xyPoint(x, y)

    @classmethod
    def segmentFromPoints(cls, points: typing.Sequence[Point]) -> Bezier:
        return Bezier(points)

    @classmethod
    def unzipPoints(cls, points: typing.Sequence[Point]) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []

        for x, y in points:
            xs.append(x)
            ys.append(y)

        return xs, ys

    @classmethod
    def pathFromSegments(cls, *segments: Bezier):
        return BContour([s.controlPoints for s in segments])

    @property
    def bContours(self):
        return self._bContours

    @property
    def contours(self):
        return self._bContours

    @property
    def boundsRectangle(self):
        return self._bounds
