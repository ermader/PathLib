"""\
A pen to convert a glyph into a list of segments

Created on October 14, 2020

@author Eric Mader
"""

import logging
from typing import Any
from .PathTypes import Point, Contour
from .Transform import Transform

class SegmentPen:
    __slots__ = "_contours", "_glyphSet", "logger", "_lastOnCurve", "_contour", "_segment"

    #
    # Instead of glyphSet, we could take a
    # callback to fetch a glyph. This would
    # remove any knowlede of how glyphSet works.
    # Just knowing that glyphSet is a dictionary
    # doesn't seem that bad, though.
    #
    def __init__(self, glyphSet: Any, logger: logging.Logger):
        self._contours: list[Contour] = []
        self._glyphSet = glyphSet
        self.logger = logger
        self._lastOnCurve: Point = (0, 0)

    def addPoint(self, pt: Point, segmentType: int, smooth: bool, name: str) -> None:
        raise NotImplementedError

    def moveTo(self, pt: Point):
        self._lastOnCurve = pt

        # This is for glyphs, which are always closed paths,
        # so we assume that the move is the start of a new contour
        self._contour: list[list[Point]] = []
        self._segment: list[Point] = []
        self.logger.debug(f"moveTo({pt})")

    def lineTo(self, pt: Point):
        # an old bug in fontTools.ttLib.tables._g_l_y_f.Glyph.draw()
        # can cause this to be called w/ a zero-length line.
        if pt != self._lastOnCurve:
            segment = [self._lastOnCurve, pt]
            self._contour.append(segment)
            self.logger.debug(f"lineTo({pt})")
            self._lastOnCurve = pt

    def curveTo(self, *points: Point):
        segment = [self._lastOnCurve]
        segment.extend(points)
        self._contour.append(segment)
        self.logger.debug(f"CurveTo({points})")
        self._lastOnCurve = points[-1]

    def qCurveTo(self, *points: Point):
        segment = [self._lastOnCurve]
        segment.extend(points)

        if len(segment) <= 3:
            self._contour.append(segment)
        else:
            # a starting on-curve point, two or more off-curve points, and a final on-curve point
            startPoint = segment[0]
            for i in range(1, len(segment) - 2):
                p1x, p1y = segment[i]
                p2x, p2y = segment[i + 1]
                impliedPoint = (0.5 * (p1x + p2x), 0.5 * (p1y + p2y))
                self._contour.append([startPoint, segment[i], impliedPoint])
                startPoint = impliedPoint
            self._contour.append([startPoint, segment[-2], segment[-1]])
        self.logger.debug(f"qCurveTo({points})")
        self._lastOnCurve = segment[-1]

    def beginPath(self) -> None:
        raise NotImplementedError

    def closePath(self):
        if self._contour:  # Ignore this if called with an empty contour
            self._contours.append(self._contour)
            if self._contour[0][0] != self._contour[-1][-1]:
                self._contour.append([self._contour[-1][-1], self._contour[0][0]])
            self._contour = []
        self.logger.debug("closePath()")

    def endPath(self) -> None:
        raise NotImplementedError

    identityTransformation = (1, 0, 0, 1, 0, 0)

    def addComponent(self, glyphName: str, transformation: tuple[float, float, float, float, float, float]):
        self.logger.debug(f"addComponent(\"{glyphName}\", {transformation}")
        if transformation != self.identityTransformation:
            xScale, xyScale, yxScale, yScale, xOffset, yOffset = transformation
            m = Transform._matrix(
                a=xScale,
                b=xyScale,
                c=yxScale,
                d=yScale,
                m=xOffset,
                n=yOffset
            )
            t = Transform(m)
        else:
            t = None

        glyph = self._glyphSet[glyphName]
        cpen = SegmentPen(self._glyphSet, self.logger)
        glyph.draw(cpen)
        contours = t.applyToContours(cpen.contours) if t else cpen.contours
        self.contours.extend(contours)

    @property
    def contours(self):
        return self._contours

