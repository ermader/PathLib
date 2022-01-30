"""\
Types for type hints in PathLib

Created on January 15, 2022

@author Eric Mader
"""

import typing

Point = tuple[float, float]
Segment = typing.Sequence[Point]
Contour = typing.Sequence[Segment]
