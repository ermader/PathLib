"""\
Test for Transform

Created on August 6, 2021

@author Eric Mader
"""
from PathLib.Transform import Transform

def test_transform():
    #
    # Example 2-6 from Mathematical Elements for Computer Graphics
    # Second Edition
    #
    # Rotate 90 degrees ccw about (4, 3)
    #
    m1 = [[1, 0, 0], [0, 1, 0], [-4, -3, 1]]  # move (4, 3) to origin
    m2 = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]   # rotate 90 degrees ccw
    m3 = [[1, 0, 0], [0, 1, 0], [4, 3, 1]]    # move origin to (4, 3)

    origin = (0, 0)
    about = (4, 3)

    assert Transform.translate(about, origin).transform == m1
    assert Transform.rotation(degrees=90, ccw=True).transform == m2
    assert Transform.translate(origin, about).transform == m3

    manualTransform = Transform(m1, m2, m3)
    rotationTransform = Transform.rotationAbout(about, degrees=90, ccw=True)

    assert manualTransform.transform == rotationTransform.transform
    assert rotationTransform.applyToPoint((8, 6)) == (1.0, 7.0)


