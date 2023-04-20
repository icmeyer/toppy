import numpy as np
import matplotlib.pyplot as plt


def generate_circle_bisect_points(x, y, nsegs):
    """
    Generate points for a circle cut by plane
    Currently only works for resulting circles cut above the center
    x: the x intercept of the circle
    y: the y intercept of the circle
    nsegs: number of line segments forming sphere
    """
    # r = (x + y**2)**(2)/(4 * y**2)
    r = (x**2 + y**2)/(2 * y)
    print(r)
    y0 = y - r
    xs = np.linspace(-x, x, nsegs + 1)
    print(xs)
    ys = (r**2 - xs**2)**(1/2) + y0

    polystring = ''
    for i in range(len(xs)):
        polystring += ' {:0.2f} {:0.2f}'.format(xs[i], ys[i])
    polystring += ' {:0.2f} {:0.2f}'.format(xs[0], ys[0])
    nentries = 2*(len(xs) + 1)
    polystring = str(nentries) + ' ' + polystring
    print(polystring)


if __name__=='__main__':
    generate_circle_bisect_points(19.89, 7.55, 20)

