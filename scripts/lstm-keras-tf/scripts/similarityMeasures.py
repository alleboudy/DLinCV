import numpy as np
from math import*


def euclidean_distance(x, y):

    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

def getError(posx, posq, actualx, actualq):
    q1 = actualq / np.linalg.norm(actualq)
    q2 = posq / np.linalg.norm(posq)
    d = abs(np.sum(np.multiply(q1, q2)))
    theta = 2 * np.arccos(d) * 180 / np.pi
    errx = np.linalg.norm(actualx - posx)
    return errx, theta
