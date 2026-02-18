import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError # import here for p36 compatibility
from matplotlib.collections import LineCollection

from qutils.dynSys.dim6 import twoBodyJ2


Re = 6371 # km
mu = 398600 # km^3/s^2

# TODO - use ode45 to prop LEO