import matplotlib.pyplot as plt
import numpy as np


__author__ = "Soyong Shin"


def get_parts():
    """
    Get parts dictionary and corresponding colors
    """

    part = {}
    part['face'] = [18, 17, 1, 15, 16, 'crimson']
    part['neck'] = [0, 1, 'maroon']
    part['back'] = [2, 0, 'darkred']
    part['larm1'] = [5, 4, 'forestgreen']
    part['larm2'] = [4, 3, 'limegreen']
    part['larm3'] = [3, 0, 'springgreen']
    part['rarm1'] = [11, 10, 'gold']
    part['rarm2'] = [10, 9, 'orange']
    part['rarm3'] = [9, 0, 'darkorange']
    part['hip'] = [12, 2, 6, 'purple']
    part['rleg1'] = [14, 13, 'darkblue']
    part['rleg2'] = [13, 12, 'midnightblue']
    part['lleg1'] = [8, 7, 'seagreen']
    part['lleg2'] = [7, 6, 'darkgreen']
    part['rfoot'] = [22, 23, 14, 24, 'mediumblue']
    part['lfoot'] = [19, 20, 8, 21, 'mediumseagreen']

    return part

def set_range(ax, offset=[0, 0, 0]):
    x, y, z = offset
    ax.set_xlim(-90 + x, 180 + x)
    ax.set_ylim(-225 + y, 45 + y)
    ax.set_zlim(-135 + z, 0 + z)

def draw_ground(ax):
    X = np.arange(-1000, 1500, 250)
    Z = np.arange(-1000, 1500, 250)
    X, Z = np.meshgrid(X,Z)
    Y = np.zeros(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.1, linewidth=0, antialiased=False)

