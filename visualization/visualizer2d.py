"""
Common 2D visualizations using pyplot
Author: Jeff Mahler
"""
import numpy as np

import IPython

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from autolab_core import Box, Contour
from perception import BinaryImage, ColorImage, DepthImage, GrayscaleImage, RgbdImage, GdImage, SegmentationImage

class Visualizer2D:
    @staticmethod
    def figure(size=(8,8), *args, **kwargs):
        """ Creates a figure.

        Parameters
        ----------
        size : 2-tuple
           size of the view window in inches
        args : list
           args of mayavi figure
        kwargs : list
           keyword args of mayavi figure

        Returns
        -------
        pyplot figure
            the current figure
        """
        return plt.figure(figsize=size, *args, **kwargs)
    
    @staticmethod
    def show(*args, **kwargs):
        """ Show the current figure """
        plt.show(*args, **kwargs)

    @staticmethod
    def clf(*args, **kwargs):
        """ Clear the current figure """
        plt.clf(*args, **kwargs)

    @staticmethod
    def savefig(*args, **kwargs):
        """ Save the current figure """
        plt.savefig(*args, **kwargs)

    @staticmethod
    def colorbar(*args, **kwargs):
        """ Adds a colorbar to the current figure """
        plt.colorbar(*args, **kwargs)

    @staticmethod
    def subplot(*args, **kwargs):
        """ Creates a subplot in the current figure """
        plt.subplot(*args, **kwargs)

    @staticmethod
    def title(*args, **kwargs):
        """ Creates a title in the current figure """
        plt.title(*args, **kwargs)

    @staticmethod
    def xlabel(*args, **kwargs):
        """ Creates an x axis label in the current figure """
        plt.xlabel(*args, **kwargs)

    @staticmethod
    def ylabel(*args, **kwargs):
        """ Creates an y axis label in the current figure """
        plt.ylabel(*args, **kwargs)

    @staticmethod
    def legend(*args, **kwargs):
        """ Creates a legend for the current figure """
        plt.legend(*args, **kwargs)

    @staticmethod
    def scatter(*args, **kwargs):
        """ Scatters points """
        plt.scatter(*args, **kwargs)

    @staticmethod
    def plot(*args, **kwargs):
        """ Plots lines """
        plt.plot(*args, **kwargs)
    
    @staticmethod
    def imshow(image, **kwargs):
        """ Displays an image.

        Parameters
        ----------
        image : :obj:`perception.Image`
            image to display
        """
        if isinstance(image, BinaryImage) or isinstance(image, GrayscaleImage):
            plt.imshow(image.data, cmap=plt.cm.gray, **kwargs)
        elif isinstance(image, ColorImage) or isinstance(image, SegmentationImage):
            plt.imshow(image.data, **kwargs)
        elif isinstance(image, DepthImage):
            plt.imshow(image.data, cmap=plt.cm.gray_r, **kwargs)
        elif isinstance(image, RgbdImage):
            # default to showing color only, for now...
            plt.imshow(image.color.data, **kwargs)
        elif isinstance(image, GdImage):
            # default to showing gray only, for now...
            plt.imshow(image.gray.data, cmap=plt.cm.gray, **kwargs)
        plt.axis('off')

    @staticmethod
    def box(b, line_width=2, color='g', style='-'):
        """ Draws a box on the current plot.

        Parameters
        ----------
        b : :obj:`autolab_core.Box`
            box to draw
        line_width : int
            width of lines on side of box
        color : :obj:`str`
            color of box
        style : :obj:`str`
            style of lines to draw
        """
        if not isinstance(b, Box):
            raise ValueError('Input must be of type Box')
            
        # get min pixels
        min_i = b.min_pt[1]
        min_j = b.min_pt[0]
        max_i = b.max_pt[1]
        max_j = b.max_pt[0]
        top_left = np.array([min_i, min_j])
        top_right = np.array([max_i, min_j])
        bottom_left = np.array([min_i, max_j])
        bottom_right = np.array([max_i, max_j])

        # create lines
        left = np.c_[top_left, bottom_left].T
        right = np.c_[top_right, bottom_right].T
        top = np.c_[top_left, top_right].T
        bottom = np.c_[bottom_left, bottom_right].T

        # plot lines
        plt.plot(left[:,0], left[:,1], linewidth=line_width, color=color, linestyle=style)
        plt.plot(right[:,0], right[:,1], linewidth=line_width, color=color, linestyle=style)
        plt.plot(top[:,0], top[:,1], linewidth=line_width, color=color, linestyle=style)
        plt.plot(bottom[:,0], bottom[:,1], linewidth=line_width, color=color, linestyle=style)

    @staticmethod
    def contour(c, subsample=1, size=10, color='g'):
        """ Draws a contour on the current plot by scattering points.

        Parameters
        ----------
        c : :obj:`autolab_core.Contour`
            contour to draw
        subsample : int
            subsample rate for boundary pixels
        size : int
            size of scattered points
        color : :obj:`str`
            color of box
        """
        if not isinstance(c, Contour):
            raise ValueError('Input must be of type Contour')
            
        for i in range(c.num_pixels)[0::subsample]:
            plt.scatter(c.boundary_pixels[i,1], c.boundary_pixels[i,0], s=size, c=color)

