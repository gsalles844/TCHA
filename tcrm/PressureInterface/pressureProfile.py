"""
:mod:`pressureProfile` -- radial pressure profile for a TC
==========================================================

.. module:: pressureProfile
    :synopsis: Returns the radial pressure field for a range
               of parametric radial wind models.

.. moduleauthor:: Craig Arthur <craig.arthur@ga.gov.au>

The available wind profiles are:
    Rankine vortex - removed!
    Jelesnianski - removed!
    Holland (with cubic core) - implemented!
    Schloemer - removed!
    Willoughby and Rahn - removed!
    McConochie et al. - removed!
    Powell et al. - removed!

SeeAlso:
Constraints:
Version: $Rev: 810 $

References:
Holland, G.J., 1980:
An Analytic model of the Wind and Pressure Profiles in Hurricanes.
Mon. Wea. Rev., 108, 1212-1218.
Jelesnianski, C.P., 1966:
Numerical Computations of Storm Surges without Bottom Stress.
Mon. Wea. Rev., 94(6), 379-394
McConochie, J.D., T.A. Hardy and L.B. Mason, 2004:
Modelling tropical cyclone over-water wind and pressure fields.
Ocean Engineering, 31, 1757-1782
Powell, M., G. Soukup, S. Cocke, S. Gulati, N. Morrisuea-Leroy,
S. Hamid, N. Dorst and L. Axe, 2005:
State of Florida hurricane loss projection model: Atmospheric science component.
Journal of Wind Engineering and Industrial Aerodynamics, 93 (8), 651-674
Schloemer, R.W., 1954:
Analysis and synthesis of hurricane wind patterns over Lake Okeechobee.
NOAA Hydromet. Rep. 31, 49 pp.
Willoughby, H.E. and M.E. Rahn, 2004:
Parametric Representation of the Primary Hurricane Vortex. Part I:
Observations and Evaluation of the Holland (1980) Model.
Mon. Wea. Rev., 132, 3033-3048

$Id: pressureProfile.py 810 2012-02-21 07:52:50Z nsummons $
"""

import logging

import Utilities.metutils as metutils

import numpy
import wind.vmax as vmax
import time


class PrsProfile:
    """
    Description: Define the radial wind profiles used in tropical
    cyclone modelling. These are radial profiles only and do not include
    asymmetries that arise due to the forward motion of the storm.

    Parameters:
        R: grid of distances from the storm centre (distances in km)
        pEnv: Environmental pressure (Pa)
        pCentre: Central pressure of storm (Pa)
        rMax: Radius of maximum winds (km)
        cLat: Latitude of storm centre
        cLon: Longitude of storm centre
        beta: Holland beta parameter
    Members:
        R: grid of distances from the storm centre (distances in km)
        pEnv: Environmental pressure (Pa)
        pCentre: Central pressure of storm (Pa)
        rMax: Radius of maximum winds (km)
        cLat: Latitude of storm centre
        cLon: Longitude of storm centre
        beta: Holland beta parameter

    Methods:
        (rankine: Rankine vortex)
        (jelesnianski: Jelesnianski's storm surge model wind field)
        holland: Holland's radial wind field
        willoughby: Holland profile with beta a function of vMax, rMax
                    and cLat
        schloemer: Holland profile with beta==1
        doubleHolland: McConochie's double vortex model

    Internal Methods:
        None
    """

    def __init__(self, R, pEnv, pCentre, rMax, cLat, cLon, beta=1.3,
                 rMax2=250., beta1=None, beta2=None ):
        """
        Initialise required fields
        """
        self.R = R
        self.cLon = cLon
        self.cLat = cLat
        self.rMax = rMax
        self.dP = pEnv-pCentre
        self.pCentre = pCentre
        self.pEnv = pEnv
        # Density of air:
        self.rho = 1.15
        self.f = metutils.coriolis(cLat)
        self.beta = beta
        self.rMax2 = rMax2
        self.rMax2 = rMax2
        self.beta1 = beta1
        self.beta2 = beta2
        self.logger = logging.getLogger()
        self.logger.debug("Storm centre: %3f %3f" %(self.cLon, self.cLat))
        self.logger.debug("Coriolis parameter: %3f" % self.f)



    def holland(self, beta=None):
        """
        Holland profile.
        """
        if beta == None:
            beta = self.beta
        t0 = time.time()

        try:
            from ._pressureProfile import fhollandpressure
            P = numpy.empty(self.R.shape)
            fhollandpressure(
                P.ravel(), self.R.ravel(), self.rMax, self.pCentre, self.dP, beta
            )
        except ImportError:
            P = self.pCentre + self.dP*numpy.exp(-(self.rMax/self.R)**beta)
        self.logger.debug("Timing for holland wind profile calculation: %.3f"
                           % (time.time()-t0))
        return P
