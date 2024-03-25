"""
This provides classes and methods that calculate the gradient level
and surface (10-m above ground) wind speed, based on a number of
parametric profiles and boundary layer models. These classes and
methods provide the wind field at a single point in time. The complete
wind swath is evaluated in the calling classes.

The :class:`windmodels.WindSpeedModel` classes define the
wind-pressure relations that define the maximum wind speed for a given
central pressure deficit.

The :class:`windmodels.WindProfileModel` classes define parametric
radial profiles of gradient level wind speed around the primary TC
vortex. These do not account for TC motion or interaction with the
surface.

The :class:`windmodels.WindFieldModel` classes define the boundary
layer models implemented, which relate the gradient level vortex to
the surface wind speed, incorporating the wavenumber-1 assymetry due
to storm forward motion, and the effects of (uniform) surface
roughness.

Wind speeds are assumed to represent a 1-minute mean wind
speed. Conversion to other averaging periods is performed by
application of gust factors in the calling classes.

:Note: Not all models are fully implemented - some cannot be fully
       implemented, as the mathematical formulation results in
       discontinuous profiles (e.g. Rankine vortex), for which a
       mathematical representation of the vorticity cannot be
       defined. The vorticity is required for the
       :class:`windmodels.KepertWindFieldModel`. Users should
       carefully select the combinations of wind profiles and wind
       fields to ensure sensible results.

"""

import numpy as np
from math import exp, sqrt
import Utilities.metutils as metutils
import logging
import warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class WindSpeedModel(object):

    """
    Abstract wind speed model.
    """

    def __init__(self, windProfileModel):
        """
        Abstract wind speed model.
        """
        self.profile = windProfileModel

    @property
    def eP(self):
        """
        Environment pressure.
        """
        eP = self.profile.eP
        if eP < 10000:
            eP = metutils.convert(eP, 'hPa', 'Pa')
        return eP

    @property
    def cP(self):
        """
        Current pressure.
        """
        cP = self.profile.cP
        if cP < 10000:
            cP = metutils.convert(cP, 'hPa', 'Pa')
        return cP

    @property
    def dP(self):
        """
        Pressure difference.
        """
        return self.eP - self.cP

    def maximum(self):
        """
        Maximum wind speed.
        """
        raise NotImplementedError


class HollandWindSpeed(WindSpeedModel):

    """
    .. |beta|   unicode:: U+003B2 .. GREEK SMALL LETTER BETA

    Holland (1980), An Analytic Model of the Wind and Pressure Profiles
    in Hurricanes. Mon. Wea. Rev, 108, 1212-1218 Density of air is
    assumed to be 1.15 kg/m^3.  |beta| is assumed to be 1.3. Other values
    can be specified.  Gradient level wind (assumed maximum).

    """

    def maximum(self):
        beta = self.profile.beta
        rho = 1.15
        return sqrt(beta * self.dP / (exp(1) * rho))



class WindProfileModel(object):

    """
    The base wind profile model.

    :param float lat: Latitude of TC centre.
    :param float lon: Longitude of TC centre.
    :param float eP: environmental pressure (hPa).
    :param float cP: centrral pressure of the TC (hPa).
    :param float rMax: Radius to maximum wind (km).
    :param windSpeedModel: A maximum wind speed model to apply.
    :type  windSpeedModel: :class:`windmodels.WindSpeedModel` instance.

    """

    def __init__(self, lat, lon, eP, cP, rMax, windSpeedModel):
        self.rho = 1.15  # density of air
        self.lat = lat
        self.lon = lon
        self.eP = eP
        self.cP = cP
        self.rMax = rMax
        self.speed = windSpeedModel(self)
        self.f = metutils.coriolis(lat)
        self.vMax_ = None

        if eP < 10000.:
            self.eP = metutils.convert(eP, 'hPa', 'Pa')
        else:
            self.eP = eP

        if cP < 10000.:
            self.cP = metutils.convert(cP, 'hPa', 'Pa')
        else:
            self.cP = cP

    @property
    def dP(self):
        """
        Pressure difference.
        """
        return self.eP - self.cP

    @property
    def vMax(self):
        """
        Maximum wind speed.
        """
        if self.vMax_:
            return self.vMax_
        else:
            return self.speed.maximum()

    @vMax.setter
    def vMax(self, value):
        """
        Set the maximum wind speed.

        :param float value: The maximum wind speed value to set.
        """
        self.vMax_ = value

    def velocity(self, R):
        """
        Calculate velocity as a function of radial distance `R`.
        Represents the velocity of teh gradient level vortex.

        :param R: :class:`numpy.ndarray` of distance of grid from
                  the TC centre.

        :returns: Array of gradient level wind speed.
        :rtype: :class:`numpy.ndarray`

        """
        raise NotImplementedError

    def vorticity(self, R):
        """
        Calculate the vorticity associated with the (gradient level)
        vortex at radius `R`.

        :param R: :class:`numpy.ndarray` of distance of grid from
                  the TC centre.

        :returns: Array of gradient level (relative) vorticity.
        :rtype: :class:`numpy.ndarray`

        """
        raise NotImplementedError


class HollandWindProfile(WindProfileModel):

    """
    .. |beta|   unicode:: U+003B2 .. GREEK SMALL LETTER BETA

    Holland profile. For `r < rMax`, we reset the wind field to a
    cubic profile to avoid the barotropic instability mentioned in
    Kepert & Wang (2001).

    :param float lat: Latitude of TC centre.
    :param float lon: Longitude of TC centre.
    :param float eP: environmental pressure (hPa).
    :param float cP: centrral pressure of the TC (hPa).
    :param float rMax: Radius to maximum wind (m).
    :param float beta: |beta| parameter.
    :param windSpeedModel: A maximum wind speed model to apply.
    :type  windSpeedModel: :class:`windmodels.WindSpeedModel` instance.

    """

    def __init__(self, lat, lon, eP, cP, rMax, beta,
                 windSpeedModel=HollandWindSpeed):
        WindProfileModel.__init__(self, lat, lon, eP, cP, rMax,
                                  windSpeedModel)
        self.beta = beta

    def secondDerivative(self):
        """
        Second derivative of profile at rMax.
        """

        beta = self.beta
        dP = self.dP
        rho = self.rho
        f = self.f
        rMax = self.rMax

        E = exp(1)

        d2Vm = ((beta * dP * (-4 * beta ** 3 * dP / rho -
                (-2 + beta ** 2) * E * (np.abs(f) * rMax) ** 2)) /
                (E * rho * sqrt((4 * beta * dP) / (E * rho) +
                 (f * rMax) ** 2) * (4 * beta * dP * rMax ** 2 / rho +
                                     E * (f * rMax ** 2) ** 2)))

        try:
            assert d2Vm < 0.0
        except AssertionError:
            log.critical(("Pressure deficit: {0:.2f} hPa,"
                          " RMW: {1:.2f} km".format(dP/100., rMax/1000.)))
            raise

        return d2Vm

    def firstDerivative(self):
        """
        First derivative of profile at rMax
        """
        beta = self.beta
        dP = self.dP
        rho = self.rho
        f = self.f
        rMax = self.rMax

        E = exp(1)

        dVm = (-np.abs(f)/2 + (E * (f**2) * rMax *
                               np.sqrt((4 * beta * dP / rho) / E +
                                       (f * rMax) ** 2)) /
               (2 * (4 * beta * dP / rho + E * (f * rMax)**2)))
        return dVm

    def velocity(self, R):
        """
        Calculate velocity as a function of radial distance.
        Represents the velocity of teh gradient level vortex.

        :param R: :class:`numpy.ndarray` of distance of grid from
                  the TC centre (metres).

        :returns: Array of gradient level wind speed.
        :rtype: :class:`numpy.ndarray`

        """

        d2Vm = self.secondDerivative()
        dVm = self.firstDerivative()

        try:
            from ._windmodels import fhollandvel
            V = np.empty_like(R)
            fhollandvel(
                V.ravel(), R.ravel(), d2Vm, dVm, self.rMax,
                self.vMax, self.beta, self.dP, self.rho, self.f, V.size
            )

        except ImportError:
            aa = ((d2Vm / 2. - (dVm - self.vMax / self.rMax) /
                   self.rMax) / self.rMax)
            bb = (d2Vm - 6 * aa * self.rMax) / 2.
            cc = dVm - 3 * aa * self.rMax ** 2 - 2 * bb * self.rMax
            delta = (self.rMax / R) ** self.beta
            edelta = np.exp(-delta)

            V = (np.sqrt((self.dP * self.beta / self.rho) *
                         delta * edelta + (R * self.f / 2.) ** 2) -
                 R * np.abs(self.f) / 2.)

            icore = np.where(R <= self.rMax)
            V[icore] = (R[icore] * (R[icore] * (R[icore] * aa + bb) + cc))
            V = np.sign(self.f) * V
        return V

    def vorticity(self, R):
        """
        Calculate the vorticity associated with the (gradient level)
        vortex.

        :param R: :class:`numpy.ndarray` of distance of grid from
                  the TC centre (metres).

        :returns: Array of gradient level (relative) vorticity.
        :rtype: :class:`numpy.ndarray`

        """
        # Calculate first and second derivatives at R = Rmax:

        d2Vm = self.secondDerivative()
        dVm = self.firstDerivative()

        try:
            from ._windmodels import fhollandvort
            Z = np.empty_like(R)
            fhollandvort(
                Z.ravel(), R.ravel(), d2Vm, dVm, self.rMax, self.vMax,
                self.beta, self.dP, self.rho, self.f, Z.size
            )

        except ImportError:
            beta = self.beta
            delta = (self.rMax / R) ** beta
            edelta = np.exp(-delta)

            Z = np.abs(self.f) + \
                (beta**2 * self.dP * (delta**2) * edelta /
                 (2 * self.rho * R) - beta**2 * self.dP * delta * edelta /
                 (2 * self.rho * R) + R * self.f**2 / 4) / \
                np.sqrt(beta * self.dP * delta * edelta /
                        self.rho + (R * self.f / 2)**2) + \
                (np.sqrt(beta * self.dP * delta * edelta /
                         self.rho + (R * self.f / 2)**2)) / R

            aa = ((d2Vm / 2 - (dVm - self.vMax /
                  self.rMax) / self.rMax) / self.rMax)
            bb = (d2Vm - 6 * aa * self.rMax) / 2
            cc = dVm - 3 * aa * self.rMax ** 2 - 2 * bb * self.rMax

            icore = np.where(R <= self.rMax)
            Z[icore] = R[icore] * (R[icore] * 4 * aa + 3 * bb) + 2 * cc
            Z = np.sign(self.f) * Z

        return Z


class WindFieldModel(object):

    """
    Wind field (boundary layer) models. These define the boundary
    layer models implemented, which relate the gradient level vortex
    to the surface wind speed, incorporating the wavenumber-1
    assymetry due to storm forward motion, and the effects of
    (uniform) surface roughness.

    :param windProfileModel: A `wind.WindProfileModel` instance.

    """

    def __init__(self, windProfileModel):
        self.profile = windProfileModel
        self.V = None
        self.Z = None

    @property
    def rMax(self):
        """
        Helper property to return the maximum radius from the
        wind profile.
        """
        return self.profile.rMax

    @property
    def f(self):
        """
        Helper property to return the coriolis force from the
        wind profile.
        """
        return self.profile.f

    def velocity(self, R):
        """
        Helper property to return the wind velocity at radiuses `R`
        from the wind profile or the precalculated attribute.
        """
        if self.V is None:
            return self.profile.velocity(R)
        else:
            return self.V

    def vorticity(self, R):
        """
        Helper property to return the wind vorticity at radiuses `R`
        from the wind profile or the precalculated attribute.
        """
        if self.Z is None:
            return self.profile.vorticity(R)
        else:
            return self.Z

    def field(self, R, lam, vFm, thetaFm, thetaMax=0.):
        """
        The wind field.
        """
        raise NotImplementedError


class KepertWindField(WindFieldModel):

    """
    Kepert, J., 2001: The Dynamics of Boundary Layer Jets within the
    Tropical Cyclone Core. Part I: Linear Theory.  J. Atmos. Sci., 58,
    2469-2484

    """

    def field(self, R, lam, vFm, thetaFm, thetaMax=0.):
        """
        :param R: Distance from the storm centre to the grid (km).
        :type  R: :class:`numpy.ndarray`
        :param lam: Direction (0=east, radians, positive anti-clockwise)
                    from storm centre to the grid.
        :type  lam: :class:`numpy.ndarray`
        :param float vFm: Foward speed of the storm (m/s).
        :param float thetaFm: Forward direction of the storm (0=east, radians,
                    positive anti-clockwise).
        :param float thetaMax: Bearing of the location of the maximum
                               wind speed, relative to the direction of
                               motion.

        """

        K = 50.  # Diffusivity
        Cd = 0.002  # Constant drag coefficient
        Vm = self.profile.vMax
        if type(self.profile) in [HollandWindProfile]:
            try:
                from ._windmodels import fkerpert

                d2Vm, dVm = self.profile.secondDerivative(), self.profile.firstDerivative()
                Ux, Vy = np.empty_like(R), np.empty_like(R)
                n = Ux.size
                fkerpert(
                    R.ravel(), lam.ravel(), self.f, self.rMax, Vm, thetaFm,
                    vFm, d2Vm, dVm, self.profile.dP, self.profile.beta, self.profile.rho,
                    Ux.ravel(), Vy.ravel(), n
                )
                return Ux, Vy
            except ImportError:
                pass

        V = self.velocity(R)
        Z = self.vorticity(R)
        if (vFm > 0) and (Vm/vFm < 5.):
            Umod = vFm * np.abs(1.25*(1. - (vFm/Vm)))
        else:
            Umod = vFm
        Vt = Umod * np.ones(V.shape)

        core = np.where(R > 2. * self.rMax)
        Vt[core] = Umod * np.exp(-((R[core] / (2.*self.rMax)) - 1.) ** 2.)

        al = ((2. * V / R) + self.f) / (2. * K)
        be = (self.f + Z) / (2. * K)
        gam = (V / (2. * K * R))

        albe = np.sqrt(al / be)

        ind = np.where(np.abs(gam) > np.sqrt(al * be))
        chi = np.abs((Cd / K) * V / np.sqrt(np.sqrt(al * be)))
        eta = np.abs((Cd / K) * V / np.sqrt(np.sqrt(al * be) + np.abs(gam)))
        psi = np.abs((Cd / K) * V / np.sqrt(np.abs(np.sqrt(al * be) -
                                                   np.abs(gam))))

        i = complex(0., 1.)
        A0 = -(chi * (1 + i * (1 + chi)) * V) / (2 * chi**2 + 3 * chi + 2)

        # Symmetric surface wind component
        u0s = A0.real * albe * np.sign(self.f)
        v0s = A0.imag

        Am = -(psi * (1 + 2 * albe + (1 + i) * (1 + albe) * eta) * Vt) / \
              (albe * ((2 + 2 * i) * (1 + eta * psi) + 3 * psi + 3 * i * eta))
        AmIII = -(psi * (1 + 2 * albe + (1 + i) * (1 + albe) * eta) * Vt) / \
                 (albe * ((2 - 2 * i + 3 * (eta + psi) + (2 + 2 * i) *
                  eta * psi)))
        Am[ind] = AmIII[ind]

        # First asymmetric surface component
        ums = (Am * np.exp(-i * (lam - thetaFm) * np.sign(self.f))).real * albe
        vms = (Am * np.exp(-i * (lam - thetaFm) * np.sign(self.f))).imag * np.sign(self.f)

        Ap = -(eta * (1 - 2 * albe + (1 + i) * (1 - albe) * psi) * Vt) / \
              (albe * ((2 + 2 * i) * (1 + eta * psi) + 3 * eta + 3 * i * psi))
        ApIII = -(eta * (1 - 2 * albe + (1 - i) * (1 - albe)*psi) * Vt) / \
                 (albe * (2 + 2 * i + 3 * (eta + psi) + (2 - 2 * i) *
                  eta * psi))
        Ap[ind] = ApIII[ind]

        # Second asymmetric surface component
        ups = (Ap * np.exp(i * (lam - thetaFm) * np.sign(self.f))).real * albe
        vps = (Ap * np.exp(i * (lam - thetaFm) * np.sign(self.f))).imag * np.sign(self.f)

        # Total surface wind in (moving coordinate system)
        us = u0s + ups + ums
        vs = v0s + vps + vms + V

        usf = us + Vt * np.cos(lam - thetaFm)
        vsf = vs - Vt * np.sin(lam - thetaFm)
        phi = np.arctan2(usf, vsf)

        # Surface winds, cartesian coordinates
        Ux = np.sqrt(usf ** 2. + vsf ** 2.) * np.sin(phi - lam)
        Vy = np.sqrt(usf ** 2. + vsf ** 2.) * np.cos(phi - lam)

        return Ux, Vy


# Automatic discovery of models and required parameters


def allSubclasses(cls):
    """
    Recursively find all subclasses of a given class.
    """
    return cls.__subclasses__() + \
        [g for s in cls.__subclasses__() for g in allSubclasses(s)]


def profile(name):
    """
    Helper function to return the appropriate wind profile
    model given a `name`.
    """
    return PROFILES[name]


def profileParams(name):
    """
    List of additional parameters required for a wind profile model.
    """
    from inspect import getfullargspec
    std = getfullargspec(WindProfileModel.__init__)[0]
    new = getfullargspec(profile(name).__init__)[0]
    params = [p for p in new if p not in std]
    return params


def field(name):
    """
    Helper function to return the appropriate wind field
    model given a `name`.
    """
    return FIELDS[name]


def fieldParams(name):
    """
    List of additional parameters required for a wind field model.
    """
    from inspect import getfullargspec
    std = getfullargspec(WindFieldModel.__init__)[0]
    new = getfullargspec(field(name).__init__)[0]
    params = [p for p in new if p not in std]
    return params


PROFILES = dict([(k.__name__.replace('WindProfile', '').lower(), k)
                 for k in allSubclasses(vars()['WindProfileModel'])])

FIELDS = dict([(k.__name__.replace('WindField', '').lower(), k)
               for k in allSubclasses(vars()['WindFieldModel'])])
