"""PSD diagnostics for MPAS's bundled Thompson–Eidhammer implementation.

SI units throughout: mass content kg m-3, number m-3, diameter m, T K.
Source: MPAS-Dev/MPAS-Model, module_mp_thompson.F (not TEMPO).
"""
import numpy as np
from scipy.special import gamma


def cloud_shape(number):
    """Integer nu_c, including Fortran NINT rounding for positive numbers."""
    return np.minimum(15., np.floor(1.e9 / np.maximum(number, 2.) + .5) + 2.)


def snow_parameters(content, temperature, tempo=False):
    """Return Field et al. two-gamma amplitude and inverse characteristic size.

    The physical PSD is evaluated by :func:`snow_distribution`; these are
    not the intercept and slope of a single exponential distribution.
    """
    tc = np.minimum(-.1, temperature - 273.15)
    sa = (5.065339, -.062659, -3.032362, .029469, -.000285,
          .31255, .000204, .003199, 0., -.015952)
    sb = (.476221, -.015896, .165977, .007468, -.000141,
          .060366, .000079, .000594, 0., -.003577)
    def coefficients(n):
        terms = (1., tc, n, tc*n, tc**2, n*n, tc**2*n, tc*n*n, tc**3, n**3)
        return (10. ** sum(c*t for c, t in zip(sa, terms)),
                sum(c*t for c, t in zip(sb, terms)))

    reference = np.maximum(content, 1.e-30) / .069
    m2 = reference
    if tempo:
        # TEMPO snow_moments applies this inversion even for bm_s=2.
        a, b = coefficients(2.)
        m2 = (reference/a)**(1./b)
    a, b = coefficients(3.)
    m3 = a * m2**b
    return reference**4 / m3**3, reference / m3


def snow_distribution(diameter, amplitude, slope):
    """Field et al. (2005) two-component snow PSD, in m-4."""
    x = slope * diameter
    return amplitude * (490.6*np.exp(-20.78*x) +
                        17.46*x**.6357*np.exp(-3.29*x))


def snow_number(amplitude, slope):
    return amplitude / slope * (490.6/20.78 +
                               17.46*gamma(1.6357)/3.29**1.6357)


def graupel_intercept(content, rain_content, rain_number, temperature, density):
    """Thompson diagnostic N0, with vertical dimension last, bottom to top.

    Reconstruct the top-down running minimum in calc_refl10cm. This uses
    instantaneous rain moments; it cannot recreate earlier microphysics steps.
    """
    lamr = (np.pi*1000.*np.maximum(rain_number, 1.e-30) /
            np.maximum(rain_content, 1.e-30))**(1./3.)
    mvd = 3.672/lamr
    levels = np.arange(content.shape[-1])
    warm_top = np.max(np.where(temperature >= 270.65, levels, 0), axis=-1)
    supercooled = ((levels > warm_top[..., None]) &
                  (rain_content/density > 1.e-12) & (mvd > 100.e-6))
    x = np.where(supercooled, 4.01 + np.log10(mvd), .01)
    y = 4.31 + np.log10(np.maximum(5.e-5, content))
    z = 3.1 + 100. / (300.*x*y/(10./x + 1. + .25*y) + 30. + 10.*y)
    n0 = np.clip(10.**z, 1.e4, 3.e6)
    return np.minimum.accumulate(n0[..., ::-1], axis=-1)[..., ::-1]
