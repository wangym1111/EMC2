"""TEMPO saved-state diagnostics (NCAR/TEMPO 17c952b).

TEMPO's internal hail split is recombined before driver output. Its saved
graupel/hail category has prognostic mass, number and volume, not separate
prognostic hail fields. All functions here use SI density and mass content.
"""
import numpy as np

TEMPO_REVISION = '17c952bdfc0adcd1059aa404b80a34397a775b3c'
TEMPO_MPAS_831_REVISION = '9adf9ef2827104640125f0c322e1218fce74e94c'
TEMPO_REFERENCE_DENSITY = 101325. / (287.04 * 298.)


def tempo_legacy_velocity_scale(temperature, air_density):
    """9adf9ef / 17c952b graupel velocity ratio, before the shared rho factor.

    The earlier graupel_sedimentation uses local air density and dynamic
    viscosity in afall. Both revisions additionally apply sqrt(rho_not/rho).
    """
    tc = temperature - 273.15
    viscosity = (1.718 + .0049*tc - np.where(tc < 0, 1.2e-5*tc**2, 0.))*1.e-5
    if np.any(viscosity <= 0):
        raise ValueError('Temperature is outside the TEMPO viscosity relation')
    exponent = .54698726
    return (.47244157 * (.504843467198652*TEMPO_REFERENCE_DENSITY/air_density)**exponent
            * viscosity**(1.-2.*exponent))


def tempo_graupel_density(mass_mixing_ratio, volume_mixing_ratio):
    """Continuous and PSD-bin densities from mass kg/kg and volume m3/kg.

    Replicates the source's clipping and positive Fortran NINT bin indexing.
    The continuous density enters the velocity relation; the discrete density
    enters the PSD mass law. Clear values are assigned the reference 500.
    """
    active = mass_mixing_ratio > 0
    volume = np.where(active, volume_mixing_ratio, 1.)
    density = np.where(active, mass_mixing_ratio/volume, 500.)
    density = np.clip(density, 50., 800.)
    index = np.clip(np.floor(density*.01+.5).astype(int), 0, 8)
    bins = np.array([50., 100., 200., 300., 400., 500., 600., 700., 800.])
    return density, bins[index]


def tempo_graupel_intercept(content):
    """Single-moment TEMPO N0 (m-4), with Jensen et al. bounds."""
    return np.clip(10.**(3.4 + 2./7.*(np.log10(np.maximum(content, 1.e-9))+8.)),
                   1.e2, 1.e6)


def tempo_graupel_velocity(diameter, density):
    """Positive fall speed before the reference-air-density correction."""
    exponent = .54698726
    coefficient = (4.*density*9.8/(3.*.504843467198652*TEMPO_REFERENCE_DENSITY))**exponent
    return coefficient * diameter**(3.*exponent-1.)
