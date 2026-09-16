================================
Construction of the Model object
================================

MPAS Thompson–Eidhammer adapter
------------------------------

``emc2.core.model.MPAS`` (also ``emc2.core.MPAS``) accepts a native
MPAS-Atmosphere NetCDF file or an ``xarray.Dataset``. It targets the
fixed-density Thompson implementation bundled with MPAS, including its
aerosol-aware extension by default. ``mcphys_scheme="TEMPO"`` selects the
TEMPO saved-state reconstruction described below. The adapter does not
implement arbitrary newer WRF variants or the CAM-MPAS physics suite.

Example::

    import emc2

    model = emc2.core.model.MPAS("history.nc", cell_indices=[100])
    radar = emc2.core.instruments.KAZR("nsa")
    model = emc2.simulator.main.make_simulated_data(
        model, radar, N_columns=1, use_rad_logic=False, parallel=False)

Use ``HSRL()`` or another EMC2 lidar instrument in the same interface.
One subcolumn preserves the resolved column without imposing additional
subgrid cloud-water variability. All five hydrometeor classes are processed
as resolved/stratiform, with binary fractions above the configurable
``q_hyd_truncation_cutoff`` (default 1e-14 kg/kg). No parameterized convective
condensate is inferred. Precipitation can occur outside cloud.

Input fields and units
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Native defaults (missing unit attributes use these units)
   :header-rows: 1
   :widths: 20 20 60

   * - Input
     - Units
     - Interpretation
   * - qc, qi, qr, qs, qg
     - kg/kg dry air
     - Required cloud water (cl), cloud ice (ci), rain (pl), snow (pi), graupel (gr).
   * - nc, ni, nr
     - particles/kg dry air
     - Required droplet, ice, rain number mixing ratios. Converted with dry-air density to cm-3.
   * - nwfa, nifa
     - particles/kg dry air
     - Optional water-/ice-friendly aerosol reservoirs. Preserved in native form, not activated droplet/ice numbers.
   * - qv
     - kg/kg dry air
     - Required vapor mixing ratio, used by attenuation and density conversion.
   * - pressure
     - Pa
     - Total pressure; alternatively pressure_base + pressure_p. EMC2 output is hPa.
   * - temperature
     - K
     - Otherwise derived from dry theta and (p/100000)**(2/7), or from theta_m after removing its moisture factor.
   * - rho
     - kg/m3
     - Dry-air density. Otherwise p / [287.04 T (1 + qv/0.622)], following Thompson's conversion.
   * - zgrid
     - m MSL
     - Required interfaces; arithmetic midpoints give mass-level heights in meters.
   * - re_cloud, re_ice, re_snow
     - m
     - Optional radiation radii, preferred when supplied; converted to microns.

Native ``Time, nCells, nVertLevels`` fields are required; already selected
single columns without ``nCells`` are supported. ``zgrid`` uses
``nVertLevelsP1`` of length ``nVertLevels + 1``. Heights must increase upward.
``xtime`` is decoded when no datetime ``Time`` coordinate exists. Time-range
selection is inclusive at the start and exclusive at the end. Multiple cells
are stacked with time using EMC2's existing machinery and can be restored
with ``model.unstack_time_lat_lon()``. Cell IDs and native aerosol/mesh fields
are retained. Select cells before running on a large mesh to limit memory use.

``variable_names={'nc': 'qnc'}`` explicitly maps alternate field names.
``unit_overrides={'qnc': 'cm^-3'}`` overrides metadata by actual field name.
Compatible units are converted, including volume number concentrations;
these are not multiplied by density again. Missing hydrometeor mass fields
are errors: provide an explicit zero field for a known absent species.
Nonfinite thermodynamics and positive mass with missing/nonpositive number
concentration are rejected. Small or negative hydrometeor mass is truncated
to zero, with its derived number, fraction, and radius also zeroed.

For non-aerosol-aware Thompson, use ``aerosol_aware=False``. When ``nc`` is
absent, an explicit ``cloud_number`` in cm-3 is required. The adapter does not
guess a land/ocean droplet concentration. Aerosols are not required to compute
hydrometeor optics once prognostic ``nc`` and ``ni`` are provided; aerosol
activation, emissions, and direct aerosol scattering are outside this adapter.

Microphysics and optical assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Liquid density is 1000 kg/m3; cloud ice is 890 kg/m3 and graupel 500 kg/m3.
  Snow uses the Thompson mass law m(D) = 0.069 D**2 (SI) in its PSD diagnostics.
  The 100 kg/m3 snow metadata is used only for EMC2's approximate bulk optics.
* Cloud droplets use the integer Thompson shape parameter
  min(15, NINT(1e9/Nc) + 2), with Nc in m-3. Rain and cloud ice use exponential
  distributions constrained by their provided mass and number moments.
* Snow uses the Field et al. two-component distribution, not an exponential
  fit. Its reference second moment is mass content / 0.069; the third moment
  depends on temperature and mass content through the source coefficients.
  Its total number is diagnosed by integrating this distribution. Radar,
  lidar, and both spectral-width paths evaluate both components.
* Graupel number is diagnosed from mass and the Thompson top-down intercept
  calculation (bounded between 1e4 and 3e6 m-4), using local rain and temperature.
  This reconstruction uses instantaneous saved fields; it cannot reproduce
  the history of earlier microphysics substeps. The adapter does not rerun
  microphysics tendencies or adjust prognostic moments with in-step size limiters.
* Fall speeds use a D**b exp(-f D), including rain and snow exponential damping,
  and the Thompson square-root reference-density correction. These are
  reflectivity-weighted sedimentation velocities; resolved vertical wind is
  not added as a Doppler shift.
* Missing cloud/ice/snow radiation radii are diagnosed using the source moment
  relationships and radius bounds. Rain and graupel radii use their exponential
  PSD moments. Supplied radii do not change the microphysics PSDs.
* ``use_rad_logic=False`` selects Thompson PSD integration. The default EMC2
  radiation path, ``use_rad_logic=True``, uses effective radii and the existing
  ModelE tables. Neither path supplies new Thompson-specific scattering tables:
  Mie ice spheres or the existing nonspherical ice tables are approximations,
  especially for snow's size-dependent mass/density. The PSD and fall-speed
  implementation is not a claim of reproducing native MPAS radar reflectivity.

Sources and upstream comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checked against ARM-DOE/EMC2 master ``dc3f6d8e6bd62db8e910781272d2b7435431ce64``;
the starting local checkout and its model/PSD modules matched that revision.
No existing adapter's scheme dispatch or default parameters are changed.

* `MPAS Registry (field names, units and dimensions)
  <https://github.com/MPAS-Dev/MPAS-Model/blob/f34984b2c89353cf31c235b08d3b4acd96b37bb3/src/core_atmosphere/Registry.xml>`_
* `MPAS Thompson source (PSD, radii, graupel diagnostics and velocities)
  <https://github.com/MPAS-Dev/MPAS-Model/blob/f34984b2c89353cf31c235b08d3b4acd96b37bb3/src/core_atmosphere/physics/physics_wrf/module_mp_thompson.F>`_
* `MPAS thermodynamic constants
  <https://github.com/MPAS-Dev/MPAS-Model/blob/f34984b2c89353cf31c235b08d3b4acd96b37bb3/src/framework/mpas_constants.F>`_
* `Thompson and Eidhammer (2014), aerosol-aware extension
  <https://doi.org/10.1175/JAS-D-13-0305.1>`_
* `Thompson et al. (2008), snow microphysics
  <https://doi.org/10.1175/2008MWR2387.1>`_
* `Current WRF Thompson implementation (a distinct, evolving variant)
  <https://github.com/WRF-model/WRF/blob/master/phys/module_mp_thompson.F>`_

Run ``python -m unittest discover -s tests -p test_mpas.py`` or
``pytest tests/test_mpas.py tests/test_model.py tests/test_psd.py`` in an EMC2
environment with its declared dependencies. Fixtures exercise native NetCDF
input, units, geometry, selections, error handling, PSD moment closure, and
serial radar/lidar integration without downloading model output.

TEMPO mode
----------

TEMPO is selected explicitly; existing Thompson calls retain their behavior::

    model = emc2.core.model.MPAS(
        "history.nc", mcphys_scheme="TEMPO", cell_indices=[100],
        hail_aware=True, unit_overrides={"volg": "L/kg"})
    model = emc2.simulator.main.make_simulated_data(
        model, emc2.core.instruments.KAZR("nsa"),
        N_columns=1, use_rad_logic=False, parallel=False)

The implementation targets `NCAR/TEMPO revision 17c952b
<https://github.com/NCAR/TEMPO/tree/17c952bdfc0adcd1059aa404b80a34397a775b3c>`_.
The revision is stored in dataset attributes. Aerosol-aware behavior still
requires ``nc``; the adapter does not rerun aerosol activation or machine-learning
droplet diagnostics. ``hail_aware`` defaults to True for TEMPO and False for
the original Thompson mode. Set it to match the producing simulation.

Hail-aware inputs and volume units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TEMPO uses the same five saved hydrometeor categories, plus required ``ng``
(graupel/hail number per kg dry air) and ``volg`` (volume per kg dry air) in
hail-aware mode. Its internal hail-hyperaware split is recombined into
``qg/ng/volg`` before output. No independent prognostic ``qh`` or ``nh`` is
assumed; in the inspected UFS-MPAS registry those fields belong to NSSL.

**Explicit volume units are required.** TEMPO's calculations use L/kg, while
the inspected UFS-MPAS registry labels ``volg`` as m3/kg and its physics
interface passes the values directly. The adapter therefore requires
``unit_overrides`` for the actual volume field, rather than trusting this
potentially inconsistent metadata. Use ``"L/kg"`` for values following that
TEMPO convention, or ``"m^3/kg"`` for genuinely SI volume values. For an
alternate field name, use e.g. ``variable_names={"volg": "qb"}`` together
with ``unit_overrides={"qb": "L/kg"}``.

The density qg/volume is clipped to the source's 50–800 kg/m3 range. Both
continuous density and the source's discrete density bins are saved. The
continuous density controls the graupel fall-speed relation; the discrete
density controls the PSD mass law. Positive graupel mass requires positive,
finite ``ng`` and volume. Clear cells may contain zero number and volume.

``hail_aware=False`` requires neither ``ng`` nor ``volg``. This mode diagnoses
graupel using TEMPO's local single-moment intercept relation and its 1e2–1e6
m-4 bounds, which differ from the older Thompson intercept calculation.

TEMPO-specific physics and limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Hail-aware graupel uses the supplied number and the binned density to
  reconstruct its exponential PSD, conserving the saved mass and number.
* The graupel fall-speed coefficient depends on continuous particle density;
  the square-root air-density correction uses TEMPO's reference density.
  Cloud ice uses TEMPO's 1493.9 coefficient rather than the older 1847.5.
* Snow uses TEMPO's ``snow_moments`` inversion and the Field two-component
  distribution. Cloud, rain, and radiation-radius relationships retain the
  shared Thompson formulas. Missing air density uses TEMPO's R_d/R_v ratio.
* Radar, lidar, both spectral-width paths, and the approximate radiation
  paths are supported. Existing optical tables remain approximate for
  variable-density graupel/hail and snow. Wet-hail dielectric effects and
  the transient internal graupel/hail split are not reconstructed from
  saved combined moments. Native TEMPO reflectivity is therefore not a
  validation target for these approximate optics.

Implementation references:

* `TEMPO parameters and coefficients
  <https://github.com/NCAR/TEMPO/blob/17c952bdfc0adcd1059aa404b80a34397a775b3c/src/module_mp_tempo_params.F90>`_
* `TEMPO graupel checks, fall speeds and hail recombination
  <https://github.com/NCAR/TEMPO/blob/17c952bdfc0adcd1059aa404b80a34397a775b3c/src/module_mp_tempo_main.F90>`_
* `TEMPO snow moments
  <https://github.com/NCAR/TEMPO/blob/17c952bdfc0adcd1059aa404b80a34397a775b3c/src/module_mp_tempo_utils.F90>`_
* `UFS-MPAS registry
  <https://github.com/ufs-community/MPAS-Model/blob/noaa/develop/src/core_atmosphere/Registry.xml>`_
  and `physics interface
  <https://github.com/ufs-community/MPAS-Model/blob/noaa/develop/src/core_atmosphere/physics/mpas_atmphys_interface.F>`_

Tests: ``python -m unittest discover -s tests -p 'test_tempo.py'`` or
``pytest tests/test_tempo.py tests/test_mpas.py tests/test_model.py tests/test_psd.py``.

Defining other model adapters
----------------------------

EMC^2 uses the :py:mod:`emc2.core.Model` object to know what fields to load from a given
file in order to obtain the required information. In order to define your
:py:mod:`emc2.core.Model` object, it is highly recommended to use class inheritance. For example::

$ class ModelE(Model):
$    def __init__(self, file_path):
$        """
$        This loads a ModelE simulation with all of the necessary
$        parameters for EMC^2 to run.
$
$        Parameters
$        ----------
$        file_path: str
$            Path to a ModelE simulation.
$        """


In particular, EMC^2 will require information about the mixing ratio and
the number concentration of four
different species: cloud liquid (*cl*), cloud ice (*cl*), precipitating liquid (*pl*),
and precipitating ice (*pi*). These precipitation classes are commonly used
in many models in order to represent the cloud microphysical properties. EMC^2
derives the radar and lidar parameters for all 4 of these species. First,
in order to specify the fields that EMC^2 needs to look for, certain entries
whose keys correspond to the names of these four precipitation species must
be specified::

$   # Names of mixing ratios of species
$   self.q_names = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
$   # Number concentration of each species
$   self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
$   # Convective fraction
$   self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
$   # Stratiform fraction
$   self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
$   # Effective radius
$   self.re_fields = {'cl': 're_mccl', 'ci': 're_mcci', 'pi': 're_mcpi', 'pl': 're_mcpl'}
$   # Convective mixing ratio
$   self.q_names_convective = {'cl': 'QCLmc', 'ci': 'QCImc', 'pl': 'QPLmc', 'pi': 'QPImc'}
$   # Stratiform mixing ratio
$   self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}

In addition, other fields must also be specified::

$   # Water vapor mixing ratio
$   self.q_field = "q"
$   # Pressure
$   self.p_field = "p_3d"
$   # Height
$   self.z_field = "z"
$   # Temperature
$   self.T_field = "t"
$   # Name of height dimension
$   self.height_dim = "plm"
$   # Name of time dimension
$   self.time_dim = "time"

What if your model does not produce output for all 4 species, as is common
in many GCMs? Simply place in zero arrays that are the same shape as your
model fields!

Finally, we need to load in the model dataset. In order to do this, the model
must be loaded into a format that is compatible with **xarray**. If you are loading
a netCDF file, this is quite easy to do as **xarray** has native support for
netCDF files. For example, ModelE's files are in netCDF format, so we can simply do::

$   self.ds = xr.open_dataset(file_path)

One thing to be aware of is that you must ensure that all of your fields that
you load are 64-bit double precision floating point numbers in order to conform
with the assumed data types in ECM^2. Otherwise, underflow
errors are likely for many of the calculations. To ensure that this is the case,
one can simply loop over the variables in the file like this::

$   super().prepare_variables()

Finally, there are many assumptions that go into the calculation of the forward
modelled radar moments. For example, there are various fall-speed relationships
of the form :math:`V = aD^b` for different types of particles. Therefore, if you
are looking at a case where you think a specific kind of ice species may be
dominant, it is important to adjust these :math:`a` and :math:`b` constants. There
are numerous papers on this subject that are included in the references below.
It is best to match these coefficients with what is used in your model for the
best comparison. In order to adjust the constants that are used in the various
routines in EMC^2, you would have to fill in these dictionaries for each
hydrometeor species::

$   # Bulk density
$   self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m**3),
$                   'ci': 500. * ureg.kg / (ureg.m**3),
$                   'pl': 1000. * ureg.kg / (ureg.m**3),
$                   'pi': 250. * ureg.kg / (ureg.m**3)}
$   # Lidar ratio
$   self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
$                       'ci': 24. * ureg.dimensionless,
$                       'pl': 5.5 * ureg.dimensionless,
$                       'pi': 24.0 * ureg.dimensionless}
$   # Lidar LDR per hydrometeor mass content
$   self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m**3)),
$                       'ci': 0.35 * 1 / (ureg.kg / (ureg.m**3)),
$                       'pl': 0.1 * 1 / (ureg.kg / (ureg.m**3)),
$                       'pi': 0.40 * 1 / (ureg.kg / (ureg.m**3))}
$   # a, b in V = aD^b
$   self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
$   self.vel_param_b = {'cl': 2. * ureg.dimensionless,
$                       'ci': 1. * ureg.dimensionless,
$                       'pl': 0.8 * ureg.dimensionless,
$                       'pi': 0.41 * ureg.dimensionless}
$   super()._add_vel_units()


++++++++++
References
++++++++++
Locatelli, J. D., and Hobbs, P. V. (1974), Fall speeds and masses
of solid precipitation particles, J. Geophys. Res., 79( 15), 2185– 2197,
doi:10.1029/JC079i015p02185.

Brown, P.R. and P.N. Francis, 1995: Improved Measurements of the Ice
Water Content in Cirrus Using a Total-Water Probe.
J. Atmos. Oceanic Technol., 12, 410–414,
https://doi.org/10.1175/1520-0426(1995)012<0410:IMOTIW>2.0.CO;2

Heymsfield, A.J., G. van Zadelhoff, D.P. Donovan, F. Fabry, R.J. Hogan,
and A.J. Illingworth, 2007: Refinements to Ice Particle Mass Dimensional
and Terminal Velocity Relationships for Ice Clouds. Part II: Evaluation
and Parameterizations of Ensemble Ice Particle Sedimentation Velocities.
J. Atmos. Sci., 64, 1068–1088, https://doi.org/10.1175/JAS3900.1