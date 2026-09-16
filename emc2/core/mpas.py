"""MPAS-Atmosphere adapter for the bundled Thompson–Eidhammer scheme."""
import numpy as np
from re import sub
import xarray as xr
from scipy.special import gamma

from .model import Model
from .instrument import ureg
from .thompson import cloud_shape, snow_parameters, snow_number, graupel_intercept
from .tempo import (TEMPO_REVISION, tempo_graupel_density, tempo_graupel_intercept,
                    tempo_graupel_velocity)


class MPAS(Model):
    """Load native MPAS columns for EMC2 radar and lidar calculations.

    Parameters
    ----------
    file_path : path-like or xarray.Dataset
        Native output with Time, nCells and nVertLevels dimensions. A selected
        single column is also accepted. See the MPAS section of usage/model.
    time_range : pair of datetime-like, optional
        Inclusive start and exclusive end, decoded from xtime if necessary.
    cell_indices : integer or sequence of integers, optional
        Select mesh cells before loading and stacking.
    variable_names : dict, optional
        Map canonical MPAS names to actual input names, e.g. {'nc': 'qnc'}.
        No WRF-style names are guessed. Input units must remain meaningful.
    unit_overrides : dict, optional
        Input field name to unit string, for files with missing/wrong metadata.
    aerosol_aware : bool
        Default True requires prognostic nc. False requires nc or an explicit
        cloud_number (cm-3); aerosols never substitute for activated droplets.
    cloud_number : float, optional
        Prescribed droplets in cm-3, only for non-aerosol-aware Thompson.
    q_hyd_truncation_cutoff : float
        Nonnegative mixing-ratio threshold in kg kg-1 (default 1e-14).
    mcphys_scheme : {'Thompson', 'TEMPO'}
        Default preserves the bundled MPAS Thompson implementation.
    hail_aware : bool, optional
        TEMPO defaults to True, requiring ng and volg. False diagnoses graupel
        number with TEMPO's single-moment relation. The actual volume field
        must have an explicit unit_overrides entry because MPAS metadata can
        conflict with TEMPO's L/kg convention. Hail is saved within graupel.

    Notes
    -----
    This targets MPAS's module_mp_thompson.F with fixed-density graupel,
    unless mcphys_scheme='TEMPO' selects the saved-state TEMPO formulation.
    Native aerosol fields
    are retained but aerosol optical scattering/activation is not simulated.
    ModelE optical tables are an explicit approximation; no Thompson-specific
    optical tables are bundled with EMC2. Use use_rad_logic=False for the
    Thompson PSDs. Resolved cloud fractions are binary; use one subcolumn
    to avoid imposing extra subgrid cloud-water variability.
    """

    def __init__(self, file_path, time_range=None, cell_indices=None,
                 variable_names=None, unit_overrides=None, aerosol_aware=True,
                 cloud_number=None, q_hyd_truncation_cutoff=1.e-14,
                 mcphys_scheme='Thompson', hail_aware=None):
        super().__init__()
        if mcphys_scheme.lower() not in ('thompson', 'tempo'):
            raise ValueError('mcphys_scheme must be Thompson or TEMPO')
        tempo = mcphys_scheme.lower() == 'tempo'
        self.hail_aware = tempo if hail_aware is None else hail_aware
        if not isinstance(self.hail_aware, (bool, np.bool_)):
            raise ValueError('hail_aware must be boolean')
        if self.hail_aware and not tempo:
            raise ValueError('hail_aware requires mcphys_scheme="TEMPO"')
        if not np.isfinite(q_hyd_truncation_cutoff) or q_hyd_truncation_cutoff < 0:
            raise ValueError('q_hyd_truncation_cutoff must be finite and nonnegative')
        if cloud_number is not None and (aerosol_aware or
                not np.isfinite(cloud_number) or cloud_number <= 0):
            raise ValueError('cloud_number must be positive and requires aerosol_aware=False')
        names = dict(variable_names or {})
        units = dict(unit_overrides or {})
        self.model_name = 'MPAS'
        self.mcphys_scheme = 'TEMPO' if tempo else 'Thompson'
        self.rad_scheme_family = 'ModelE'
        self.process_conv = False
        self.time_dim, self.height_dim = 'Time', 'nVertLevels'
        self.lat_dim, self.lon_dim = 'nCells', 'unused_lon'
        self.hyd_types = ['cl', 'ci', 'pl', 'pi', 'gr']
        self.ice_hyd_types = ['ci', 'pi', 'gr']
        self.aerosol_aware = aerosol_aware
        self.q_field, self.T_field = 'mpas_qv', 'mpas_temperature'
        self.p_field, self.z_field = 'mpas_pressure', 'mpas_height'
        self.q_names = dict(zip(self.hyd_types, ('qc', 'qi', 'qr', 'qs', 'qg')))
        self.q_names = {h: names.get(v, v) for h, v in self.q_names.items()}
        self.aerosol_fields = {k: names.get(k, k) for k in ('nwfa', 'nifa')}
        density = dict(cl=1000., ci=890., pl=1000., pi=100., gr=500.)
        self.Rho_hyd = {h: v*ureg.kg/ureg.m**3 for h, v in density.items()}
        self.fluffy = {h: .5*ureg.dimensionless for h in self.ice_hyd_types}
        self.lidar_ratio = {h: v*ureg.dimensionless for h, v in
                            dict(cl=18., ci=24., pl=5.5, pi=24., gr=24.).items()}
        self.LDR_per_hyd = {h: v*ureg.m**3/ureg.kg for h, v in
                            dict(cl=.03, ci=.35, pl=.1, pi=.4, gr=.4).items()}
        self.vel_param_a = dict(cl=3.16946e7, ci=1847.5, pl=4854., pi=40., gr=442.)
        if tempo:
            self.vel_param_a['ci'] = 1493.9
        self.vel_param_b = {h: v*ureg.dimensionless for h, v in
                            dict(cl=2., ci=1., pl=1., pi=.55, gr=.89).items()}
        if self.hail_aware:
            # Reference-density metadata; radar evaluates the actual local density.
            self.vel_param_a['gr'] = float(tempo_graupel_velocity(1., 500.))
            self.vel_param_b['gr'] = (3.*.54698726-1.)*ureg.dimensionless
        self._add_vel_units()

        if isinstance(file_path, xr.Dataset):
            ds = file_path.copy(deep=False)
            if cell_indices is not None:
                ds = ds.isel(nCells=np.atleast_1d(cell_indices))
        else:
            with xr.open_dataset(file_path) as source:
                if cell_indices is not None:
                    source = source.isel(nCells=np.atleast_1d(cell_indices))
                ds = source.load()
        if 'Time' not in ds.dims:
            ds = ds.expand_dims(Time=[0])
        if 'nVertLevels' not in ds.dims:
            raise ValueError('MPAS requires the nVertLevels dimension')
        if 'xtime' in ds and not ('Time' in ds.coords and
                np.issubdtype(ds.Time.dtype, np.datetime64)):
            values = ds.xtime.values
            if values.ndim == 2:
                values = [b''.join(row) if row.dtype.kind == 'S' else ''.join(row)
                          for row in values]
            strings = [(v.decode() if isinstance(v, bytes) else str(v)).strip('\x00 ').replace('_', 'T')
                       for v in values]
            ds = ds.assign_coords(Time=np.asarray(strings, dtype='datetime64[ns]'))
        if time_range is not None:
            if 'Time' not in ds.coords or not np.issubdtype(ds.Time.dtype, np.datetime64):
                raise ValueError('time_range requires datetime Time or MPAS xtime')
            start, end = np.asarray(time_range, dtype='datetime64[ns]')
            ds = ds.isel(Time=(ds.Time >= start) & (ds.Time < end))
        if any(ds.sizes[d] == 0 for d in ('Time', 'nVertLevels')) or ds.sizes.get('nCells', 1) == 0:
            raise ValueError('MPAS selection contains no columns or levels')

        def field(key, target, default):
            name = names.get(key, key)
            if name not in ds:
                raise ValueError(f'Missing MPAS field {name!r} ({key})')
            arr = ds[name].astype('float64')
            unit = str(units.get(name, arr.attrs.get('units', default)))
            unit = unit.replace('nb', '').replace('#', '').replace('{', '').replace('}', '')
            unit = unit.replace('m MSL', 'm').strip()
            unit = sub(r'([A-Za-z]+)([-+]\d+)', r'\1**\2', unit)
            try:
                factor = (1.*ureg(unit)).to(target).magnitude
            except Exception as exc:
                raise ValueError(f'Invalid units {unit!r} for MPAS field {name!r}; expected {target}') from exc
            return arr * factor

        # Only mass-level fields enter the simulator; mesh metadata stays native.
        template = field('qv', 'dimensionless', 'kg/kg')
        expected = {'Time', 'nVertLevels'} | ({'nCells'} if 'nCells' in ds.dims else set())
        if set(template.dims) != expected:
            raise ValueError(f'qv dimensions must be {sorted(expected)}')
        order = [d for d in ('Time', 'nCells', 'nVertLevels') if d in expected]
        template = template.transpose(*order)

        def mass_grid(arr):
            if set(arr.dims) - expected or 'nVertLevels' not in arr.dims:
                raise ValueError('MPAS atmospheric fields must be on mass levels')
            return arr.broadcast_like(template).transpose(*order)

        self.ds = ds
        self.ds[self.q_field] = template
        if names.get('pressure', 'pressure') in ds:
            pressure = mass_grid(field('pressure', 'Pa', 'Pa'))
        else:
            pressure = mass_grid(field('pressure_base', 'Pa', 'Pa') +
                                 field('pressure_p', 'Pa', 'Pa'))
        if names.get('temperature', 'temperature') in ds:
            temperature = mass_grid(field('temperature', 'K', 'K'))
        else:
            if names.get('theta', 'theta') in ds:
                theta = mass_grid(field('theta', 'K', 'K'))
            else:
                theta = mass_grid(field('theta_m', 'K', 'K'))/(1.+template*461.6/287.)
            temperature = theta * (pressure/1.e5)**(2./7.)
        if names.get('rho', 'rho') in ds:
            rho = mass_grid(field('rho', 'kg/m**3', 'kg/m**3'))
        else:
            epsilon = 287.04/461.5 if tempo else .622
            rho = pressure/(287.04*temperature*(1.+template/epsilon))
        for label, value in (('pressure', pressure), ('temperature', temperature), ('density', rho)):
            if bool((~np.isfinite(value) | (value <= 0)).any()):
                raise ValueError(f'MPAS {label} must be finite and positive')
        if bool((~np.isfinite(template) | (template < 0)).any()):
            raise ValueError('MPAS qv must be finite and nonnegative')
        self.ds[self.p_field] = pressure*.01
        self.ds[self.T_field] = temperature
        self.ds['rho_a'] = rho
        z = field('zgrid', 'm', 'm')
        if 'nVertLevelsP1' not in z.dims or z.sizes['nVertLevelsP1'] != ds.sizes['nVertLevels']+1:
            raise ValueError('zgrid must have nVertLevels+1 interfaces on nVertLevelsP1')
        # Drop interface labels before averaging to prevent xarray label alignment.
        axis = z.get_axis_num('nVertLevelsP1')
        mid = .5*(np.take(z.values, range(ds.sizes['nVertLevels']), axis=axis) +
                  np.take(z.values, range(1, ds.sizes['nVertLevels']+1), axis=axis))
        zmid = xr.DataArray(mid, dims=[self.height_dim if d == 'nVertLevelsP1' else d for d in z.dims],
                           coords={d: ds[d] for d in z.dims if d != 'nVertLevelsP1' and d in ds.coords})
        if self.height_dim in ds.coords:
            zmid = zmid.assign_coords({self.height_dim: ds[self.height_dim]})
        self.ds[self.z_field] = mass_grid(zmid)
        if not bool(np.isfinite(self.ds[self.z_field]).all()):
            raise ValueError('MPAS heights must be finite')
        if not bool((self.ds[self.z_field].diff(self.height_dim) > 0).all()):
            raise ValueError('MPAS heights must increase from bottom to top')
        for key, unit in ((self.p_field, 'hPa'), (self.T_field, 'K'),
                          (self.q_field, 'kg/kg'), (self.z_field, 'm'), ('rho_a', 'kg/m**3')):
            self.ds[key].attrs = {'units': unit}
        self.ds['mpas_zero'] = xr.zeros_like(template)
        for h, native in zip(self.hyd_types, ('qc', 'qi', 'qr', 'qs', 'qg')):
            q = mass_grid(field(native, 'dimensionless', 'kg/kg'))
            if bool((~np.isfinite(q)).any()):
                raise ValueError(f'MPAS {native} contains nonfinite values')
            q = q.where(q > q_hyd_truncation_cutoff, 0.)
            self.q_names_stratiform[h] = f'mpas_q_{h}'
            self.ds[self.q_names_stratiform[h]] = q
            self.ds[self.q_names_stratiform[h]].attrs = {'units': 'kg/kg'}
            self.strat_frac_names[h] = f'mpas_frac_{h}'
            self.ds[self.strat_frac_names[h]] = (q > 0).astype(float)
            for mapping in (self.q_names_convective, self.conv_frac_names, self.conv_re_fields):
                mapping[h] = 'mpas_zero'
            self.N_field[h] = f'mpas_n_{h}'
            self.strat_re_fields[h] = f'mpas_re_{h}'
        self.conv_frac_names_for_rad = self.conv_frac_names.copy()
        self.strat_frac_names_for_rad = self.strat_frac_names.copy()
        number_fields = [('cl', 'nc'), ('ci', 'ni'), ('pl', 'nr')]
        if self.hail_aware:
            number_fields.append(('gr', 'ng'))
        for h, key in number_fields:
            if h == 'cl' and names.get(key, key) not in ds and not aerosol_aware and cloud_number is not None:
                number = xr.full_like(template, cloud_number)
            else:
                # Native number mixing ratios are per kg dry air; volume units
                # are accepted explicitly without multiplying by density twice.
                name = names.get(key, key)
                unit = str(units.get(name, ds[name].attrs.get('units', '1/kg'))) if name in ds else '1/kg'
                if 'kg' in unit:
                    number = mass_grid(field(key, '1/kg', '1/kg')) * rho * 1.e-6
                else:
                    number = mass_grid(field(key, '1/cm**3', '1/kg'))
            q = self.ds[self.q_names_stratiform[h]]
            if bool(((q > 0) & (~np.isfinite(number) | (number <= 0))).any()):
                raise ValueError(f'MPAS {key} must be positive and finite wherever {h} mass is present')
            self.ds[self.N_field[h]] = number.where(q > 0, 0.)
        content = {h: self.ds[self.q_names_stratiform[h]]*rho for h in self.hyd_types}
        amp, slope = snow_parameters(content['pi'], temperature, tempo=tempo)
        self.ds[self.N_field['pi']] = snow_number(amp, slope).where(content['pi'] > 0, 0.)*1.e-6
        if self.hail_aware:
            volume_name = names.get('volg', 'volg')
            if volume_name not in units:
                raise ValueError(f'TEMPO volume units are ambiguous: set unit_overrides[{volume_name!r}] '
                                 'explicitly to "L/kg" or "m^3/kg" for your producer')
            volume = mass_grid(field('volg', 'm**3/kg', 'L/kg'))
            qg = self.ds[self.q_names_stratiform['gr']]
            if bool(((qg > 0) & (~np.isfinite(volume) | (volume <= 0))).any()):
                raise ValueError('TEMPO volg must be positive and finite wherever graupel mass is present')
            continuous, binned = tempo_graupel_density(qg.values, volume.values)
            for key, value in (('mpas_graupel_density', continuous), ('mpas_graupel_psd_density', binned)):
                self.ds[key] = xr.DataArray(value, dims=template.dims, coords=template.coords,
                                            attrs={'units': 'kg/m**3'})
            density['gr'] = self.ds.mpas_graupel_psd_density
            self.Rho_hyd['gr'] = 'variable'
            self.variable_density['gr'] = 'mpas_graupel_psd_density'
        else:
            if tempo:
                n0g = tempo_graupel_intercept(content['gr'].values)
            else:
                n0g = graupel_intercept(content['gr'].values, content['pl'].values,
                                       self.ds[self.N_field['pl']].values*1.e6, temperature.values, rho.values)
            self.ds['mpas_graupel_n0'] = xr.DataArray(n0g, dims=template.dims, coords=template.coords)
            lamg = (np.pi*500.*self.ds.mpas_graupel_n0/content['gr'].where(content['gr'] > 0))**.25
            self.ds[self.N_field['gr']] = (self.ds.mpas_graupel_n0/lamg).fillna(0.)*1.e-6
        for h in self.hyd_types:
            self.ds[self.N_field[h]].attrs = {'units': 'cm^-3'}
            n = self.ds[self.N_field[h]]*1.e6
            mu = cloud_shape(n) if h == 'cl' else 0.
            lam = (np.pi*density[h]/6.*n*gamma(mu+4.) /
                   (content[h].where(content[h] > 0)*gamma(mu+1.)))**(1./3.)
            re = .5*(mu+3.)/lam
            if h == 'pi':
                re = (.5/slope).clip(5.01e-6, 999.e-6)
            elif h == 'cl':
                re = re.clip(2.51e-6, 50.e-6)
            elif h == 'ci':
                re = re.clip(2.51e-6, 125.e-6)
            native_re = {'cl': 're_cloud', 'ci': 're_ice', 'pi': 're_snow'}.get(h)
            if native_re and names.get(native_re, native_re) in ds:
                re = mass_grid(field(native_re, 'm', 'm'))
                if bool(((content[h] > 0) & (~np.isfinite(re) | (re <= 0))).any()):
                    raise ValueError(f'{native_re} must be positive where mass is present')
            self.ds[self.strat_re_fields[h]] = (re*1.e6).where(content[h] > 0, 0.)
            self.ds[self.strat_re_fields[h]].attrs = {'units': 'micron'}
        self.ds.attrs = dict(self.ds.attrs, emc2_mpas_microphysics='MPAS bundled Thompson–Eidhammer',
                             emc2_optics='ModelE lookup tables: approximate Thompson optics')
        if tempo:
            self.ds.attrs.update(emc2_mpas_microphysics='TEMPO saved-state PSD reconstruction',
                                 emc2_tempo_revision=TEMPO_REVISION,
                                 emc2_tempo_hail_aware=int(self.hail_aware))
        self.check_and_stack_time_lat_lon()

    def _get_terminal_velocity_params(self, hyd_type, p_diam_array=None):
        """Represent v=a D**b exp(-fD) through EMC2's array coefficient API."""
        if p_diam_array is None:
            return super()._get_terminal_velocity_params(hyd_type)
        damping = {'pl': 195., 'pi': 100.}.get(hyd_type, 0.)
        a = self.vel_param_a[hyd_type].magnitude*np.exp(-damping*p_diam_array)
        b = np.full_like(p_diam_array, self.vel_param_b[hyd_type].magnitude)
        return a, b, True
