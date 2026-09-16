"""Small native MPAS fixtures; no model output downloads required."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from scipy.special import gamma

from emc2.core.model import MPAS
from emc2.core.thompson import cloud_shape, snow_parameters, snow_distribution
from emc2.simulator.psd import calc_and_set_psd_params


def mpas_dataset():
    dims = ('Time', 'nCells', 'nVertLevels')
    shape = (2, 2, 3)
    ds = xr.Dataset(coords={'Time': np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[ns]'),
                            'nCells': [10, 20], 'nVertLevels': [1, 2, 3],
                            'nVertLevelsP1': [10, 20, 30, 40]})
    values = dict(qv=.004, qc=1.e-4, qi=1.e-5, qr=2.e-4, qs=3.e-5,
                  qg=4.e-5, nc=1.e8, ni=1.e5, nr=1.e6,
                  nwfa=3.e8, nifa=1.e6, pressure=80000., temperature=260., rho=.8)
    for name, value in values.items():
        ds[name] = xr.DataArray(np.full(shape, value), dims=dims)
        ds[name].attrs['units'] = ('nb kg^{-1}' if name.startswith('n') else
                                 'kg kg^{-1}' if name.startswith('q') else
                                 dict(pressure='Pa', temperature='K', rho='kg m^{-3}')[name])
    ds['zgrid'] = (('nCells', 'nVertLevelsP1'), [[0., 100., 300., 600.], [20., 120., 320., 620.]])
    ds.zgrid.attrs['units'] = 'm MSL'
    return ds


class TestMPAS(unittest.TestCase):
    def test_mappings_units_and_input_unchanged(self):
        source = mpas_dataset()
        original = source.copy(deep=True)
        m = MPAS(source)
        self.assertEqual(m.hydrometeor_classes, ['cl', 'ci', 'pl', 'pi', 'gr'])
        self.assertEqual(m.ice_hyd_types, ['ci', 'pi', 'gr'])
        self.assertFalse(m.process_conv)
        self.assertEqual(m.ds[m.T_field].dims, (m.time_dim, m.height_dim))
        np.testing.assert_allclose(m.ds[m.p_field], 800.)
        np.testing.assert_allclose(m.ds[m.N_field['cl']], 80.)
        np.testing.assert_allclose(m.ds[m.q_names_stratiform['gr']], 4.e-5)
        np.testing.assert_allclose(m.ds[m.z_field][0], [50., 200., 450.])
        self.assertIn('nwfa', m.ds)
        xr.testing.assert_identical(source, original)

    def test_single_cell_and_time_selection(self):
        m = MPAS(mpas_dataset(), cell_indices=1, time_range=('2020-01-02', '2020-01-03'))
        self.assertEqual(m.time_dim, 'Time')
        self.assertEqual(m.ds[m.T_field].shape, (1, 3))
        np.testing.assert_allclose(m.ds[m.z_field][0], [70., 220., 470.])

    def test_unstack(self):
        m = MPAS(mpas_dataset())
        m.unstack_time_lat_lon(squeeze_single_dims=False)
        self.assertEqual(m.ds.sizes['Time'], 2)
        self.assertEqual(m.ds.sizes['nCells'], 2)
        np.testing.assert_allclose(m.ds[m.N_field['ci']], .08)

    def test_density_and_temperature_fallback(self):
        ds = mpas_dataset().drop_vars(['rho', 'temperature'])
        ds['theta'] = xr.full_like(ds.qv, 280.)
        ds.theta.attrs['units'] = 'K'
        m = MPAS(ds)
        t = 280.*.8**(2./7.)
        rho = 80000./(287.04*t*(1.+.004/.622))
        np.testing.assert_allclose(m.ds[m.T_field], t)
        np.testing.assert_allclose(m.ds.rho_a, rho)
        np.testing.assert_allclose(m.ds[m.N_field['cl']], 100.*rho)

    def test_explicit_alias_and_volume_number(self):
        ds = mpas_dataset().rename(nc='droplets')
        ds['droplets'][:] = 80.e6
        ds.droplets.attrs['units'] = 'm^-3'
        m = MPAS(ds, variable_names={'nc': 'droplets'})
        np.testing.assert_allclose(m.ds[m.N_field['cl']], 80.)

    def test_moist_potential_temperature(self):
        ds = mpas_dataset().drop_vars('temperature')
        ds['theta_m'] = xr.full_like(ds.qv, 280.*(1.+.004*461.6/287.))
        ds.theta_m.attrs['units'] = 'K'
        m = MPAS(ds)
        np.testing.assert_allclose(m.ds[m.T_field], 280.*.8**(2./7.))

    def test_pressure_split_and_unit_override(self):
        ds = mpas_dataset().drop_vars('pressure')
        ds['pressure_base'] = xr.full_like(ds.qv, 70000.)
        ds['pressure_p'] = xr.full_like(ds.qv, 10000.)
        m = MPAS(ds, unit_overrides={'pressure_base': 'Pa', 'pressure_p': 'Pa'})
        np.testing.assert_allclose(m.ds[m.p_field], 800.)

    def test_native_effective_radius(self):
        ds = mpas_dataset()
        ds['re_cloud'] = xr.full_like(ds.qv, 12.e-6)
        ds.re_cloud.attrs['units'] = 'm'
        m = MPAS(ds)
        np.testing.assert_allclose(m.ds[m.strat_re_fields['cl']], 12.)

    def test_missing_number_fails_and_prescribed_is_explicit(self):
        ds = mpas_dataset().drop_vars('nc')
        with self.assertRaisesRegex(ValueError, 'nc'):
            MPAS(ds)
        with self.assertRaises(ValueError):
            MPAS(ds, aerosol_aware=False)
        m = MPAS(ds, aerosol_aware=False, cloud_number=150.)
        np.testing.assert_allclose(m.ds[m.N_field['cl']], 150.)

    def test_bad_units_dimensions_and_heights(self):
        for field, value in [('pressure', 'K'), ('nc', 'kg/kg')]:
            ds = mpas_dataset()
            ds[field].attrs['units'] = value
            with self.assertRaises(ValueError):
                MPAS(ds)
        ds = mpas_dataset()
        ds['zgrid'][:] = 0.
        with self.assertRaisesRegex(ValueError, 'heights'):
            MPAS(ds)
        ds = mpas_dataset().rename(nVertLevels='other')
        with self.assertRaisesRegex(ValueError, 'nVertLevels'):
            MPAS(ds)

    def test_cloud_number_validation_and_clear_air(self):
        ds = mpas_dataset()
        ds.nc[:] = 0.
        with self.assertRaisesRegex(ValueError, 'nc'):
            MPAS(ds)
        ds.qc[:] = -1.e-16
        m = MPAS(ds)
        np.testing.assert_array_equal(m.ds[m.N_field['cl']], 0.)
        np.testing.assert_array_equal(m.ds[m.strat_frac_names['cl']], 0.)
        np.testing.assert_array_equal(m.ds[m.strat_re_fields['cl']], 0.)

    def test_xtime_and_netcdf(self):
        ds = mpas_dataset().drop_vars('Time')
        ds['xtime'] = ('Time', np.array(['2020-01-01_00:00:00', '2020-01-02_00:00:00'], dtype='S19'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mpas.nc'
            ds.to_netcdf(path)
            m = MPAS(path, cell_indices=[0], time_range=('2020-01-01', '2020-01-02'))
        self.assertEqual(m.ds.sizes['Time'], 1)
        self.assertTrue(np.issubdtype(m.ds.Time.dtype, np.datetime64))

    def test_psd_mass_and_number(self):
        m = MPAS(mpas_dataset(), cell_indices=[0])
        for h in m.hyd_types:
            m.ds[f'strat_q_subcolumns_{h}'] = m.ds[m.q_names_stratiform[h]].expand_dims(subcolumn=[0])
            m.ds[f'strat_n_subcolumns_{h}'] = m.ds[m.N_field[h]].expand_dims(subcolumn=[0])
            fit = calc_and_set_psd_params(m, h)
            n0, lam, mu = fit.N_0, fit['lambda'], fit.mu
            if h == 'pi':
                diam = np.geomspace(1.e-9, .1, 20000)
                nd = snow_distribution(diam, float(n0[0, 0, 0]), float(lam[0, 0, 0]))
                mass = trapezoid(.069*diam**2*nd, x=diam)
                self.assertAlmostEqual(mass/(3.e-5*.8), 1., delta=.003)
            else:
                mass = np.pi*m.Rho_hyd[h].magnitude/6.*n0*gamma(mu+4)/lam**(mu+4)
                np.testing.assert_allclose(mass, m.ds[f'strat_q_subcolumns_{h}']*.8)
                number = n0*gamma(mu+1)/lam**(mu+1)
                np.testing.assert_allclose(number, m.ds[f'strat_n_subcolumns_{h}']*1.e6)

    def test_shape_and_fall_speed(self):
        np.testing.assert_array_equal(cloud_shape(np.array([50.e6, 100.e6, 200.e6, 400.e6])),
                                      [15., 12., 7., 5.])
        m = MPAS(mpas_dataset())
        diam = np.array([1.e-4, 1.e-3, 3.e-3])
        a, b, _ = m._get_terminal_velocity_params('pl', diam)
        np.testing.assert_allclose(a*diam**b, 4854.*diam*np.exp(-195.*diam))

    def test_clear_psd(self):
        ds = mpas_dataset()
        for key in ('qc', 'qi', 'qr', 'qs', 'qg'):
            ds[key][:] = 0.
        m = MPAS(ds, cell_indices=[0])
        for h in m.hyd_types:
            m.ds[f'strat_q_subcolumns_{h}'] = m.ds[m.q_names_stratiform[h]].expand_dims(subcolumn=[0])
            m.ds[f'strat_n_subcolumns_{h}'] = xr.full_like(m.ds[f'strat_q_subcolumns_{h}'], np.nan)
            fit = calc_and_set_psd_params(m, h)
            np.testing.assert_array_equal(fit.N_0, 0.)
            self.assertTrue(np.isfinite(fit.to_array()).all())


class TestMPASIntegration(unittest.TestCase):
    def test_serial_radar_and_lidar(self):
        from emc2.core.instruments import KAZR, HSRL
        from emc2.simulator.main import make_simulated_data
        for instrument in (KAZR('nsa'), HSRL()):
            for radiation in (False, True):
                with self.subTest(instrument=instrument.instrument_class, radiation=radiation):
                    ds = mpas_dataset()
                    ds.pressure[:] = [90000., 80000., 70000.]
                    m = MPAS(ds, cell_indices=[0])
                    make_simulated_data(m, instrument, 1, parallel=False, use_rad_logic=radiation)
                    field = ('sub_col_Ze_tot_strat' if instrument.instrument_class == 'radar'
                             else 'sub_col_beta_p_tot_strat')
                    self.assertTrue(np.isfinite(m.ds[field]).any())
                    for h in m.hyd_types:
                        prefix = 'sub_col_Ze_' if instrument.instrument_class == 'radar' else 'sub_col_beta_p_'
                        self.assertTrue(np.isfinite(m.ds[prefix+h+'_strat']).any())

    def test_two_pass_spectral_width(self):
        from emc2.core.instruments import KAZR
        from emc2.simulator.main import make_simulated_data
        ds = mpas_dataset()
        ds.pressure[:] = [90000., 80000., 70000.]
        m = MPAS(ds, cell_indices=[0])
        make_simulated_data(m, KAZR('nsa'), 1, parallel=False, use_rad_logic=False,
                            single_pass_spectral_width=False)
        self.assertTrue(np.isfinite(m.ds.sub_col_sigma_d_tot_strat).any())
        one_pass = MPAS(ds, cell_indices=[0])
        make_simulated_data(one_pass, KAZR('nsa'), 1, parallel=False, use_rad_logic=False)
        np.testing.assert_allclose(m.ds.sub_col_sigma_d_tot_strat,
                                   one_pass.ds.sub_col_sigma_d_tot_strat, rtol=1.e-6)


if __name__ == '__main__':
    unittest.main()
