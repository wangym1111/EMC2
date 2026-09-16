"""TEMPO saved-state fixtures and serial simulator regression checks."""
import unittest
import numpy as np
import xarray as xr
from scipy.special import gamma
from scipy.integrate import trapezoid

from emc2.core.model import MPAS
from emc2.core.tempo import (tempo_graupel_density, tempo_graupel_velocity,
                             TEMPO_REFERENCE_DENSITY)
from emc2.core.thompson import snow_distribution
from emc2.simulator.psd import calc_and_set_psd_params
from test_mpas import mpas_dataset


def tempo_dataset():
    ds = mpas_dataset()
    ds.pressure[:] = [90000., 80000., 70000.]
    ds['ng'] = xr.full_like(ds.qg, 2000.)
    ds.ng.attrs['units'] = 'nb kg^{-1}'
    ds['volg'] = ds.qg*1000./xr.DataArray([200., 400., 600.], dims='nVertLevels',
                                          coords={'nVertLevels': ds.nVertLevels})
    # Deliberately reproduce the misleading UFS-MPAS Registry metadata.
    ds.volg.attrs['units'] = 'm^{3} kg^{-1}'
    return ds


def tempo_model(ds=None, **kwargs):
    return MPAS(tempo_dataset() if ds is None else ds, mcphys_scheme='TEMPO',
                unit_overrides={'volg': 'L/kg'}, **kwargs)


class TestTEMPO(unittest.TestCase):
    def test_prognostic_number_volume_and_density(self):
        ds = tempo_dataset()
        original = ds.copy(deep=True)
        model = tempo_model(ds, cell_indices=[0])
        self.assertEqual(model.mcphys_scheme, 'TEMPO')
        self.assertEqual(model.hyd_types, ['cl', 'ci', 'pl', 'pi', 'gr'])
        self.assertNotIn('ha', model.N_field)
        self.assertTrue(model.hail_aware)
        self.assertEqual(model.Rho_hyd['gr'], 'variable')
        np.testing.assert_allclose(model.ds[model.N_field['gr']], .0016)
        np.testing.assert_allclose(model.ds.mpas_graupel_density[0], [200., 400., 600.])
        np.testing.assert_allclose(model.ds.mpas_graupel_psd_density, model.ds.mpas_graupel_density)
        self.assertNotIn('mpas_graupel_n0', model.ds)
        xr.testing.assert_identical(ds, original)

    def test_volume_units_are_explicit_and_convertible(self):
        with self.assertRaisesRegex(ValueError, 'volume units'):
            MPAS(tempo_dataset(), mcphys_scheme='TEMPO')
        ds = tempo_dataset()
        ds.volg[:] *= .001
        si = MPAS(ds, mcphys_scheme='TEMPO', unit_overrides={'volg': 'm^3/kg'})
        liter = tempo_model()
        np.testing.assert_allclose(si.ds.mpas_graupel_density, liter.ds.mpas_graupel_density)

    def test_density_bins_follow_fortran_rounding_and_clipping(self):
        expected = np.array([50., 349., 350., 800.])
        continuous, bins = tempo_graupel_density(np.ones(4), 1./np.array([10., 349., 350., 900.]))
        np.testing.assert_array_equal(continuous, expected)
        np.testing.assert_array_equal(bins, [100., 300., 400., 800.])

    def test_missing_or_invalid_prognostics(self):
        for name in ('ng', 'volg'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    tempo_model(tempo_dataset().drop_vars(name))
                ds = tempo_dataset()
                ds[name][:] = 0.
                with self.assertRaises(ValueError):
                    tempo_model(ds)
        with self.assertRaises(ValueError):
            MPAS(tempo_dataset(), mcphys_scheme='other')
        with self.assertRaises(ValueError):
            MPAS(tempo_dataset(), hail_aware=True)

    def test_single_moment_tempo_and_thompson_remain_distinct(self):
        ds = tempo_dataset().drop_vars(['ng', 'volg'])
        model = MPAS(ds, mcphys_scheme='TEMPO', hail_aware=False)
        content = 4.e-5*.8
        expected_n0 = 10.**(3.4+2./7.*(np.log10(content)+8.))
        np.testing.assert_allclose(model.ds.mpas_graupel_n0, expected_n0)
        reference = MPAS(ds)
        self.assertFalse(reference.hail_aware)
        self.assertEqual(reference.mcphys_scheme, 'Thompson')
        self.assertFalse(np.allclose(reference.ds.mpas_graupel_n0, model.ds.mpas_graupel_n0))

    def test_clear_graupel(self):
        ds = tempo_dataset()
        for name in ('qg', 'ng', 'volg'):
            ds[name][:] = 0.
        model = tempo_model(ds)
        np.testing.assert_array_equal(model.ds[model.N_field['gr']], 0.)
        np.testing.assert_array_equal(model.ds[model.strat_re_fields['gr']], 0.)
        self.assertTrue(np.isfinite(model.ds.mpas_graupel_density).all())

    def test_tempo_density_fallback_and_renamed_volume(self):
        ds = tempo_dataset().drop_vars('rho').rename(volg='qb')
        model = MPAS(ds, mcphys_scheme='TEMPO', variable_names={'volg': 'qb'},
                      unit_overrides={'qb': 'L/kg'}, cell_indices=[0])
        expected = np.array([90000., 80000., 70000.])/(287.04*260.*(1.+.004/(287.04/461.5)))
        np.testing.assert_allclose(model.ds.rho_a[0], expected)
        np.testing.assert_allclose(model.ds[model.N_field['gr']][0], 2000.*expected*1.e-6)

    def test_psd_moments_and_radii(self):
        model = tempo_model(cell_indices=[0])
        for h in model.hyd_types:
            q = model.ds[model.q_names_stratiform[h]].expand_dims(subcolumn=[0])
            n = model.ds[model.N_field[h]].expand_dims(subcolumn=[0])
            model.ds[f'strat_q_subcolumns_{h}'] = q
            model.ds[f'strat_n_subcolumns_{h}'] = n
            fit = calc_and_set_psd_params(model, h)
            if h == 'pi':
                diam = np.geomspace(1.e-9, .1, 20000)
                nd = snow_distribution(diam, float(fit.N_0[0, 0, 0]), float(fit['lambda'][0, 0, 0]))
                self.assertAlmostEqual(trapezoid(.069*diam**2*nd, x=diam)/(3.e-5*.8), 1., delta=.003)
                self.assertAlmostEqual(trapezoid(nd, x=diam)/float(n[0, 0, 0]*1.e6), 1., delta=1.e-4)
            else:
                density = (model.ds.mpas_graupel_psd_density if h == 'gr'
                           else model.Rho_hyd[h].magnitude)
                mass = np.pi*density/6.*fit.N_0*gamma(fit.mu+4)/fit['lambda']**(fit.mu+4)
                np.testing.assert_allclose(mass.transpose(*q.dims), q*.8)
                number = fit.N_0*gamma(fit.mu+1)/fit['lambda']**(fit.mu+1)
                np.testing.assert_allclose(number, n*1.e6)
                if h == 'gr':
                    radius = 1.5/fit['lambda']*1.e6
                    np.testing.assert_allclose(radius[0], model.ds[model.strat_re_fields[h]])

    def test_fall_speeds(self):
        d = np.array([.0001, .001, .01])
        v = tempo_graupel_velocity(d, 400.)
        expected = (4.*400.*9.8/(3.*.504843467198652*TEMPO_REFERENCE_DENSITY))**.54698726
        np.testing.assert_allclose(v, expected*d**.64096178)
        self.assertTrue(np.all(tempo_graupel_velocity(d, 800.) > v))
        model = tempo_model()
        a, b, _ = model._get_terminal_velocity_params('ci', d)
        np.testing.assert_allclose(a*d**b, 1493.9*d)


class TestTEMPOIntegration(unittest.TestCase):
    def test_serial_instruments_and_hail_modes(self):
        from emc2.core.instruments import KAZR, HSRL
        from emc2.simulator.main import make_simulated_data
        for instrument in (KAZR('nsa'), HSRL()):
            for hail in (True, False):
                for radiation in (False, True):
                    with self.subTest(instrument=instrument.instrument_class, hail=hail, radiation=radiation):
                        model = tempo_model(cell_indices=[0], hail_aware=hail)
                        make_simulated_data(model, instrument, 1, use_rad_logic=radiation, parallel=False)
                        prefix = 'sub_col_Ze_' if instrument.instrument_class == 'radar' else 'sub_col_beta_p_'
                        for h in model.hyd_types:
                            self.assertTrue(np.isfinite(model.ds[prefix+h+'_strat']).any())

    def test_two_pass_velocity_and_spectral_width(self):
        from emc2.core.instruments import KAZR
        from emc2.simulator.main import make_simulated_data
        models = []
        for single_pass in (True, False):
            m = tempo_model(cell_indices=[0])
            make_simulated_data(m, KAZR('nsa'), 1, use_rad_logic=False, parallel=False,
                                single_pass_spectral_width=single_pass)
            models.append(m)
        np.testing.assert_allclose(models[0].ds.sub_col_sigma_d_tot_strat,
                                   models[1].ds.sub_col_sigma_d_tot_strat, rtol=1.e-6)
        self.assertTrue((models[0].ds.sub_col_Vd_gr_strat < 0).all())


if __name__ == '__main__':
    unittest.main()
