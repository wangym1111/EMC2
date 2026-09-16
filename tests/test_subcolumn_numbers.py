"""Regression tests for clear-air number concentrations in subcolumns."""
import unittest

import numpy as np
import xarray as xr

from emc2.core.model import Model, MPAS
from emc2.simulator.subcolumn import set_q_n


class TestSubcolumnNumbers(unittest.TestCase):
    def test_zero_fraction_and_cloudy_number_scaling(self):
        for columns in (1, 2):
            for radiation in (False, True):
                with self.subTest(columns=columns, radiation=radiation):
                    model = Model()
                    model.ds = xr.Dataset(coords={'time': [0], 'height': [0, 1, 2, 3]})
                    model.num_subcolumns = columns
                    model.N_field = {'cl': 'nc'}
                    model.q_names_stratiform = {'cl': 'qc'}
                    model.strat_frac_names = {'cl': 'fraction'}
                    model.strat_frac_names_for_rad = {'cl': 'fraction'}
                    # Clear cells include zero N and a leftover positive native N.
                    for key, values in {'nc': [0., 20., 5., 7.],
                                        'qc': [0., 1.e-4, 0., 1.e-4],
                                        'fraction': [0., .5, 0., 1.]}.items():
                        model.ds[key] = (('time', 'height'), [values])
                    mask = np.tile([False, True, False, True], (columns, 1, 1))
                    if columns > 1:
                        mask[1, 0, 1] = False
                    model.ds['strat_frac_subcolumns_cl'] = (
                        ('subcolumn', 'time', 'height'), mask)
                    set_q_n(model, 'cl', is_conv=False, use_rad_logic=radiation, parallel=False)
                    number = model.ds.strat_n_subcolumns_cl.values
                    expected = np.where(mask, np.array([0., 40., 0., 7.]), 0.)
                    np.testing.assert_array_equal(number, expected)
                    self.assertTrue(np.isfinite(number).all())
                    np.testing.assert_array_equal(model.ds.nc, [[0., 20., 5., 7.]])

    def test_mpas_clear_air_through_radar_and_lidar(self):
        from test_tempo import tempo_dataset
        from emc2.core.instruments import KAZR, HSRL
        from emc2.simulator.main import make_simulated_data

        for scheme in ('Thompson', 'TEMPO'):
            for instrument in (KAZR('nsa'), HSRL()):
                for radiation in (False, True):
                    with self.subTest(scheme=scheme, instrument=instrument.instrument_class,
                                      radiation=radiation):
                        ds = tempo_dataset()
                        for name in ('qc', 'qi', 'qr', 'qs', 'qg', 'nc', 'ni', 'nr', 'ng', 'volg'):
                            ds[name].values[..., 0] = 0.
                        model = MPAS(ds, cell_indices=0, mcphys_scheme=scheme,
                                     unit_overrides={'volg': 'L/kg'})
                        make_simulated_data(model, instrument, 1, use_rad_logic=radiation,
                                            parallel=False)
                        for hyd in model.hyd_types:
                            number = model.ds[f'strat_n_subcolumns_{hyd}']
                            self.assertTrue(bool(np.isfinite(number).all()), hyd)
                            np.testing.assert_array_equal(number.isel(nVertLevels=0), 0.)
                            np.testing.assert_allclose(number.values[0], model.ds[model.N_field[hyd]])
                        # These native numbers are /kg; .8 kg/m3 converts them to /cm3.
                        np.testing.assert_allclose(model.ds.strat_n_subcolumns_cl.values[..., 1:], 80.)
                        np.testing.assert_allclose(model.ds.strat_n_subcolumns_ci.values[..., 1:], .08)


if __name__ == '__main__':
    unittest.main()
