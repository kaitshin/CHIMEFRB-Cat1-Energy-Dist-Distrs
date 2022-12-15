import astropy.units as u
import numpy as np
import pickle

from debias.fitting import Fitter
from debias.frb_population_data import FRBPopulationData
from debias.model_objects import get_model_inj, StochasticDMModel
from debias.utils import apply_cuts, get_cat_width

FILE_PATH = "/data/user-data/kshin/"


def get_stochmodel_from_fiducial(all_frbs_cut, det_frbs_cut, fid_filename, pdcosmicz_filename):
    """
    """
    with open(fid_filename, 'rb') as handle:
        model_fiducial = pickle.load(handle)

    params = dict(
        zip(list(model_fiducial.parameters)[:-1], 
            model_fiducial.get_parameters(list(model_fiducial.parameters)[:-1]))
    )
    del params['alpha']
    del params['norm']
    del params['flu_pivot']
    del params['dm_shape']
    del params['dm_loc']
    del params['dm_scale']
    del params['dm_uniform_min']
    del params['dm_uniform_max']
    del params['dm_uniform_fraction']

    pnames = np.array([
        'log_rate',
        'gamma',
        'log_energy_char',
        'n',
        'alpha',
        'mu_host',
        'sigma_host'
    ])
    init_guess = np.array([
        np.log(9e4),         # log_rate
        -1.16-1,             # gamma
        np.log(10**41.84),   # log_energy_char
        1.77,                # n
        -1.55,               # alpha
        2.16,                # mu_host
        0.51                 # sigma_host
    ])
    param_dict = dict(zip(pnames, init_guess))

    ## TODO(this should be redundant after running resampling with the surv_rate_factor)
    surv_rate_factor = (det_frbs_cut.inj_efficiency * len(all_frbs_cut)) / all_frbs_cut.f_sky
    param_dict['survey_rate_factor'] = surv_rate_factor
    ##
    params.update(param_dict)

    # log spacing to z=1.0
    test_model = StochasticDMModel(DM_max=6500.0, z_max=4.0, N_z=100,
        PDMcosmicz_grid_fname=pdcosmicz_filename,
        **params)

    return test_model


def main():
    cat_orig = FRBPopulationData.from_catalog(FILE_PATH+"catalog-2022.json")
    all_frbs = FRBPopulationData.from_injections_all(FILE_PATH+"200813_flat_alpha_base.h5")
    det_frbs = FRBPopulationData.from_injections_det(FILE_PATH+"200813_flat_alpha.p",
        FILE_PATH+"200813_flat_alpha_base.h5")

    print('applying cuts')
    (cat, cat_all_scat, all_frbs_cut, all_frbs_allscat, det_frbs_cut, det_frbs_allscat,
        bins) = apply_cuts(cat_orig, all_frbs, det_frbs)
    # update catalog numerical width values w/ the fully cut catalog
    cat.width = get_cat_width(cat, cat)

    print('creating DistributionModel for injections sample')
    model_inj = get_model_inj(all_frbs, det_frbs_cut)

    print('creating StochasticDMModel')
    stoch_model = get_stochmodel_from_fiducial(
        all_frbs_cut, det_frbs_cut,
        fid_filename=FILE_PATH+'model_1dfit_5Jyms_alpha_withrate.p',
        pdcosmicz_filename=FILE_PATH+'PDMcosmicz_grid0.001_4.0_100_100.0_6500.0_80.0_150.p'
    )
    # From Pragya
    SURVEY_LEN_DAYS = 219957. / 1024 * u.day
    # Compensate for sensitivity drift.
    SURVEY_LEN_DAYS = SURVEY_LEN_DAYS / (1 + 1.5 * 0.03)
    SURVEY_LEN_YEARS = SURVEY_LEN_DAYS.to(u.year).value
    stoch_model.set_parameters(survey_rate_factor=SURVEY_LEN_YEARS, E_pivot=10**39)

    print('initializing StochasticDMModel with initial guesses')
    pnames = np.array(['log_rate', 'gamma', 'log_energy_char', 'n', 'alpha', 'mu_host', 'sigma_host'])
    init_guess = np.array([11.64941239, -1.3907243,  95.09390747,  0.82453358, -0.69282951,  4.67746159, 0.24537635])
    pdict = dict(zip(pnames, init_guess))
    start_model = stoch_model.copy()
    start_model.set_parameters(**pdict)

    print('instantiating Fitter object')
    fitter = Fitter(start_model, pnames, cat, det_frbs_cut, model_inj, bins)
    range_prior_dict = {
        'log_rate': [-2, 15],
        'gamma': [-2.5, 2],
        'log_energy_char': [np.log(10**38.), np.log(10**49.)],
        'n': [-2, 8],
        'alpha': [-5, 5],
        'mu_host': [np.log(20), np.log(500)],
        'sigma_host': [0.1, 4.0]
    }
    fitter.set_range_prior(**range_prior_dict)

    print('getting fit result from scipy.optimize.minimize')
    resx = fitter.maximize_posterior()
    print(resx)


    print('\nstarting MCMC run')
    samples, sampler = fitter.run_emcee(resx, nwalkers=40, nburnin=2000, nsteps=25000)
    with open(FILE_PATH+'emcee_sampler_output.p', 'wb') as handle:
        pickle.dump(sampler, handle)

if __name__ == "__main__":
    main()
