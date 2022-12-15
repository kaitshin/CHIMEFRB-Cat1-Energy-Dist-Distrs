import emcee
import numpy as np

from multiprocessing import Pool
from scipy import optimize


class Fitter:
    """Implements posterior and contains functionality for optimizing/sampling.

    Parameters:
    -----------
    model: instance of `model_objs.DistributionModel`
    fit_parameter_names: list of strings
        Which model parameters will be varied in the fit. Initial guesses for
        these parameters contained in `model`.
    catalog: instance of `model_objs.FRBPopulationData`
        Data to be fit
    detected_injections: instance of `model_objs.FRBPopulationData`
    injection_model: instance of `model_objs.DistributionModel`
        Model from which the `detected_injections` where drawn.
    bins: sequence of tuples
        Defines the fit space and how events are histogrammed into bins.
        Sequence of tuples of `(property_name, bin_array)` pairs.

    """

    def __init__(self, model, fit_parameter_names, catalog, detected_injections,
            injection_model, bins):

        self._starting_model = model.copy()
        self._working_model = model.copy()
        self._fit_parameter_names = fit_parameter_names
        self._catalog = catalog
        self._detected_injections = detected_injections
        self._inj_model_rate_at_detections = injection_model.survey_rate(detected_injections)
        if type(bins)==list:
            self._bins = bins
        elif type(bins)==dict:
            # making sure this is a dict of just snr_log_data_bins and dm_log_data_bins for the stoch model analysis
            assert np.all(np.sort(np.array(list(bins.keys()))) == np.array(['dm_log_data_bins', 'snr_log_data_bins']))
            mod_bins = [
                ('snr',bins['snr_log_data_bins']),
                ('dm',bins['dm_log_data_bins'])
            ]
            self._bins = mod_bins
        else:
            raise ValueError('bins format not recognized, should be list or dict')

        self._priors = {par_name : lambda value: 0 
                        for par_name in fit_parameter_names}

        # This never changes, so compute it now.
        self._catalog_histogram = self.histogram_events(catalog)

    def reset_working_model(self):
        self._working_model = self._starting_model.copy()

    @property
    def fit_parameter_names(self):
        return self._fit_parameter_names[:]

    @property
    def starting_parameters(self):
        return self._starting_model.get_parameters(self.fit_parameter_names)

    @property
    def current_parameters(self):
        return self._working_model.get_parameters(self.fit_parameter_names)

    def set_parameters(self, values):
        kwargs = {self.fit_parameter_names[ii]: values[ii] for ii in
                  range(len(self.fit_parameter_names))}
        self._working_model.set_parameters(**kwargs)

    def update_starting_model(self, **kwargs):
        self._starting_model.set_parameters(**kwargs)
        self.reset_working_model()

    def set_range_prior(self, **kwargs):
        for par_name, r in kwargs.items():
            self.set_prior(par_name, RangePrior(r[0], r[1]))

    def set_prior(self, par_name, log_prior_fun):
        """Set a prior on a parameter.

        Parameters
        ----------
        par_name : string
            From `self.fit_parameter_names`
        log_prior_fun: callable
            Returns a the log-prior when evaluated at the parameter value.
            Arbitrary offset.
        """
        if not par_name in self._fit_parameter_names:
            raise ValueError("Parameter %s is not a fit parameter." % par_name)
        self._priors[par_name] = log_prior_fun

    def detected_injections_weights(self):
        return (self._working_model.survey_rate(self._detected_injections) /
                self._inj_model_rate_at_detections)

    def histogram_events(self, events, weights=None):
        data = []
        bins_list = []
        for property_name, bin_array in self._bins:
            data.append(getattr(events, property_name))
            bins_list.append(bin_array)
        # XXX `histogramdd` has a couple of allowed call sequences that might
        # have different performance. To optimize later if this turns out to be
        # a bottle neck.
        hist_data, _ = np.histogramdd(data, bins_list, weights=weights)
        return hist_data

    def model_histogram(self):
        return self.histogram_events(self._detected_injections,
                                     self.detected_injections_weights())

    def log_prior(self):
        log_prior = 0
        for parname in self.fit_parameter_names:
            value = self._working_model.get_parameters(str(parname))
            log_prior += self._priors[parname](value)
        return log_prior

    def log_likelihood(self):
        mu = self.model_histogram()
        N = self._catalog_histogram

        assert not np.any(N[np.where(mu == 0)])
        # then we can replace the nans without worrying
        mu_no_zeros = mu.copy()
        mu_no_zeros[mu == 0] = 1
        log_mu = np.log(mu_no_zeros)

        out = np.sum(N * log_mu - mu)
        return out

    def log_posterior(self):
        log_prior = self.log_prior()
        if np.isinf(log_prior) or np.isneginf(log_prior):
            return log_prior

        return self.log_likelihood() + log_prior

    def maximize_posterior(self):
        def neg_log_posterior(pars):
            self.set_parameters(pars)
            out = -self.log_posterior()
            return out
        pars0 = self._starting_model.get_parameters(self.fit_parameter_names)
        result = optimize.minimize(neg_log_posterior, pars0, method='Nelder-Mead')#, options={'disp':True})

        print(f'Optimization terminated successfully? {result.success}')
        # Should print/return some diagnostic/convergence data too.
        return result.x

    def run_emcee(self, resx, nwalkers, nburnin, nsteps, progress=True, multiprocess=True):
        def log_posterior_fn(pars):
            self.set_parameters(pars)
            logpost = self.log_posterior()

            if np.isnan(logpost):
                raise ValueError('!!! These parameters are causing a nan log posterior!\n', pars)

            return logpost

        npar = len(resx)
        p0 = 0.01 * np.random.randn(nwalkers, npar) + resx

        global fitter_instance
        fitter_instance = self

        if multiprocess:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, npar,
                    _set_pars_eval_log_posterior, pool=pool)
                state = sampler.run_mcmc(p0, nburnin, progress=progress)
                sampler.reset()
                sampler.run_mcmc(state, nsteps, progress=progress)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, npar, _set_pars_eval_log_posterior)
            state = sampler.run_mcmc(p0, nburnin, progress=progress)
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=progress)


        samples = sampler.get_chain(flat=True)
        return samples, sampler


class RangePrior:

    def __init__(self, min_, max_):
        self._min = min_
        self._max = max_

    def __call__(self, value):
        if value < self._min or value > self._max:
            return -np.inf
        else:
            return 0


def _set_pars_eval_log_posterior(pars):
    fitter_instance.set_parameters(pars)
    return fitter_instance.log_posterior()
