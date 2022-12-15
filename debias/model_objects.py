import abc
import astropy.units as u
import mpmath
import numpy as np
import pickle

from astropy.constants import c
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from scipy.integrate import simps, trapz
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.stats import lognorm
from tqdm import tqdm


class BaseModel(abc.ABC):
    def __init__(self, **kwargs):
        npars = len(self.parameters)
        if 'prefix' in kwargs:
            prefix = kwargs.pop('prefix')
        else:
            prefix = ''
        params, kwargs = self._set_init_parameters(npars, prefix, **kwargs)

        # check that we consumed all input parameters
        if kwargs:
            raise TypeError("Received unexpected keyword arguments: " + str(kwargs.keys()))
        self._params = params

    def _set_init_parameters(self, npars, prefix, **kwargs):
        init_params = np.ones(npars)

        inds = []
        values = []
        prelen = len(prefix)
        for key in list(kwargs):
            if key[:prelen] == prefix:
                try:
                    ind = self.parameters.index(key[prelen:])
                except ValueError as e:
                    raise ValueError("Key `" + key[prelen:] + "` not in model parameters") from e
                inds.append(ind)
                values.append(kwargs.pop(key))
        inds = np.array(inds)
        values = np.array(values)

        init_params[inds] = values

        return init_params, kwargs

    def set_parameters(self, **kwargs):
        for key in list(kwargs):
            param_idx = self.parameters.index(key)
            self._params[param_idx] = kwargs.pop(key)

    def get_parameters(self, keys):
        if type(keys) == str: # only 1 param, not in list
            param_idx = self.parameters.index(keys)
            return self._params[param_idx]
        else:
            keys_list = keys
            params_list = []
            for keys in list(keys_list):
                param_idx = self.parameters.index(keys)
                params_list.append(self._params[param_idx])
            return params_list

    def zero_unif_frac(self):
        # only init_params have unif_frac != 0
        self.set_parameters(dm_uniform_fraction=0, width_uniform_fraction=0, scat_uniform_fraction=0)

    def copy(self):
        '''returns another object with the same _params, aka a deepcopy'''
        paramkwargs = dict(zip(self.parameters, self._params))
        copyobj = self.__class__(**paramkwargs)
        return copyobj

    def lognorm_model(self, ptype, data):
        unif_max  = self.get_parameters(f'{ptype}_uniform_max')
        unif_min  = self.get_parameters(f'{ptype}_uniform_min')
        unif_frac = self.get_parameters(f'{ptype}_uniform_fraction')

        unif = np.ones_like(data) / (unif_max - unif_min)
        unif[data>unif_max] = 0
        unif[data<unif_min] = 0

        lnorm = lognorm.pdf(data, s=self.get_parameters(f'{ptype}_shape'),
            loc=self.get_parameters(f'{ptype}_loc'), scale=self.get_parameters(f'{ptype}_scale'))

        lognorm_unif_distr = (1-unif_frac)*lnorm + unif_frac*unif
        return lognorm_unif_distr


    @abc.abstractmethod
    def probabilities(self, det_frbs_cut):
        """Overall probabilities for the model.
        Should call `probs` for the probabilities of each individual property.
        Return a np.array of length(det_frbs_cut)
        """
        return np.zeros_like(det_frbs_cut)

    @abc.abstractmethod
    def probs(self, ptype, ddata):
        """Probabilities of each individual property.
        Should be called by `probabilities`
        Return a np.array of length(ddata)
        """
        return np.zeros(len(ddata))

    @abc.abstractmethod
    def rate(self, det_frbs_cut):
        """Rate, depending on probabilities of each individual property.
        Return a np.arary of length(det_frbs_cut)
        """
        return np.zeros_like(det_frbs_cut)


class DistributionModel(BaseModel):
    parameters = (
        'norm',
        'alpha',
        'flu_pivot',
        'dm_shape',
        'dm_loc',
        'dm_scale',
        'dm_uniform_min',
        'dm_uniform_max',
        'dm_uniform_fraction',
        'width_shape',
        'width_loc',
        'width_scale',
        'width_uniform_min',
        'width_uniform_max',
        'width_uniform_fraction',
        'scat_shape',
        'scat_loc',
        'scat_scale',
        'scat_uniform_min',
        'scat_uniform_max',
        'scat_uniform_fraction',
        'survey_rate_factor',
    )

    def probabilities(self, det_frbs_cut):
        return self.probs('snr', det_frbs_cut.fluence) * self.probs('dm', det_frbs_cut.dm) * self.probs('width', det_frbs_cut.width) * self.probs('scat', det_frbs_cut.scat)

    def rate(self, det_frbs_cut):
        return self.get_parameters('norm') * self.probabilities(det_frbs_cut)

    def probs(self, ptype, ddata):
        if ptype == 'snr':
            return self.powerlaw_model(ddata)
        else:
            return self.lognorm_model(ptype, ddata)

    def survey_rate(self, det_frbs_cut):
        return self.get_parameters("survey_rate_factor") * self.rate(det_frbs_cut)

    def powerlaw_model(self, data):
        ''' this is the CDF '''
        flu_pivot, alpha = self.get_parameters(['flu_pivot', 'alpha'])
        # return (data/flu_pivot)**(alpha)
        return 1/flu_pivot * -alpha * (data/flu_pivot)**(alpha-1)


class StochasticDMModel(DistributionModel):
    parameters = (
        'log_rate',         # normalization
        'gamma',            # pivot index for schechter fn
        'log_energy_char',  # characteristic energy E_char for schechter fn
        'n',                # exponent for modeling SFR evoln
        'alpha',            # spectral index
        'mu_host',          # center of lognorm distr of DM'_host ("logmean") in log space
        'sigma_host',       # spread of lognorm distr of DM'_host ("logsigma") in log space
        'width_shape',
        'width_loc',
        'width_scale',
        'width_uniform_min',
        'width_uniform_max',
        'width_uniform_fraction',
        'scat_shape',
        'scat_loc',
        'scat_scale',
        'scat_uniform_min',
        'scat_uniform_max',
        'scat_uniform_fraction',
        'survey_rate_factor',
        'E_pivot'
    )

    def __init__(self, beta=3., F=0.31, z_min=1e-3, z_max=4.0, N_z=100,
                 DM_min=100.0, DM_max=6500.0, DM_local=80.0, N_DM=150,
                 rate_interp=False,
                 PDMcosmicz_grid_fname=None,
                 **kwargs):
        """
        initializing with DM = DM_EG + DM_L
        """
        npars = len(self.parameters)
        if 'prefix' in kwargs:
            prefix = kwargs.pop('prefix')
        else:
            prefix = ''
        params, kwargs = self._set_init_parameters(npars, prefix, **kwargs)
        # check that we consumed all input parameters
        if kwargs:
            raise TypeError("Received unexpected keyword arguments: " + str(kwargs.keys()))
        self._params = params

        self.rate_interp = rate_interp

        self.beta = beta
        self.F = F

        self._z_min = z_min
        self._z_max = z_max
        self._N_z = N_z

        self._zi = int(N_z*.68) # the index+1 where the array goes from log to lin binning (z=1.0, only for Nz=100)
        diff = np.diff(np.logspace(np.log10(z_min), 0, num=self._zi))[-1]
        self.z_arr = np.concatenate((
            np.logspace(np.log10(z_min), 0, num=self._zi),
            np.arange(1.0+diff, z_max+2*diff, diff)
        ))

        self.fluence_arr = np.logspace(0., 5.7, 200) # 1 ~ 5e5 Jy ms

        self._DM_min = DM_min
        self._DM_max = DM_max
        self._DM_local = DM_local
        self._N_DM = N_DM

        # DM = DM_EG + DM_L
        # DM_EG = DM - DM_L
        # ultimately DM_cosmic will be marginalized over, so we choose to define that the same as DM_EG
        self.DM_arr = np.linspace(DM_min, DM_max, N_DM)
        self.DM_EG_arr = self.DM_arr - self._DM_local
        self.DM_cosmic_arr = self.DM_arr - self._DM_local

        self._set_grid_params = False

        # loads in P(DM_cosmic | z) grid if it exists; otherwise, generates it
        if PDMcosmicz_grid_fname is None:
            raise ValueError("You should pass in a path for either loading or saving the appropriate P(DM_cosmic|z) grid.\
            Filename formatting should be f'PDMcosmicz_grid{z_min}_{z_max}_{N_z}_{DM_min}_{DM_max}_{DM_L}_{N_DM}'")

        assert f'PDMcosmicz_grid{z_min}_{z_max}_{N_z}_{DM_min}_{DM_max}_{DM_local}_{N_DM}' in PDMcosmicz_grid_fname
        if not hasattr(self, PDMcosmicz_grid_fname): # if copying, this won't be executed
            self.PDMcosmicz_grid_fname = PDMcosmicz_grid_fname


        try:
            self.PDMcosmicz_grid = pickle.load( open(self.PDMcosmicz_grid_fname, 'rb') )
            # print('Loaded pre-computed P(DM_cosmic|z) grid')
        except FileNotFoundError:
            self.PDMcosmicz_grid = self.get_PDMcosmicz_grid()
            with open(self.PDMcosmicz_grid_fname, 'wb') as handle:
                pickle.dump(self.PDMcosmicz_grid, handle)

    def copy(self):
        '''returns another object with the same _params, aka a deepcopy
        assumes the object being copied already has a pre-computed P(DM_cosmic|z) grid loaded and filename stored
        '''
        paramkwargs = dict(zip(self.parameters, self._params))
        paramkwargs.update({'PDMcosmicz_grid_fname': self.PDMcosmicz_grid_fname})
        copyobj = self.__class__(**paramkwargs)
        copyobj.set_grid_parameters()
        return copyobj

    def set_parameters(self, **kwargs):
        """same as the inherited version of the function, except also sets a flag"""
        for key in list(kwargs):
            param_idx = self.parameters.index(key)
            self._params[param_idx] = kwargs.pop(key)
        self._set_grid_params = False

    def set_grid_parameters(self, DM_EG=False):
        """
        should be called after set_parameters whenever calculating the posterior
        also precomputes P_FDM_interp given the params
        """
        # generates P(DM_host | z) grid
        self.PDMhostz_grid = self.get_PDMhostz_grid(DM_EG=DM_EG)

        # if DM_EG, generates P(DM_EG | z) grid
        # else, generates P(DM | z) grid
        #  based on = \int dDM_host P(DM_host|z) P(DM_cosmic = DM_EG - DM_host | z)
        self.PDM_givenz_grid = self.get_PDM_givenz_grid()

        # generates P(F, z) and P(F, DM)
        self.P_Fz = self.fluence_distribution(self.z_arr, self.fluence_arr*u.Jy*u.ms).value
        self.P_FDM = self.get_P_FDM()
        self.P_FDM_interp = RegularGridInterpolator((self.fluence_arr, self.DM_arr), self.P_FDM, bounds_error=False)

    def probabilities(self, det_frbs_cut):
        return self.probs('snr_dm', det_frbs_cut) * self.probs('width', det_frbs_cut.width) * self.probs('scat', det_frbs_cut.scat)

    def rate(self, det_frbs_cut):
        return self.probabilities(det_frbs_cut)

    def probs(self, ptype, ddata):
        if ptype == 'snr_dm':
            if not self._set_grid_params:
                self.set_grid_parameters()
                self._set_grid_params = True

            flu = ddata.fluence
            dm = ddata.dm
            points = np.vstack((flu, dm)).T
            return self.P_FDM_interp(points)
        else:
            return self.lognorm_model(ptype, ddata)

    def get_PDMhostz_grid(self, DM_EG):
        """
        Generates P(DM_host | z) grid from an analytic expression
            DM_host is evaluated at DM_EG - DM_cosmic at each DM_cosmic value array
            Rather than do double-for-loop iterations, everything is pre-computed on a grid

        Args:
            DM_EG (bool):
                if DM_EG==True, then DM_host is evaluated for the DM_EG grid
                if DM_EG==False, then DM_host is evaluated for the DM=DM_EG+DM_L grid
        """
        # needed for P(DM_EG | z)
        if DM_EG:
            DM_host_arr = self.DM_EG_arr[:, None] - self.DM_cosmic_arr[None, :]
        # needed for P(DM = DM_EG+DM_L | z)
        else:
            DM_host_arr = (self.DM_EG_arr[:, None] + self._DM_local) - self.DM_cosmic_arr[None, :]
 
        DM_prime = (1+self.z_arr[:, None, None]) * DM_host_arr[None, :]
        logDM = np.log(DM_prime, out=np.zeros_like(DM_prime), where=(DM_prime>0)) # infs -> 0 for where DM_host_arr<=0

        logmean, logsigma = self.get_parameters(['mu_host', 'sigma_host'])

        norm = 1/DM_host_arr * 1/(logsigma*np.sqrt(2*np.pi))
        norm[np.where(DM_host_arr<0)] = 0

        return norm * np.exp(-0.5 * ((logDM-logmean)/logsigma)**2)

    def get_PDMcosmicz_grid(self):
        """
        Generates P(DM_cosmic | z) grid by iterating over redshift and
        calling `self.get_PDMcosmic_z` at each z, which evaluates P(DM_cosmic|z) for
        an array of DM_cosmic at a given z
        """
        PDF_slices_cosmic = np.zeros((self._N_z, self._N_DM))
        print('Generating P(DM_cosmic|z)')
        for i, z in enumerate(tqdm(self.z_arr)):
            PDF_slices_cosmic[i] = self.get_PDMcosmic_z(z)
        return PDF_slices_cosmic

    def get_PDMcosmic_z(self, z):
        """
        Evaluates P(DM_cosmic|z) for an array of DM_cosmic at a given z.

        Args:
            z (float):
                redshift
            DM_min (float):
                min DM in DM_cosmic
            DM_max (float):
                max DM in DM_cosmic
            N_DM (int):
                number of DMs in DM_cosmic
            beta (float, optional):
                parameter for the model;
                see Macquart+20, James+21 for details
            F (float, optional):
                parameter for feedback for the model;
                see Macquart+20, James+21 for details

        Returns:
            normalized np.array of P(DM_cosmic|z)

        Adapted from ```frb.dm.prob_dmz.grid_P_DMcosmic_z(beta=3., F=0.31, zvals=None)```
        in https://github.com/FRBs/FRB
        """
        from frb.dm import igm, cosmic

        # loading grid
        f_C0 = cosmic.grab_C0_spline()

        avgDM = igm.average_DM(z).value

        # Params
        sigma = self.F / np.sqrt(z)
        C0 = f_C0(sigma)
        #  Delta
        Delta = self.DM_cosmic_arr / avgDM
        # PDF time
        PDF = cosmic.DMcosmic_PDF(Delta, C0, sigma)
        PDF[np.where(Delta==0)] = 0
        # Normalize P(DM|z) over DM axis to sum to 1
        PDF /= simps(PDF, self.DM_cosmic_arr)

        return PDF

    def get_PDM_givenz_grid(self):
        """
        Generates P(DM|z) and makes sure it is normalized at each z in the z_arr
        """
        PDM_givenz_grid = np.zeros((self._N_z, self._N_DM))
        for zz, z in enumerate(self.z_arr):
            PDM_givenz_grid[zz] = simps( self.PDMhostz_grid[zz] * self.PDMcosmicz_grid[zz],
                                   self.DM_cosmic_arr )
            norm = simps(PDM_givenz_grid[zz], self.DM_cosmic_arr)
            if norm==0 and np.all(PDM_givenz_grid[zz]==0):
                PDM_givenz_grid[zz] = np.zeros_like(PDM_givenz_grid[zz])
            else:
                PDM_givenz_grid[zz] /= norm
        return PDM_givenz_grid

    def get_P_FDM(self):
        """
        P(F, DM) = \int dz P(F, z) P(DM | z)
        """
        P_FDM = np.zeros((len(self.fluence_arr), self._N_DM))

        for dd, DM in enumerate(self.DM_EG_arr):
            # P_FDM[:, dd] = simps(P_Fz * self.PDM_givenz_grid[:,dd], self.z_arr)

            # last index of log spacing (z=1.0)
            zi=self._zi-1

            logpiece = self.z_arr[:zi+1] * self.P_Fz[:,:zi+1] * self.PDM_givenz_grid[:zi+1,dd]
            logint = simps(logpiece, np.log(self.z_arr[:zi+1]))

            linpiece = self.P_Fz[:,zi:] * self.PDM_givenz_grid[zi:,dd]
            linint = simps(linpiece, self.z_arr[zi:])

            P_FDM[:, dd] = logint + linint


        # P_FDM /= simps(np.trapz(P_FDM, x=self.fluence_arr, axis=0), self.DM_EG_arr)

        return P_FDM

    def energy_from_fluence(self, z, fluence):
        """
        Gets FRB energy from fluence using the spectral index
        alpha (F \propto \nu^\alpha) from the model

        Args:
            z (float):
                redshift
            fluence (astropy.units.quantity.Quantity):
                array of fluences with astropy units, preferably Jy ms

        Returns:
            np.array of energies (astropy.units.quantity.Quantity) in ergs

        note to self —
            since we no longer assume dm is a perfect 1-1 proxy for distance, in the schechter
            model, the quantity i had as propto energy ('fludm2') would be replaced by this
        """
        alpha = self.get_parameters('alpha')
        freq_bandwidth = 1.0*u.GHz
        energy = ((4*np.pi*cosmo.luminosity_distance(z[None,:])**2) / (1+z[None,:])**(2+alpha)) * freq_bandwidth * fluence[:,None]

        return energy.to(u.erg)

    def schechter_function(self, z, fluence):
        """
        Returns a Schechter function of the distribution of energies given fluences
        using parameters from the model

        .. math::
        P(E) = \frac{1}{E_{char}} \left(\frac{E}{E_{char}}\right)^\gamma \exp{\left[-\frac{E}{E_{char}}\right]}

        Args:
            z (float):
                redshift
            fluence (astropy.units.quantity.Quantity):
                array of fluences with astropy units, preferably Jy ms

        Returns:
            np.array of distribution of energies P(E|z) [1/erg]
        """
        log_energy_char, gamma = self.get_parameters(['log_energy_char', 'gamma'])
        energy = self.energy_from_fluence(z, fluence)
        assert energy.unit == u.erg # since this is what the char. energy unit is in

        energy_char = np.exp(log_energy_char) * u.erg
        llim = self.get_parameters('E_pivot') / energy_char.value

        schec_fn = 1/energy_char * (energy/energy_char)**(gamma) * np.exp(-energy/energy_char)
        norm_factor = float(np.asarray(mpmath.gammainc(gamma+1, llim, regularized=False)))
        if norm_factor==0:
            raise ValueError(f"the normalization factor is 0: log_energy_char={log_energy_char}, gamma={gamma}")

        return schec_fn / norm_factor

    def SFR(self, z):
        """
        Avg SFR density given by Madau & Dickinson (2014)
        .. math::
            \sfr(z) = 0.015 \frac{(1+z)^{2.7}}{1+\left(\frac{1+z}{2.9}\right)^{5.6}}

        Args:
            z (float):
                redshift

        Returns:
            SFR density (astropy.units.quantity.Quantity) [Msun/yr/Mpc^3]
        """
        return 1.0025738 * ((1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)) * u.Msun/u.yr/u.Mpc**3

    def SFR_evolution(self, z, n):
        """
        Rate of FRBs per comoving volume [yr^-1 Gpc^-3]

        .. math::
            \Phi(z) = \frac{\Phi_0}{1+z} \left(\frac{\sfr(z)}{\sfr(0)} \right)^n

        Args:
            z (float):
                redshift
            n (float):
                SFR evolution factor

        Returns:
            SFR evolution rate (astropy.units.quantity.Quantity) [1/yr/Gpc^3]
        """
        log_rate = np.exp(self.get_parameters('log_rate')) * 1./u.yr/u.Gpc**3

        rate_interp_factor = 1
        if self.rate_interp:
            alpha = self.get_parameters('alpha')
            rate_interp_factor = (1+z)**alpha

        return rate_interp_factor * log_rate / (1+z) * (self.SFR(z)/self.SFR(0))**n

    def energy_distribution(self, z, fluences):
        """
        Returns the energy distribution of FRBs scaled by the SFR rate given fluences
        using parameters from the model

        .. math::
        P(E,z) = P(E) \chi(z)^2 \frac{d\chi}{dz} \Phi(z)

        Args:
            z (float):
                redshift
            fluence (astropy.units.quantity.Quantity):
                array of fluences with astropy units, preferably Jy ms

        Returns:
            np.array of rate distribution of energies R(E,z) [1/erg/yr]
        """
        n = self.get_parameters('n')
        return self.schechter_function(z, fluences) * differential_volume_element(z) * self.SFR_evolution(z, n)

    def fluence_distribution(self, z, fluences):
        """
        Returns the fluence distribution at a redshift using parameters
        from the model

        Args:
            z (float):
                redshift
            fluence (astropy.units.quantity.Quantity):
                array of fluences with astropy units, preferably Jy ms

        Returns:
            np.array of rate distribution of fluences P(F,z) [1/Jy/ms/yr]
        """
        freq_bandwidth = 1.0*u.GHz

        if self.rate_interp:
            alpha = 0 # gets rid of the k-corr from the 'true spectral idx' interp
        else:
            alpha = self.get_parameters('alpha')

        dE_dF = ((4*np.pi*cosmo.luminosity_distance(z)**2) / (1+z)**(2+alpha)) * freq_bandwidth

        return (self.energy_distribution(z, fluences) * dE_dF).to(1/u.Jy/u.ms/u.yr)


class StochasticDMTimeDelayModel(StochasticDMModel):
    parameters = (
        'log_rate',         # normalization
        'gamma',            # pivot index for schechter fn
        'log_energy_char',  # characteristic energy E_char for schechter fn
        'tau',              # time delay w.r.t. SFR
        'alpha',            # spectral index
        'mu_host',          # center of lognorm distr of DM'_host ("logmean") in log space
        'sigma_host',       # spread of lognorm distr of DM'_host ("logsigma") in log space
        'width_shape',
        'width_loc',
        'width_scale',
        'width_uniform_min',
        'width_uniform_max',
        'width_uniform_fraction',
        'scat_shape',
        'scat_loc',
        'scat_scale',
        'scat_uniform_min',
        'scat_uniform_max',
        'scat_uniform_fraction',
        'survey_rate_factor',
        'E_pivot'
    )

    def __init__(self, beta=3., F=0.31, z_min=1e-3, z_max=4.0, N_z=100,
                 DM_min=100.0, DM_max=6500.0, DM_local=80.0, N_DM=150,
                 rate_interp=False,
                 PDMcosmicz_grid_fname=None,
                 **kwargs):
        """
        initializing with DM = DM_EG + DM_L
        """
        npars = len(self.parameters)
        if 'prefix' in kwargs:
            prefix = kwargs.pop('prefix')
        else:
            prefix = ''
        params, kwargs = self._set_init_parameters(npars, prefix, **kwargs)
        # check that we consumed all input parameters
        if kwargs:
            raise TypeError("Received unexpected keyword arguments: " + str(kwargs.keys()))
        self._params = params

        self.rate_interp = rate_interp

        self.beta = beta
        self.F = F

        self._z_min = z_min
        self._z_max = z_max
        self._N_z = N_z

        self._zi = int(N_z*.68) # the index+1 where the array goes from log to lin binning (z=1.0, only for Nz=100)
        diff = np.diff(np.logspace(np.log10(z_min), 0, num=self._zi))[-1]
        self.z_arr = np.concatenate((
            np.logspace(np.log10(z_min), 0, num=self._zi),
            np.arange(1.0+diff, z_max+2*diff, diff)
        ))

        self.fluence_arr = np.logspace(0., 5.7, 200) # 1 ~ 5e5 Jy ms

        self._DM_min = DM_min
        self._DM_max = DM_max
        self._DM_local = DM_local
        self._N_DM = N_DM

        # DM = DM_EG + DM_L
        # DM_EG = DM - DM_L
        # ultimately DM_cosmic will be marginalized over, so we choose to define that the same as DM_EG
        self.DM_arr = np.linspace(DM_min, DM_max, N_DM)
        self.DM_EG_arr = self.DM_arr - self._DM_local
        self.DM_cosmic_arr = self.DM_arr - self._DM_local

        self._set_grid_params = False

        # loads in P(DM_cosmic | z) grid if it exists; otherwise, generates it
        if PDMcosmicz_grid_fname is None:
            raise ValueError("You should pass in a path for either loading or saving the appropriate P(DM_cosmic|z) grid.\
            Filename formatting should be f'PDMcosmicz_grid{z_min}_{z_max}_{N_z}_{DM_min}_{DM_max}_{DM_L}_{N_DM}'")

        assert f'PDMcosmicz_grid{z_min}_{z_max}_{N_z}_{DM_min}_{DM_max}_{DM_local}_{N_DM}' in PDMcosmicz_grid_fname
        if not hasattr(self, PDMcosmicz_grid_fname): # if copying, this won't be executed
            self.PDMcosmicz_grid_fname = PDMcosmicz_grid_fname


        try:
            self.PDMcosmicz_grid = pickle.load( open(self.PDMcosmicz_grid_fname, 'rb') )
            # print('Loaded pre-computed P(DM_cosmic|z) grid')
        except FileNotFoundError:
            self.PDMcosmicz_grid = self.get_PDMcosmicz_grid()
            with open(self.PDMcosmicz_grid_fname, 'wb') as handle:
                pickle.dump(self.PDMcosmicz_grid, handle)

    def SFR_evolution(self, z, tau):
        """
        Rate of FRBs per comoving volume [yr^-1 Gpc^-3]

        .. math::
            \Phi(z) = \frac{\Phi_0}{1+z} \left(\frac{\sfr(z)}{\sfr(0)} \right)^n

        Args:
            z (float):
                redshift
            n (float):
                SFR evolution factor

        Returns:
            SFR evolution rate (astropy.units.quantity.Quantity) [1/yr/Gpc^3]
        """
        log_rate = np.exp(self.get_parameters('log_rate')) * 1./u.yr/u.Gpc**3

        rate_interp_factor = 1
        if self.rate_interp:
            alpha = self.get_parameters('alpha')
            rate_interp_factor = (1+z)**alpha

        delayed_zs = get_delayed_z(z, tau)

        return rate_interp_factor * log_rate / (1+z) * (self.SFR(delayed_zs)/self.SFR(0))

    def energy_distribution(self, z, fluences):
        """
        Returns the energy distribution of FRBs scaled by the SFR rate given fluences
        using parameters from the model

        .. math::
        P(E,z) = P(E) \chi(z)^2 \frac{d\chi}{dz} \Phi(z)

        Args:
            z (float):
                redshift
            fluence (astropy.units.quantity.Quantity):
                array of fluences with astropy units, preferably Jy ms

        Returns:
            np.array of rate distribution of energies R(E,z) [1/erg/yr]
        """
        tau = self.get_parameters('tau') * u.Gyr
        return self.schechter_function(z, fluences) * differential_volume_element(z) * self.SFR_evolution(z, tau)



## other misc used functions for stochasticdmmodel
tmp_zs = np.linspace(0, 6, int(1e5))
tmp_lbts = cosmo.lookback_time(tmp_zs)
tmp_interp_fn = interp1d(tmp_lbts,tmp_zs)
def get_delayed_z(z, tau):
    """
    t and \tau are lookback time in Gyr
    this returns z_a( t(z) - \tau ) using astropy functions
    """
    # getting t(z) - \tau
    t_arr = cosmo.lookback_time(z) # t(z)
    delayed_t_arr = t_arr - tau
    delayed_t_arr[delayed_t_arr<=cosmo.lookback_time(1e-8)] = 0

    # getting z_a( t(z) - \tau )
    return tmp_interp_fn(delayed_t_arr)


def dchi_dz(z):
    """
    .. math::
    \frac{d\chi}{dz} = \frac{c}{H(z)}
    
    Args:
        z (float):
            redshift

    Returns:
        d\chi/dz (astropy.units.quantity.Quantity)
    """
    return c / cosmo.H(z)


def differential_volume_element(z):
    """
    Differential volume element such that
    \chi(z)^2 d\chi/dz is equivalent to a volume element dV/dz [Gpc^3]
    
    Args:
        z (float):
            redshift

    Returns:
        \chi(z)^2 d\chi/dz (astropy.units.quantity.Quantity) [Gpc^3]
    """
    return (cosmo.comoving_distance(z)**2 * dchi_dz(z)).to(u.Gpc**3)


## creating DistributionModel for injections sample
def get_model_inj(all_frbs, det_frbs_cut):
    inj_survey_rate_factor = (det_frbs_cut.inj_efficiency * len(all_frbs)) / all_frbs.f_sky
    conv_factor = 1/(2 * np.sqrt(2*np.log(2)))
    init_params = {
        'init_norm':1.0,
        'init_alpha':-1.0,
        'init_flu_pivot':0.2, # fluence_min in Jy ms
        'init_dm_shape':0.7624938199356888,
        'init_dm_loc':0.0,
        'init_dm_scale':462.3851062584323,
        'init_dm_uniform_min':0.0,
        'init_dm_uniform_max':5000.0,
        'init_dm_uniform_fraction':0.1,
        'init_width_shape':1.0364933996435612,
        'init_width_loc':0.0,
        'init_width_scale':0.0010586528808992726 * conv_factor,
        'init_width_uniform_min':0.0,
        'init_width_uniform_max':0.1  * conv_factor,
        'init_width_uniform_fraction':0.1,
        'init_scat_shape':2.0644474968496342,
        'init_scat_loc':0.0,
        'init_scat_scale':0.0007422981186491531,
        'init_scat_uniform_min':0.0,
        'init_scat_uniform_max':0.1,
        'init_scat_uniform_fraction':0.1,
        'init_survey_rate_factor':inj_survey_rate_factor,
    }

    model_inj = DistributionModel(prefix='init_', **init_params)
    return model_inj

