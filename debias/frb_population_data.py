import h5py
import json
import numpy as np
import pickle

from debias.frb_population import FRBPopulation


FRBDTYPE = np.dtype([
    ("inj_program_id", np.int64),
    ("snr", np.float64),
    ("dm", np.float64),
    ("width", np.float64),
    ("cat_orig_width", list),
    ("box_car_width", np.float64),
    ("cat_orig_scat", 'U32'),
    ("scat", np.float64),
    ("fluence", np.float64),
    ("f_sky", np.float64),
    ("inj_efficiency", np.float64),
    ("event_number", 'U32'),
    ("repeater_of", 'U32'),
    ("spec_ind", np.float64),
    ("spec_run", np.float64),
    ("dm_excess_ymw16", np.float64),
    ("dm_excess_ne2001", np.float64),
    ("dm_gal_ymw16", np.float64),
    ("dm_gal_ne2001", np.float64),
    ("rfi_grade", np.float64),
    ("x", np.float64),
    ("y", np.float64),
    ("ra", np.float64),
    ("dec", np.float64),
    ("gal_lat", np.float64),
    ("toa_inf", np.float64),
    ("toa_inf_offset", np.float64),
])


class FRBPopulationData:
    """Stores data for a bunch of FRBs.

    """

    def __init__(self, **kwargs):
        self._property_set = {}
        for key in FRBDTYPE.names:
            if key in kwargs:
                self._property_set[key] = True
            else:
                self._property_set[key] = False
        nfrbs = -1
        for key, data in kwargs.items():
            if key not in FRBDTYPE.names:
                raise ValueError("Got unexpected FRB property: %s." % key)
            if nfrbs == -1:
                nfrbs = len(data)
                self._frbdata = np.empty(nfrbs, dtype=FRBDTYPE)
            self._frbdata[key] = data

    @classmethod
    def from_data(cls, frbdata, property_set):
        self = cls.__new__(cls)
        self._frbdata = frbdata
        self._property_set = property_set
        return self

    @classmethod
    def from_injections_all(cls, sched_filename):
        fluence, dm, width, scat = get_inj_sched_properties(sched_filename, inj_program_id=None, get_all=True)
        cls.f_sky = get_f_sky(sched_filename)

        return cls(fluence=fluence, dm=dm, width=width, scat=scat)

    @classmethod
    def from_injections_det(cls, filename, sched_filename):
        # fluence in Jy ms
        inj_program_id, snr, rfi_grade, dm_inj, x, dm_gal_ne2001, dm_gal_ymw16 = get_injections_properties(filename)
        fluence, dm, width, scat = get_inj_sched_properties(sched_filename, inj_program_id, get_all=False)
        cls.inj_efficiency = get_inj_efficiency(filename, sched_filename)

        return cls(inj_program_id=inj_program_id, snr=snr, rfi_grade=rfi_grade, x=x,
            dm_gal_ne2001=dm_gal_ne2001, dm_gal_ymw16=dm_gal_ymw16,
            fluence=fluence, dm=dm, width=width, scat=scat)

    @classmethod
    def from_catalog(cls, filename):
        snr, dm, cat_orig_width, width_nums, box_car_width, cat_orig_scat, cat_scat_nums, event_number, repeater_of, \
            dm_excess_ymw16, dm_excess_ne2001, gal_lat, spec_ind, spec_run = get_catalog_properties(filename)
        return cls(snr=snr, dm=dm, cat_orig_width=cat_orig_width, width=width_nums, box_car_width=box_car_width,
            cat_orig_scat=cat_orig_scat, scat=cat_scat_nums,
            event_number=event_number, repeater_of=repeater_of,
            dm_excess_ymw16=dm_excess_ymw16, dm_excess_ne2001=dm_excess_ne2001, gal_lat=gal_lat)#, spec_ind=spec_ind)


    def __getitem__(self, sl):
        """Create a subpopulation of FRBs.

        Parameters:
        sl: 1D slice or index array
            What slice of the population to select. e.g. `np.s_[:100]` for the first 100 FRBs.

        """
        return FRBPopulationData.from_data(self._frbdata[sl], self.property_set)

    def __getattr__(self, name):
        if name not in FRBDTYPE.names:
            raise AttributeError("Object has no attribute %s." % name)
        elif self._property_set[name]:
            return self._frbdata[name]
        else:
            raise AttributeError("The %s property has not yet been set." % name)

    def __setattr__(self, name, value):
        if name in FRBDTYPE.names:
            self._frbdata[name] = value
            self._property_set[name] = True
        else:
            self.__dict__[name] = value

    def __len__(self):
        return len(self._frbdata)

    @property
    def data(self):
        """Export population to an array."""
        return self._frbdata.copy()

    @property
    def property_set(self):
        """Export self._property_set."""
        return dict(self._property_set)


def get_injections_properties(filename):
    def _extract_injections(self, resp):
        """
        A function which takes a response from FRB master to the injections
        database and returns a time sorted dataframe of injections and
        detections.
        Written by Marcus
        """
        import pandas as pd

        resp_json = resp.json()
        inj = pd.DataFrame(resp_json["injections"]).sort_values("injection_time")
        det = pd.DataFrame(resp_json["detections"]).sort_values("timestamp_utc")
        print("Injections shape: {}".format(inj.shape))
        print("Detections shape: {}".format(det.shape))
        inj["injection_time"] = inj["injection_time"].astype(np.datetime64)
        det["timestamp_utc"] = det["timestamp_utc"].astype(np.datetime64)
        data = pd.merge(
            inj,
            det,
            left_on="id",
            right_on="det_id",
            how="outer", # inner will make it so only detected shows up. Outer so that all shows up
            suffixes=("_inj", "_det")
        )
        data = pd.concat(
            [
                data.drop(['extra_injection_parameters'], axis=1),
                data["extra_injection_parameters"].apply(pd.Series)
            ],
            axis=1
        )
        data = data.set_index("timestamp_utc")
        data = data.tz_localize("UTC")
        return data

    # obtaining flat alpha injections
    try:
        with open(filename, 'rb') as handle:
            inj = pickle.load(handle)
    except FileNotFoundError:
        import requests

        # don't forget  mimic-add-fit-spec-coeffs -o 200813_flat_alpha_base.h5
        print('retreiving schedule file')
        resp = requests.get(
            "http://localhost:8001/v2/mimic/injection/program/"+filename[:-2]+".h5")
        inj = _extract_injections(resp)
        with open(filename, 'wb') as handle:
            pickle.dump(inj, handle)

    inj_program_id = inj.injection_program_id.values
    inj_snr = inj.combined_snr.values
    inj_rfi_grade = inj.rfi_grade_level2.values
    inj_dm = inj.dm_inj.values
    inj_x = inj['beam_x'].values
    inj_dm_gal_ymw_2016_max = inj['dm_gal_ymw_2016_max'].values
    inj_dm_gal_ne_2001_max = inj['dm_gal_ne_2001_max'].values

    return inj_program_id, inj_snr, inj_rfi_grade, inj_dm, inj_x, inj_dm_gal_ne_2001_max, inj_dm_gal_ymw_2016_max


def get_catalog_properties(filename):
    # obtaining catalog data
    with open(filename) as fftmp:
        cat = json.load(fftmp)
    cat = cut_cat_exposure(cat)

    cat_snr = np.array([frb['bonsai_snr'] for frb in cat]).astype(float)
    cat_dm  = np.array([frb['fitburst_dm'] for frb in cat])
    cat_width = np.array([frb['pulse_width_ms'] for frb in cat], dtype=object)
    cat_width_nums = np.zeros(len(cat))
    cat_bcwidth = np.array([frb['box_car_width'] for frb in cat])
    cat_scattering = np.array([frb['scattering_time_ms'] for frb in cat])
    cat_scattering_nums = get_cat_scat_nums(cat)
    cat_event_number = np.array([frb['event_number'] for frb in cat])
    cat_repeater_of = np.array([frb['repeater_of'] for frb in cat])
    cat_dm_excess_ymw16 = np.array([frb['dm_excess_ymw16'] for frb in cat])
    cat_dm_excess_ne2001 = np.array([frb['dm_excess_ne2001'] for frb in cat])
    cat_gal_lat = np.array([frb['gb'] for frb in cat])
    cat_si = get_cat_si(cat)
    cat_sr = get_cat_sr(cat)

    return (cat_snr, cat_dm, cat_width, cat_width_nums, cat_bcwidth, cat_scattering, cat_scattering_nums,
        cat_event_number, cat_repeater_of, cat_dm_excess_ymw16, cat_dm_excess_ne2001, cat_gal_lat, cat_si, cat_sr)


def get_inj_sched_properties(filename, inj_program_id, get_all):
    '''gets data from the injection base files

    if get_all, returns detected injected frbs in inj_program_id ordering
    (from the inj scheduler file)

    also assumes snr_estimate exists in the base file

    since all_frbs fluence is in Jy s (the min is 2e-4, the min fluence in Jy s in the config file)
    the fluence values are multiplied by 1e3 to convert them to Jy ms
    '''
    # obtaining flat alpha schedule file
    ff = h5py.File(filename, 'r')

    dset_frb = ff['frb']
    if get_all:
        all_frbs = FRBPopulation.from_data(dset_frb[()], dict(dset_frb.attrs))
        return (all_frbs.band_mean_fluence() * 1e3, all_frbs['dm'].data,
            all_frbs['width'].data * 1/(2 * np.sqrt(2*np.log(2))), all_frbs.scat600)

    dset_inj = ff['to_inject']
    dset_speccoeffs = ff['to_inject_fit_spec_coeffs'][()]
    dset_snr_est = dset_inj['snr_estimate']

    # detected
    assert inj_program_id is not None
    frb_iis = inj_program_id
    dset_frb_injfrbs = np.empty(len(frb_iis), dtype=dset_frb[0].dtype)
    slice_iis = np.linspace(0, 1e8, 11, dtype=int)
    last_ii = 0
    for ii in range(len(slice_iis)-1):    
        dset_frb_slicing = slice(slice_iis[ii], slice_iis[ii+1])
        idxs = np.where((frb_iis >= slice_iis[ii]) & (frb_iis <= slice_iis[ii+1]))[0]
        num_new = len(idxs)
        if num_new == 0: break

        ## todo: assert preserves ordering of inj.inj_program_id? this probs doesn't scale tho
        assert np.all(frb_iis[idxs] == inj_program_id)

        frb_iis_inslice = frb_iis[idxs] - slice_iis[ii]    
        dset_frb_injfrbs[last_ii:last_ii+num_new] = dset_frb[dset_frb_slicing][frb_iis_inslice]
        last_ii += num_new
    det_frbs = FRBPopulation.from_data(dset_frb_injfrbs, dict(dset_frb.attrs))

    return (det_frbs.band_mean_fluence() * 1e3, det_frbs['dm'].data,
        det_frbs['width'].data * 1/(2 * np.sqrt(2*np.log(2))), det_frbs.scat600)


def get_cat_scat_nums(cat):
    ''' replaces the ULs with 1/2 the values
    '''
    cat_scats = np.array([frb['scattering_time_ms'] for frb in cat])
    cat_scats_final = np.copy(cat_scats)
    # replacing ULs
    UL_iis = np.array([x for x in range(len(cat_scats_final)) if '<' in cat_scats_final[x]])
    cat_scats_final[UL_iis] = np.array([0.5 * float(x[1:]) for x in cat_scats_final[UL_iis]])

    return cat_scats_final.astype(float)


def get_cat_si(cat_cut):
    cat_si = np.array([frb['spectral_index'] for frb in cat_cut], dtype=object)
    cat_si = np.array([item for sublist in cat_si for item in sublist if len(sublist)==1])
    return cat_si


def get_cat_sr(cat_cut):
    cat_sr = np.array([frb['spectral_running'] for frb in cat_cut], dtype=object)
    cat_sr = np.array([item for sublist in cat_sr for item in sublist if len(sublist)==1])
    return cat_sr


def cut_cat_exposure(cat):
    '''cuts events detected during exposure issues
    Events detected during precommissioning phase: 13
    Events detected on days with low sensitivity: 23
    Events detected on days with software upgrades: 3

    ^ 39 total events cut due to exposure issues:
    ['FRB20180729B' 'FRB20180806A' 'FRB20180817A' 'FRB20180810A'
     'FRB20180729A' 'FRB20180814B' 'FRB20180814A' 'FRB20180801A'
     'FRB20181209A' 'FRB20180727A' 'FRB20180911A' 'FRB20180928A'
     'FRB20180730A' 'FRB20180810B' 'FRB20180725A' 'FRB20180812A'
     'FRB20181208A' 'FRB20180911C' 'FRB20190329B' 'FRB20190327A'
     'FRB20190329C' 'FRB20190616A' 'FRB20190619D' 'FRB20190617C'
     'FRB20190609A' 'FRB20190612B' 'FRB20190617A' 'FRB20190614A'
     'FRB20190617B' 'FRB20190619C' 'FRB20190612A' 'FRB20190618A'
     'FRB20190613A' 'FRB20190619A' 'FRB20190608A' 'FRB20190611A'
     'FRB20190612C' 'FRB20190613B' 'FRB20190619B']
    '''
    # exposure issues
    exclude_events = np.array(['9564770', '13401643',  '9541535',  '9976641', '10353713', '10889573', '11121416', '10886879',
        '9641579',  '9574801',  '9461009', '10366092', '13435448', '17423783', '22154974',  '9386707',
        '10648497', '22109717', '34730029', '34532625', '34775137', '41597828', '41802046', '41692959',
        '41473417', '41683788', '41208961', '41788644', '41366878', '41643446', '41365524', '41710598',
        '41422618', '41742525', '41189480', '41358073', '41412448', '41464641', '41750859'])
    exclude_iis = np.array([x for x in range(len(cat)) if cat[x]['event_number'] in exclude_events])
    cat = np.delete(cat, exclude_iis)

    return cat


def get_f_sky(sched_filename):
    import h5py
    f = h5py.File(sched_filename, mode='r')
    f_sky = f.attrs["sky coverage fraction"]

    return f_sky


def get_inj_efficiency(filename, sched_filename):
    """reads the injections database and the number of to_injects
    in the schedule file to determine injections efficiency"""
    import h5py, pickle
    with open(filename, 'rb') as handle:
        inj = pickle.load(handle)

    f = h5py.File(sched_filename, mode='r')

    return len(inj) / len(f['to_inject'])
