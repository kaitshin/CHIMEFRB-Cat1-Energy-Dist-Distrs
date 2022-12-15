import numpy as np


def get_cat_width(cat, cat_cut=None):
    '''Replaces multiple width values by an "effective width" estimated by
    fitting a line through single-burst widths (without upper limits) and their box car widths

    Single-burst upper limits are replaced by 1/2 the UL value for now
    '''
    def get_cat_width_params(line, cat_cut):
        '''The relation is fit to the sample after cuts
        '''
        from scipy.optimize import curve_fit

        cat_width = cat_cut['cat_orig_width'].data
        cat_bcwidth = cat_cut['box_car_width'].data

        sing_width_noUL_iis = np.array([x for x in range(len(cat_width)) if 
            (len(cat_width[x])==1 and '<' not in cat_width[x][0])])
        sing_width_noULs = np.array([float(x[0]) for x in cat_width[sing_width_noUL_iis]])
        mult_width_iis = np.array([x for x in range(len(cat_width)) if (len(cat_width[x])>1)])

        params = curve_fit(line, cat_bcwidth[sing_width_noUL_iis], sing_width_noULs)

        return params

    def line(x, m, b):
        return m*x + b

    if cat_cut is None:
        print('!!! Only for width jackknifes')
        params = get_cat_width_params(line, cat)
    else:
        params = get_cat_width_params(line, cat_cut)

    cat_width_final = cat['cat_orig_width'].data
    cat_bcwidth = cat['box_car_width'].data

    sing_width_noUL_iis = np.array([x for x in range(len(cat_width_final)) if
        (len(cat_width_final[x])==1 and '<' not in cat_width_final[x][0])])
    sing_width_noULs = np.array([float(x[0]) for x in cat_width_final[sing_width_noUL_iis]])
    mult_width_iis = np.array([x for x in range(len(cat_width_final)) if (len(cat_width_final[x])>1)])

    cat_width_final[sing_width_noUL_iis] = sing_width_noULs
    cat_width_final[mult_width_iis] = line(cat_bcwidth[mult_width_iis], *params[0])

    # replacing ULs
    UL_iis = np.array([x for x in range(len(cat_width_final)) if type(cat_width_final[x])==list])
    if len(UL_iis > 0):
        cat_width_final[UL_iis] = np.array([0.5 * float(x[0][1:]) for x in cat_width_final[UL_iis]])

    return cat_width_final


def cut_far_sidelobes(cat, det_frbs):
    # cat far sidelobes
    far_sidelobes = np.array(['28210057', '29204654', '30139995'])
    far_sidelobes_iis = np.array([x for x in range(len(cat)) if cat.event_number[x] in far_sidelobes])

    mask = np.ones(len(cat), dtype=bool)
    mask[far_sidelobes_iis] = False
    cat = cat[mask]

    # inj far sidelobes
    detected_mask = np.logical_and.reduce((
        ((det_frbs.x <= 5) & (det_frbs.x >= -5)),
    ))
    det_frbs = det_frbs[detected_mask]

    return cat, det_frbs


def cut_near_AMB(cat, det_frbs):
    # cat sources too near the 'AMB' cutoff
    fitburst_dms = np.array([cat.dm[x] for x in range(len(cat))])
    dms_excess_ymw16 = np.array([cat.dm_excess_ymw16[x] for x in range(len(cat))])
    dms_ym16 = fitburst_dms - dms_excess_ymw16
    dms_excess_ne2001 = np.array([cat.dm_excess_ne2001[x] for x in range(len(cat))])
    dms_ne2001 = fitburst_dms - dms_excess_ne2001
    near_amb_iis = np.where( fitburst_dms < 1.5 * np.maximum(dms_ne2001, dms_ym16) )[0]

    mask = np.ones(len(cat), dtype=bool)
    mask[near_amb_iis] = False
    cat = cat[mask]

    # inj sources too near the 'AMB' cutoff
    detected_mask = np.logical_and.reduce((
        ((det_frbs.dm >= 1.5 * np.maximum(det_frbs.dm_gal_ne2001, det_frbs.dm_gal_ymw16))),
    ))
    det_frbs = det_frbs[detected_mask]

    return cat, det_frbs


def cut_high_ne2001(cat, det_frbs):
    # cat sources with dm_ne2001>=100
    fitburst_dms = np.array([cat.dm[x] for x in range(len(cat))])
    dms_excess_ne2001 = np.array([cat.dm_excess_ne2001[x] for x in range(len(cat))])
    dms_ne2001 = fitburst_dms - dms_excess_ne2001

    high_iis = np.where( dms_ne2001>=100 )[0]

    mask = np.ones(len(cat), dtype=bool)
    mask[high_iis] = False
    cat = cat[mask]

    # inj sources with dm_ne2001>=100
    detected_mask = np.logical_and.reduce((
        ((det_frbs.dm_gal_ne2001 < 100)),
    ))
    det_frbs = det_frbs[detected_mask]

    return cat, det_frbs


def cut_high_scat(cat, all_frbs, det_frbs, scat_cut):
    # cat high scats
    cat_scats = np.array([frb.cat_orig_scat for frb in cat])
    high_scats_ii = np.array([x for x in range(len(cat_scats)) if ('<' not in cat_scats[x] and float(cat_scats[x]) >= scat_cut)
        or ('<' in cat_scats[x] and float(cat_scats[x][1:]) >= scat_cut)])
    mask = np.ones(len(cat), dtype=bool)
    mask[high_scats_ii] = False
    cat = cat[mask]

    # inj high scats
    low_all_ii = np.where(all_frbs.scat < scat_cut)[0]
    all_frbs = all_frbs[low_all_ii]

    low_det_ii = np.where(det_frbs.scat < scat_cut)[0]
    det_frbs = det_frbs[low_det_ii]

    return cat, all_frbs, det_frbs


def cut_high_width(cat, all_frbs, det_frbs, width_cut):
    # cat high widths
    # cat_widths = np.array([frb.cat_orig_width for frb in cat])
    cat_widths = get_cat_width(cat)
    high_widths_ii = np.array([x for x in range(len(cat_widths)) if cat_widths[x] >= width_cut])
    mask = np.ones(len(cat), dtype=bool)
    mask[high_widths_ii] = False
    cat = cat[mask]

    # inj high widths
    low_all_ii = np.where(all_frbs.width < width_cut)[0]
    all_frbs = all_frbs[low_all_ii]

    low_det_ii = np.where(det_frbs.width < width_cut)[0]
    det_frbs = det_frbs[low_det_ii]

    return cat, all_frbs, det_frbs


def cut_cat_repeat_bursts(cat):
    repeater_iis = np.array([x for x in range(len(cat)) if len(cat.repeater_of[x]) > 0])
    repeater_of = np.array([cat.repeater_of[x] for x in range(len(cat))])
    event_nums = np.array([cat.event_number[x] for x in range(len(cat))])

    # for repeat events, keep only the first (in time) that made the previous cuts
    remove_iis = np.array([], dtype=int)
    for repeater_name in list(set(repeater_of[repeater_iis])):
        argsort_repeat_event_nums = np.argsort(event_nums[repeater_iis])
        duplicate_iis = np.where(repeater_of[repeater_iis][argsort_repeat_event_nums] == repeater_name)[0]
        if len(duplicate_iis) > 1:
            remove_iis = np.append(remove_iis, repeater_iis[argsort_repeat_event_nums][duplicate_iis[1:]])

    mask = np.ones(len(cat), dtype=bool)
    mask[remove_iis] = False
    cat = cat[mask]
    return cat


def apply_cuts(cat_orig, all_frbs_orig, det_frbs_orig,
    dm_cut=100.0, snr_cut=12.0, scat_cut=10e-3, width_cut=None, catalog=False):
    """Takes three FRBPopulationData objects and applies cuts such that
    the post-cut sample of FRBs can be used for statistical analyses.

    Parameters:
    cat_orig:
        Instance of `frb_population_data.FRBPopulationData`.
        Contains CHIME/FRB catalog 1 FRBs.
    all_frbs_orig: 
        Instance of `frb_population_data.FRBPopulationData`.
        Contains all of the injected FRB population.
    det_frbs_orig:
        Instance of `frb_population_data.FRBPopulationData`.
        Contains the detected injected FRB population from the scheduler file.

    Optional parameters:
    dm_cut: float. default 100 pc/cc
    snr_cut: float. default 12
    scat_cut: float. default 10 ms (10e-3 s)
    width_cut: float. default None

    Other cuts done regardless of kwargs: far sidelobes, sources near 'AMB' cutoff,
    repeat bursts (for the catalog bursts) keeping only the first detection

    Returns:
        cat_cut: cat_orig with all the cuts applied
        cat_all_scat: cat_orig with all the cuts applied except scattering
        all_frbs_cut: all_frbs_orig with all the cuts applied
        all_frbs_all_scat: all_frbs_orig with all the cuts applied except scattering
        det_frbs_cut: det_frbs_orig with all the cuts applied
        det_frbs_all_scat: det_frbs_orig with all the cuts applied except scattering
        bins: dict of bins associated with the cuts made here
    """
    # far sidelobes
    cat_all_scat, det_frbs_all_scat = cut_far_sidelobes(cat_orig, det_frbs_orig)

    # sources too near the 'AMB' cutoff
    cat_all_scat, det_frbs_all_scat = cut_near_AMB(cat_all_scat, det_frbs_all_scat)

    # low SNR
    cat_all_scat = cat_all_scat[cat_all_scat.snr >= snr_cut]
    detected_mask = np.logical_and.reduce((det_frbs_all_scat.snr >= snr_cut,
        ((det_frbs_all_scat.rfi_grade > 7.0) | (det_frbs_all_scat.snr > 30.0)),
    ))
    det_frbs_all_scat = det_frbs_all_scat[detected_mask]

    # low dm
    cat_all_scat = cat_all_scat[cat_all_scat.dm >= dm_cut]
    all_frbs_all_scat = all_frbs_orig[all_frbs_orig.dm >= dm_cut]
    det_frbs_all_scat = det_frbs_all_scat[det_frbs_all_scat.dm >= dm_cut]

    # high scat
    cat_cut, all_frbs_cut, det_frbs_cut = cut_high_scat(cat_all_scat,
        all_frbs_all_scat, det_frbs_all_scat, scat_cut)

    # cut high dm_ne2001 bursts (>=100) for populations analysis
    if not catalog:
        cat_cut, det_frbs_cut = cut_high_ne2001(cat_cut, det_frbs_cut)

    # high width
    if width_cut:
        cat_cut, all_frbs_cut, det_frbs_cut = cut_high_width(cat_cut, all_frbs_cut, det_frbs_cut, width_cut)
        cat_all_scat, all_frbs_all_scat, det_frbs_all_scat = cut_high_width(cat_all_scat, all_frbs_all_scat, det_frbs_all_scat, width_cut)

    # repeat bursts
    cat_cut = cut_cat_repeat_bursts(cat_cut)
    cat_all_scat = cut_cat_repeat_bursts(cat_all_scat)

    # checking resulting length of post-cut catalog, injections data objs
    if catalog:
        if dm_cut==100.0 and snr_cut==12.0 and scat_cut==10e-3 and width_cut==None:
            assert( len(cat_cut)==265 and len(det_frbs_cut)==18487 )

    # also return bins for the data associated with these cuts
    if catalog:
        bins = {
            'snr_log_data_bins': np.logspace(np.log10(snr_cut), np.log10(300), 100),
            'dm_log_data_bins': np.logspace(np.log10(dm_cut), np.log10(5000), 100),
            'dm_log_data_plot_bins': np.logspace(np.log10(dm_cut), np.log10(5000), 26),
            'width_log_data_bins': np.concatenate((
                np.array([1e-6, 2.5e-4, 5e-4, 7.5e-4, 1e-3]),
                np.logspace(np.log10(1e-3), np.log10(4e-2), 61)[1:]
            )),
            'width_log_data_plot_bins': np.logspace(np.log10(3e-5), np.log10(4e-2), 26),
            'scat_log_data_bins': np.concatenate((
                np.array([1e-6, 5e-4, 1e-3, 1.5e-3, 2e-3]),
                np.logspace(np.log10(2e-3), np.log10(scat_cut), 61)[1:]
            )),
            'scat_log_data_plot_bins': np.concatenate((
                np.logspace(np.log10(8e-5), np.log10(scat_cut), 18)[:-1],
                np.logspace(np.log10(scat_cut), np.log10(0.10), 26-17))),
        }
    else:
        # these define the edges for the bins, so if i want 20 bins, i want 21 bin edges
        bins = {
            'snr_log_data_bins': np.logspace(np.log10(snr_cut), np.log10(200.0), 16),
            'dm_log_data_bins': np.logspace(np.log10(dm_cut), np.log10(3500.0), 21),
        }
    return cat_cut, cat_all_scat, all_frbs_cut, all_frbs_all_scat, det_frbs_cut, det_frbs_all_scat, bins


def get_weights(model, det_frbs_cut, probabilities_init):
    return model.probabilities(det_frbs_cut) / probabilities_init


