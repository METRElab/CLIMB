"""CR scoring (ported unchanged from eyeblink/analysis.py, rhythm functions removed)."""

import numpy as np
import pandas as pd

def trial_traces(res, signal='eyelid', pre=0.2, post=None, starts=None):
    t, x, trials = res['t'], res['data'][signal], res['trials']
    dt = t[1] - t[0]
    if post is None:
        post = float((trials['isi'] + trials['us_duration']).max()) + 0.5
    if starts is None:
        starts = trials['start'].values
    offs = np.arange(int(round(-pre / dt)), int(round(post / dt)))
    i0 = np.searchsorted(t, np.asarray(starts) - 1e-9)
    idx = np.clip(i0[:, None] + offs[None, :], 0, len(t) - 1)
    return offs * dt, x[idx]

def first_sustained(mask, n_min):
    run = np.zeros(len(mask) + 1, dtype=int)
    for i in range(len(mask) - 1, -1, -1):
        run[i] = run[i + 1] + 1 if mask[i] else 0
    to_end = len(mask) - np.arange(len(mask))
    ok = mask & ((run[:-1] >= n_min) | (run[:-1] == to_end))
    idx = np.flatnonzero(ok)
    return int(idx[0]) if idx.size else None

def _score_windows(rel, amp, ends, cr_threshold, cr_min_dur, cr_window_start):
    dt = rel[1] - rel[0]
    n_min = max(1, int(round(cr_min_dur / dt)))
    out = []
    for k, end in enumerate(ends):
        w = (rel >= cr_window_start) & (rel < end)
        a, tw = amp[k, w], rel[w]
        i_on = first_sustained(a >= cr_threshold, n_min)
        pk = int(np.argmax(a))
        out.append((i_on is not None, tw[i_on] if i_on is not None else np.nan, a[pk], tw[pk]))
    return out

def measure_crs(res, cr_threshold=0.1, cr_min_dur=0.02, cr_window_start=0.1, probe_extra=0.3, pre=0.2):
    rel, tr = trial_traces(res, 'eyelid', pre=pre)
    amp = tr - tr[:, rel < 0].mean(axis=1, keepdims=True)
    trials = res['trials'].copy()
    ends = np.where(trials.us_present, trials.isi, trials.max_isi + probe_extra)
    scored = _score_windows(rel, amp, ends, cr_threshold, cr_min_dur, cr_window_start)
 

    out = []
    for k, (row, (is_cr, onset, pk_amp, pk_time)) in enumerate(zip(trials.itertuples(), scored)):
        unmasked = not row.us_present            # no UR in the window -> the peak is meaningful
        i_us = min(np.searchsorted(rel, row.isi), len(rel) - 1)
        out.append(dict(cr=is_cr, onset=onset,
                        amp_at_us=amp[k, i_us],
                        peak_amp=pk_amp if unmasked else np.nan,
                        peak_time=pk_time if (unmasked and is_cr) else np.nan))
    trials = trials.join(pd.DataFrame(out, index=trials.index))
    trials['onset_rel_us'] = trials['onset'] - trials['isi']
    trials['peak_rel_us'] = trials['peak_time'] - trials['isi']
    trials['trial_in_phase'] = trials.groupby('phase', sort=False).cumcount()
    trials['block'] = trials['trial_in_phase'] // 10
    return trials, rel, amp

def false_cr_rate(res, cr_threshold=0.1, cr_min_dur=0.02, cr_window_start=0.1, probe_extra=0.3,
                  pre=0.2, margin=0.5):
    trials = res['trials']
    win = float(trials.max_isi.max()) + probe_extra          
    fake = trials['start'].values - 0.1 - win
    prev_end = np.r_[-np.inf, (trials['start'] + trials['isi'] + trials['us_duration']).values[:-1] + margin]
    keep = (fake - pre) > prev_end
    if not keep.any():
        return np.nan, 0
    rel, tr = trial_traces(res, 'eyelid', pre=pre, post=win, starts=fake[keep])
    amp = tr - tr[:, rel < 0].mean(axis=1, keepdims=True)
    scored = _score_windows(rel, amp, np.full(keep.sum(), win), cr_threshold, cr_min_dur, cr_window_start)
    return 100 * np.mean([s[0] for s in scored]), int(keep.sum())

def trials_to_criterion(meas, phase, crit_pct=80):
    blocks = meas[meas.phase == phase].groupby('block')['cr'].mean() * 100
    hit = blocks[blocks >= crit_pct]
    return None if hit.empty else int(hit.index[0])


def summarize_isi_sweep(results):
    rows = []
    for isi, res in results.items():
        m, _, _ = measure_crs(res)
        late = m[m.trial_in_phase >= m.trial_in_phase.max() / 2]
        paired, probe = late[late.trial_type == 'paired'], late[late.trial_type == 'probe']
        rows.append(dict(isi=isi, pct_cr=100 * late.cr.mean(),
                         amp_at_us=paired.amp_at_us.mean(),
                         onset_mean=paired.onset.mean(), onset_sd=paired.onset.std(),
                         peak_mean=probe.peak_time.mean(), peak_sd=probe.peak_time.std(),
                         n_probe=len(probe), n_probe_crs=int(probe.cr.sum())))
    s = pd.DataFrame(rows).set_index('isi')
    s['onset_cv'] = s.onset_sd / s.onset_mean
    s['peak_cv'] = s.peak_sd / s.peak_mean
    return s

def summarize(meas):
    summary = meas.groupby(['phase', 'trial_type'], sort=False).agg(
        n=('cr', 'size'), pct_cr=('cr', 'mean'),
        onset_ms=('onset', 'mean'), onset_sd_ms=('onset', 'std'), amp_at_us=('amp_at_us', 'mean'),
        peak_ms=('peak_time', 'mean'), peak_sd_ms=('peak_time', 'std'), peak_amp=('peak_amp', 'mean'))
    summary['pct_cr'] *= 100
    summary[['onset_ms', 'onset_sd_ms', 'peak_ms', 'peak_sd_ms']] *= 1000
    return summary


def learning_curve(meas):
    keys = ['phase', 'block']
    lc = meas.groupby(keys, sort=False).agg(n=('cr', 'size'), pct_cr=('cr', 'mean'))
    lc['pct_cr'] *= 100
    paired = meas[meas.trial_type == 'paired'].groupby(keys, sort=False)
    lc['amp_at_us'] = paired['amp_at_us'].mean()
    lc['onset_rel_us_ms'] = paired['onset_rel_us'].mean() * 1000
    unmasked = meas[meas.trial_type != 'paired'].groupby(keys, sort=False)
    lc['probe_peak_amp'] = unmasked['peak_amp'].mean()
    return lc
