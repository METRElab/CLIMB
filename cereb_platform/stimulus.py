"""Trial schedules and stimulus arrays (ported from eyeblink/stimulus.py, rhythm removed)."""

from collections import deque

import numpy as np
import pandas as pd
import scipy.signal


def _mf_noise(n, p, rng):
    """Low-pass filtered Gaussian noise for one mossy-fibre channel."""
    dt = p['dt']
    a = np.exp(-dt / p['mf_noise_tau'])
    noise = scipy.signal.lfilter([1 - a], [1, -a], rng.standard_normal(n))
    return noise * (p['mf_noise_std'] / noise.std())


def make_stimulus(trials, duration, p, seed=0):
    """CS, US, CF burst and MF noise arrays, one sample per time step."""
    dt = p['dt']
    n = int(np.ceil(duration / dt)) + 1
    cs, us, cf = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr in trials.itertuples():
        i0 = int(round(tr.start / dt))
        cs[i0:i0 + int(round(tr.cs_duration / dt))] = 1.0
        if tr.us_present:
            j0 = int(round((tr.start + tr.isi) / dt))
            us[j0:j0 + int(round(tr.us_duration / dt))] = 1.0
            reshow = getattr(tr, 'cs_reshow', 0.0)
            if reshow > 0:
                cs[j0:j0 + int(round(reshow / dt))] = 1.0
            k0 = j0 + int(round(p['cf_latency'] / dt))
            cf[k0:k0 + max(1, int(round(p['cf_duration'] / dt)))] = 1.0

    noise = np.zeros(n)
    if p['mf_noise_std'] > 0:
        noise = _mf_noise(n, p, np.random.default_rng(seed))
    return dict(cs=cs, us=us, cf=cf, mf_noise=noise)


def array_node(arr, dt):
    last = len(arr) - 1

    def f(t):
        return arr[min(int(round(t / dt)), last)]
    return f


def build_schedule(phases, iti_range=(4.0, 6.0), t0=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, t = [], t0
    for ph in phases:
        isis = np.atleast_1d(ph.get('isi', 0.5)).astype(float)
        us_dur = ph.get('us_duration', 0.2)
        n = ph['n_trials']
        probe_frac = ph.get('probe_frac', 0.0)

        probe_mask = np.zeros(n, dtype=bool)
        if ph['type'] == 'paired' and probe_frac > 0:
            block = int(round(1 / probe_frac))
            for b in range(0, n, block):
                probe_mask[b + rng.integers(0, min(block, n - b))] = True

        for i in range(n):
            isi = float(rng.choice(isis))
            cs_dur = ph.get('cs_duration')
            if cs_dur is None:
                cs_dur = isi + us_dur
            if ph['type'] == 'cs_alone':
                ttype = 'cs_alone'
            elif probe_mask[i]:
                ttype = 'probe'
            else:
                ttype = 'paired'
            rows.append(dict(phase=ph['name'], trial_type=ttype, start=t, isi=isi,
                             cs_duration=cs_dur, us_duration=us_dur,
                             us_present=(ttype == 'paired'), max_isi=float(isis.max()),
                             cs_reshow=float(ph.get('cs_reshow', 0.0))))
            t += rng.uniform(*iti_range)

    trials = pd.DataFrame(rows)
    trials.index.name = 'trial'
    return trials, t


def make_delay(delay, dimensions, dt=0.001):
    n_steps = int(round(delay / dt))
    if n_steps == 0:
        return lambda t, x: x
    buf = deque([np.zeros(dimensions)] * n_steps)

    def delay_fn(t, x):
        buf.append(x.copy())
        return buf.popleft()
    return delay_fn


def rising_edges(x, threshold=0.5):
    """Indices where x goes from below threshold to at or above it."""
    above = np.asarray(x) >= threshold
    return np.flatnonzero(above[1:] & ~above[:-1]) + 1


def falling_edges(x, threshold=0.5):
    above = np.asarray(x) >= threshold
    return np.flatnonzero(~above[1:] & above[:-1]) + 1
