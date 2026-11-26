"""One model on one protocol, built so it can be stepped trial by trial."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import __version__, protocols
from .analysis import (false_cr_rate, learning_curve, measure_crs, summarize,
                       trials_to_criterion)
from .loader import load_model
from .stimulus import build_schedule, make_stimulus


class Experiment:
    """Schedule, stimulus and (once built) a Nengo simulator for one model.

    The live runner only uses: ``trials``, ``dt``, ``n_total``, ``start_steps``,
    ``stim``, ``eyelid``, ``n_done``, ``build()``, ``step(n)``, ``partial_meas(k)`` and
    ``finish(outdir)``. Tests replace build/step/finish with a fake simulator."""

    def __init__(self, model_spec, protocol='delay', isi_ms=None, seed=1, overrides=None,
                 phases=None, base_patch=None):
        self.model_spec = str(model_spec)
        self.layer_cls = load_model(model_spec)
        self.p = self.layer_cls.merge_params(overrides={**(overrides or {}), 'seed': seed})
        if base_patch:                 # internal only (e.g. short ITIs for `check`)
            self.p.update(base_patch)
        self.protocol = protocol
        self.phases = phases if phases is not None else protocols.resolve(protocol, isi_ms)
        self.isi_ms = round(self.phases[0]['isi'] * 1000)
        self.dt = self.p['dt']
        self.trials, self.duration = build_schedule(self.phases, self.p['iti_range'],
                                                    seed=self.p['seed'])
        self.stim = make_stimulus(self.trials, self.duration, self.p, seed=self.p['seed'])
        self.n_total = int(round(self.duration / self.dt))
        self.start_steps = np.round(self.trials['start'].values / self.dt).astype(int)
        self.eyelid = np.zeros(self.n_total + 2)
        self.sim = None
        self.probes = None

    def input_fingerprint(self):
        """Short hash of everything the model receives: the trial schedule and the CS, US,
        climbing-fibre and mossy-fibre noise arrays. Two models given the same input have
        the same fingerprint."""
        h = hashlib.sha256()
        cols = ['phase', 'trial_type', 'start', 'isi', 'cs_duration', 'us_duration']
        h.update(self.trials[cols].to_csv(index=False).encode())
        for key in ('cs', 'us', 'cf', 'mf_noise'):
            h.update(np.ascontiguousarray(self.stim[key], dtype=np.float64).tobytes())
        h.update(repr((self.dt, self.p['mf_delay'])).encode())
        return h.hexdigest()[:16]

    @property
    def name(self):
        return self.layer_cls.name

    @property
    def n_trials(self):
        return len(self.trials)

    # ------------------------------------------------------------------ simulation
    def _tap(self, t, x):
        i = int(round(t / self.dt))
        if i < len(self.eyelid):
            self.eyelid[i] = x[0]

    def build(self):
        import nengo
        from .circuit import build_model
        layer = self.layer_cls(self.p)                     # fresh layer per run
        model, self.probes = build_model(self.stim, self.p, layer, eyelid_tap=self._tap)
        self.sim = nengo.Simulator(model, dt=self.dt, seed=self.p['seed'], progress_bar=False)

    def step(self, n):
        self.sim.run_steps(int(n))

    @property
    def n_done(self):
        return 0 if self.sim is None else int(self.sim.n_steps)

    def close(self):
        if self.sim is not None and not getattr(self, '_closed', False):
            self._closed = True
            self.sim.close()

    # ------------------------------------------------------------------ scoring
    def partial_res(self, n_trials):
        n = self.n_done
        return dict(t=np.arange(1, n + 1) * self.dt, data=dict(eyelid=self.eyelid[1:n + 1]),
                    trials=self.trials.iloc[:n_trials])

    def partial_meas(self, n_trials):
        """measure_crs on the first ``n_trials`` trials (all must be complete)."""
        return measure_crs(self.partial_res(n_trials))[0]

    def full_res(self):
        data = {k: np.asarray(self.sim.data[pr]).squeeze() for k, pr in self.probes.items()}
        return dict(t=self.sim.trange(), data=data, trials=self.trials, params=dict(self.p),
                    label=self.protocol, model=self.name)

    def metadata(self):
        return dict(version=__version__, model=self.name, model_file=self.model_spec,
                    protocol=self.protocol, isi_ms=self.isi_ms, seed=self.p['seed'],
                    input_fingerprint=self.input_fingerprint(),
                    n_trials=self.n_trials, params=self.p,
                    overrides_vs_base=self.layer_cls.overrides_vs_base(self.p))

    def finish(self, outdir):
        """Score the whole run, save tables, figures and params.json, close the simulator.
        Returns the full trial table."""
        from . import plots
        import matplotlib.pyplot as plt
        res = self.full_res()
        self.close()
        meas, rel, amp = measure_crs(res)
        fp, n_fp = false_cr_rate(res)
        scalars = dict(false_cr_pct=fp, false_cr_windows=n_fp)
        for ph in dict.fromkeys(meas['phase']):
            scalars[f'criterion_block_{ph}'] = trials_to_criterion(meas, ph)
        tables = dict(trials=meas, summary=summarize(meas), learning_curve=learning_curve(meas))
        figures = dict(aligned=plots.plot_aligned(res, meas), learning=plots.plot_learning(meas),
                       probe_traces=plots.plot_probe_traces(meas, rel, amp))

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.reset_index().to_csv(outdir / f'{name}.csv', index=False)
        import pandas as pd
        pd.DataFrame([scalars]).to_csv(outdir / 'scalars.csv', index=False)
        for name, fig in figures.items():
            if fig is not None:
                fig.savefig(outdir / f'{name}.png', dpi=150)
                plt.close(fig)
        np.savez_compressed(outdir / 'traces.npz', rel=rel, amp=amp)
        (outdir / 'params.json').write_text(json.dumps(self.metadata(), indent=2, default=str))
        return meas


def timestamp():
    return time.strftime('%Y%m%d-%H%M%S')
