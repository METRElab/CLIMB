"""`check`: a fast smoke test for a granule-layer model file.

Runs in well under a minute for typical models: loads the file, checks the parameter
whitelist, builds the layer on its own, runs two short delay trials through the
reference circuit, checks the output and that the same seed gives the same result.
"""

import time

import numpy as np

from .experiment import Experiment
from .loader import load_model

CHECK_PHASES = [dict(name='check', type='paired', n_trials=2, isi=0.5, cs_duration=None,
                     probe_frac=0.0)]
CHECK_PATCH = dict(iti_range=(2.0, 2.0))
DETERMINISM_STEPS = 500


def _line(ok, text):
    print(f"  {'ok ' if ok else 'FAIL'} {text}")
    return ok


def run_check(model_spec, seed=1):
    print(f'checking {model_spec}')
    ok = True
    try:
        cls = load_model(model_spec)
        p = cls.merge_params(overrides={'seed': seed})
    except Exception as e:
        _line(False, f'load + parameters: {type(e).__name__}: {e}')
        return False
    _line(True, f'loads as {cls.__name__} (name={cls.name!r})')
    ov = cls.overrides_vs_base(p)
    _line(True, 'parameters within the whitelist' +
          (f'; overrides {", ".join(f"{k}={v["used"]}" for k, v in ov.items())}' if ov else ''))

    import nengo
    try:
        with nengo.Network(seed=seed):
            mf = nengo.Ensemble(100, 1, radius=2)
            out = cls(p).build(mf)
        if not isinstance(out, nengo.Ensemble):
            raise TypeError(f'build() returned {type(out).__name__}, expected nengo.Ensemble')
        ok &= _line(out.n_neurons > 0, f'build(mf) returns an Ensemble with {out.n_neurons} neurons')
    except Exception as e:
        _line(False, f'build(mf): {type(e).__name__}: {e}')
        return False

    try:
        exp = Experiment(model_spec, phases=CHECK_PHASES, seed=seed, base_patch=CHECK_PATCH)
        t0 = time.perf_counter()
        exp.build()
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        exp.step(exp.n_total)
        t_run = time.perf_counter() - t0
        exp.close()
    except Exception as e:
        _line(False, f'runs in the reference circuit: {type(e).__name__}: {e}')
        return False
    eyelid = exp.eyelid[1:exp.n_total + 1]
    ok &= _line(np.all(np.isfinite(eyelid)), 'eyelid output has no NaN or inf')
    ok &= _line(eyelid.min() >= 0 and eyelid.max() <= 1, f'eyelid stays in [0, 1] '
                f'(max {eyelid.max():.3f})')

    a = Experiment(model_spec, phases=CHECK_PHASES, seed=seed, base_patch=CHECK_PATCH)
    b = Experiment(model_spec, phases=CHECK_PHASES, seed=seed, base_patch=CHECK_PATCH)
    for e in (a, b):
        e.build(); e.step(DETERMINISM_STEPS); e.close()
    same = np.allclose(a.eyelid[:DETERMINISM_STEPS], b.eyelid[:DETERMINISM_STEPS])
    ok &= _line(same, 'same seed gives the same eyelid trace')

    rate = exp.n_total * exp.dt / t_run
    full = {}
    for name in ('delay', 'trace'):
        e = Experiment(model_spec, protocol=name, seed=seed)
        full[name] = e.duration / rate
    _line(True, f'built in {t_build:.1f} s; simulates {rate:.2f} s of model time per second')
    _line(True, 'a full run at max speed would take about ' +
          ', '.join(f'{v / 60:.0f} min ({k})' for k, v in full.items()))
    print('passed' if ok else 'FAILED')
    return bool(ok)
