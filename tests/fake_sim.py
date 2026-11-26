"""A stand-in for the Nengo simulator: an eyelid that learns a CR timed to the ISI."""
import numpy as np

from cereb_platform.experiment import Experiment


class FakeExperiment(Experiment):
    def build(self):
        n = len(self.eyelid)
        target = np.zeros(n)
        learn = self.p['learn_trials']
        for k, tr in enumerate(self.trials.itertuples()):
            amp = min(1.0, k / learn) * 0.6
            i_cs = int(round(tr.start / self.dt))
            i_us = int(round((tr.start + tr.isi) / self.dt))
            width = int(0.12 / self.dt)
            idx = np.arange(i_us - 3 * width, i_us + width)
            idx = idx[(idx >= 0) & (idx < n) & (idx > i_cs + 100)]
            target[idx] = np.maximum(target[idx],
                                     amp * np.exp(-0.5 * ((idx - i_us) / width) ** 2))
            if tr.us_present:
                j = np.arange(i_us + 15, min(i_us + 250, n))
                target[j] = np.maximum(target[j], 0.9 * np.exp(-(j - i_us - 15) / 60))
        self._target = target
        self._n = 0

    def step(self, n):
        i0, i1 = self._n + 1, self._n + int(n)
        self.eyelid[i0:i1 + 1] = self._target[i0:i1 + 1]
        self._n = i1

    @property
    def n_done(self):
        return getattr(self, '_n', 0)

    def close(self):
        pass

    def full_res(self):
        n = self.n_done
        z = np.zeros(n)
        data = dict(eyelid=self.eyelid[1:n + 1], cs=self.stim['cs'][1:n + 1],
                    us=self.stim['us'][1:n + 1], pkj=z, cn=z, io=z)
        return dict(t=np.arange(1, n + 1) * self.dt, data=data, trials=self.trials,
                    params=dict(self.p), label=self.protocol, model=self.name)


class MismatchedInputExperiment(FakeExperiment):
    """For tests: fake_slow gets slightly different mossy-fibre noise."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if self.name == 'fake_slow':
            self.stim['mf_noise'] = self.stim['mf_noise'] * 1.01
