import numpy as np
import nengo

from cereb_platform import GranuleLayer

def _syn_gain(tau, dt): # weights are too weak in the model to work in nengo -> scale by this
    return 1.0 / (1.0 - np.exp(-dt / tau)) # modified version of nengo's scaling

def build_reservoir(mf, p):
    # formula 1
    dt, nz = p['dt'], p['n_granule']
    rng = np.random.default_rng(p['res_seed'])

    # push-pull MF input
    I0 = np.maximum(rng.normal(1.0, p['res_vI'], nz), 0)
    f = np.where(rng.random(nz) < p['res_pushpull'], -1.0, 1.0)

    granule = nengo.Ensemble(nz, 1, neuron_type=nengo.RectifiedLinear(),
                             gain=np.ones(nz), bias=I0, label='granule')
    if p['granule_noise_std'] > 0: # granule noise
        granule.noise = nengo.processes.WhiteNoise(
            dist=nengo.dists.Gaussian(0, p['granule_noise_std']), scale=False)
    
    W_in = (f * p['res_mf_mod'] * I0)[:, None]
    if mf.dimensions > 1:
        # extra MF channels (metronome): each granule cell gets its own push-pull sign.
        rng_ch = np.random.default_rng([p['res_seed'], 7])
        f_ch = np.where(rng_ch.random((nz, mf.dimensions - 1)) < p['res_pushpull'], -1.0, 1.0)
        W_in = np.hstack([W_in, f_ch * p['res_mf_mod'] * I0[:, None]])
    nengo.Connection(mf, granule.neurons, transform=W_in)

    # recurrent inhibition
    A = rng.random((nz, nz)) < p['res_a']
    W = A * np.maximum(p['res_w'] * (1 + p['res_vw'] * rng.standard_normal((nz, nz))), 0)
    nengo.Connection(granule.neurons, granule.neurons,
                     transform=-(2.0 / nz) * _syn_gain(p['res_tau_w'], dt) * W,
                     synapse=nengo.Lowpass(p['res_tau_w']))
    return granule

class Reservoir(GranuleLayer):
    name = 'reservoir'
    defaults = dict(
        res_seed=1, res_w=14, res_tau_w=0.05, res_a=0.04, res_mf_mod=0.1,
        res_vI=0.1, res_pushpull=0.5, res_vw=0.0,
        pfpc_learning_rate=30)

    def build(self, mf):
        self.granule = build_reservoir(mf, self.p)
        return self.granule
