"""The reference circuit (ported from eyeblink/circuit.py).

Only the granule layer is swappable. The rest is fixed, apart from the keys in
params.OVERRIDABLE. Compared with the original, the metronome channel is gone and an
``eyelid_tap`` lets the live runner read the eyelid every step without a growing probe.
"""

import numpy as np
import nengo

from .stimulus import array_node, make_delay


def build_model(stim, p, layer, eyelid_tap=None):
    """Build the network. ``eyelid_tap(t, x)``, if given, is called every step with the
    eyelid value."""
    dt = p['dt']
    model = nengo.Network(seed=p['seed'])
    with model:
        # stimuli
        cs = nengo.Node(array_node(stim['cs'], dt), label='CS')
        us = nengo.Node(array_node(stim['us'], dt), label='US')
        cf = nengo.Node(array_node(stim['cf'], dt), label='CF burst')
        mf_noise = nengo.Node(array_node(stim['mf_noise'], dt), label='MF noise')
        bias = nengo.Node([1.0], label='tonic drive')

        # delayed + noisy mf input (channel 0 = CS)
        delay_node = nengo.Node(make_delay(p['mf_delay'], dimensions=1, dt=dt),
                                size_in=1, size_out=1)
        nengo.Connection(cs, delay_node, synapse=None)
        cereb_sens = nengo.Ensemble(n_neurons=100, dimensions=1, radius=2)
        nengo.Connection(delay_node, cereb_sens)
        nengo.Connection(mf_noise, cereb_sens)
        cs_mf = delay_node

        # granule-Golgi layer (plug-in)
        granule = layer.build(cereb_sens)
        if not isinstance(granule, nengo.Ensemble):
            raise TypeError(f'{layer.name}: build() must return a nengo.Ensemble, '
                            f'got {type(granule).__name__}')

        # PCs + learning (eligibility trace)
        pkj = nengo.Ensemble(n_neurons=p['n_pkj'], dimensions=1, radius=2, seed=1)
        nengo.Connection(bias, pkj, transform=p['pc_baseline'])
        learn = nengo.Connection(
            granule.neurons, pkj,
            transform=np.zeros((1, granule.n_neurons)),
            learning_rule_type=nengo.learning_rules.PES(
                learning_rate=p['pfpc_learning_rate'],
                pre_synapse=nengo.synapses.Alpha(p['elig_tau'])))

        # cerebellar nuclei: PC and mf inputs
        cn = nengo.Ensemble(n_neurons=p['n_cn'], dimensions=1, radius=2)
        nengo.Connection(bias, cn, transform=p['pc_baseline'])
        nengo.Connection(pkj, cn, transform=-1)
        learn_mfcn = None

        if p['mfcn_plasticity']:
            mf_col = nengo.Ensemble(n_neurons=100, dimensions=1, radius=1.5,
                                    encoders=nengo.dists.Choice([[1]]),
                                    intercepts=nengo.dists.Uniform(0.3, 0.9),
                                    label='MF collaterals')
            nengo.Connection(cs_mf, mf_col)
            nengo.Connection(mf_noise, mf_col)
            learn_mfcn = nengo.Connection(
                mf_col, cn, function=lambda x: [0.0],
                learning_rule_type=nengo.learning_rules.PES(
                    learning_rate=p['mfcn_learning_rate']))

        cereb_output = nengo.Node(None, size_in=1)
        nengo.Connection(cn, cereb_output, synapse=None)

        # error + learning gate (always on)
        gate = nengo.Node([0])

        no_delay_node = nengo.Node(make_delay(p['no_delay'], dimensions=1, dt=dt),
                                   size_in=1, size_out=1)
        nengo.Connection(cn, no_delay_node, synapse=p['no_tau'])

        def io_fn(t, x):
            cf_in, no_in, g = x
            io_act = max(p['io_baseline'] + p['cf_gain'] * cf_in - p['no_gain'] * no_in, 0.0)
            return (io_act - p['io_baseline']) * (1.0 - g)
        io = nengo.Node(io_fn, size_in=3, size_out=1, label='IO')
        nengo.Connection(cf, io[0], synapse=None)
        nengo.Connection(no_delay_node, io[1], synapse=None)
        nengo.Connection(gate, io[2], synapse=None)
        nengo.Connection(io, learn.learning_rule, synapse=None)

        if p['mfcn_plasticity']:
            mfcn_error = nengo.Ensemble(n_neurons=200, dimensions=1, radius=2)
            nengo.Connection(pkj, mfcn_error)
            nengo.Connection(bias, mfcn_error, transform=-p['pc_baseline'])
            nengo.Connection(mfcn_error, learn_mfcn.learning_rule)
            nengo.Connection(gate, mfcn_error.neurons, transform=[[-10]] * 200)

        # eyelid motor: CR (from CN) + UR (from US)
        ur_delay = nengo.Node(make_delay(p['ur_latency'], dimensions=1, dt=dt),
                              size_in=1, size_out=1)
        nengo.Connection(us, ur_delay, synapse=None)

        drive = nengo.Node(lambda t, x: p['cr_gain'] * max(x[0], 0.0) + p['ur_gain'] * x[1],
                           size_in=2)
        nengo.Connection(cereb_output, drive[0], synapse=0.005)
        nengo.Connection(ur_delay, drive[1], synapse=None)

        eyelid = nengo.Node(lambda t, x: np.tanh(max(x[0], 0.0)), size_in=1)
        nengo.Connection(drive, eyelid, synapse=nengo.synapses.Alpha(p['plant_tau']))

        if eyelid_tap is not None:
            tap = nengo.Node(eyelid_tap, size_in=1, size_out=0, label='eyelid tap')
            nengo.Connection(eyelid, tap, synapse=None)

        probes = dict(
            cs=nengo.Probe(cs, synapse=None),
            us=nengo.Probe(us, synapse=None),
            cn=nengo.Probe(cn, synapse=0.01),
            pkj=nengo.Probe(pkj, synapse=0.01),
            io=nengo.Probe(io, synapse=None),
            eyelid=nengo.Probe(eyelid, synapse=None),
        )
        if learn_mfcn is not None:
            probes['mfcn'] = nengo.Probe(learn_mfcn, 'output', synapse=0.01)

    return model, probes
