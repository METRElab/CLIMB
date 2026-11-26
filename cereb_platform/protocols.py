"""The inputs a user can pick. v0.1 offers delay and trace conditioning."""

ISI_MIN_MS = 200
ISI_MAX_MS = 2500
ISI_IDEAL_MS = (350, 1600)     # shown as the ideal range in the page
DELAY_DEFAULT_ISI_MS = 500

PROTOCOLS = {
    'delay': dict(
        label='Delay',
        isi_settable=True,
        description='CS stays on until the US ends; paired trials with 10% CS-alone probes.',
        phases=[dict(name='acquisition', type='paired', n_trials=150, isi=0.5,
                     cs_duration=None, probe_frac=0.1)],
    ),
    'trace': dict(
        label='Trace',
        isi_settable=False,
        description='250 ms CS, then a gap before the US at 750 ms; 10% probes.',
        phases=[dict(name='trace', type='paired', n_trials=300, isi=0.75,
                     cs_duration=0.25, probe_frac=0.1)],
    ),
}


def resolve(name, isi_ms=None):
    """Phase list for a protocol, with the ISI applied (delay only).

    Raises ValueError for an unknown protocol or an ISI outside 200-2500 ms."""
    if name not in PROTOCOLS:
        raise ValueError(f'unknown protocol {name!r}; choose from {sorted(PROTOCOLS)}')
    spec = PROTOCOLS[name]
    phases = [dict(ph) for ph in spec['phases']]
    if isi_ms is not None:
        if not spec['isi_settable']:
            raise ValueError(f'the ISI is fixed for {name} conditioning')
        isi_ms = float(isi_ms)
        if not ISI_MIN_MS <= isi_ms <= ISI_MAX_MS:
            raise ValueError(f'ISI must be between {ISI_MIN_MS} and {ISI_MAX_MS} ms')
        for ph in phases:
            ph['isi'] = isi_ms / 1000
    return phases


def isi_ms_of(name, isi_ms=None):
    phases = resolve(name, isi_ms)
    return round(phases[0]['isi'] * 1000)


def config():
    """What the page needs to draw its controls."""
    return dict(
        protocols=[dict(id=k, label=v['label'], isi_settable=v['isi_settable'],
                        description=v['description'],
                        isi_ms=round(v['phases'][0]['isi'] * 1000),
                        n_trials=sum(ph['n_trials'] for ph in v['phases']))
                   for k, v in PROTOCOLS.items()],
        isi=dict(min=ISI_MIN_MS, max=ISI_MAX_MS, ideal=list(ISI_IDEAL_MS),
                 default=DELAY_DEFAULT_ISI_MS),
    )
