"""Reference circuit parameters.

Everything here belongs to the fixed (reference) circuit. A granule-layer model may
change only the keys in OVERRIDABLE; anything else it sets must be one of its own keys
(a name that is not in BASE_PARAMS).
"""

BASE_PARAMS = dict(
    seed=1,
    dt=0.001,

    n_granule=2000,

    # mf input
    mf_delay=0.1,
    mf_noise_std=0.1,
    mf_noise_tau=0.02,        # time constant of noise (s)
    granule_noise_std=0.0,

    # PCs and cerebellar nuclei
    n_pkj=200,
    n_cn=200,
    pc_baseline=1.0,          # tonic PC activity
    pfpc_learning_rate=2e-5,  # pf -> PC (cortical site); models may override
    elig_tau=0.05,            # eligibility trace
    mfcn_plasticity=False,    # mf -> CN (nuclear site)
    mfcn_learning_rate=2e-6,  # slower than the cortical site

    # climbing fibre / inferior olive error signal
    cf_latency=0.02,          # us onset -> cf arrival
    cf_duration=0.02,         # cf burst duration (s)
    cf_gain=10.0,             # cf burst -> IO
    io_baseline=1.0,          # tonic IO activity
    no_gain=1.0,              # nucleo-olivary inhibition
    no_delay=0.05,            # CN -> IO delay (s)
    no_tau=0.02,

    # eyelid motor
    cr_gain=1.0,              # CN output -> eyelid
    ur_gain=3.0,              # reflexive responses
    ur_latency=0.015,         # US -> UR latency (s)
    plant_tau=0.025,

    # protocol
    iti_range=(4.0, 6.0),
)

# Reference-circuit keys a granule-layer model is allowed to change.
OVERRIDABLE = frozenset({'n_granule', 'granule_noise_std', 'pfpc_learning_rate'})

# Eyelid level that counts as a blink (same threshold measure_crs uses for CRs).
BLINK_THRESHOLD = 0.1
