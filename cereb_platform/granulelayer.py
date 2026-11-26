"""The granule-layer contract.

A granule-layer model is a subclass of GranuleLayer that builds a Nengo ensemble.

Input:  ``mf``, a 1-D ``nengo.Ensemble`` carrying the mossy-fibre signal. Its decoded
        value is the conditioned stimulus (1 while the CS is on, 0 otherwise) after a
        ``mf_delay`` (100 ms) delay, plus low-pass filtered noise (``mf_noise_std``).
Output: a ``nengo.Ensemble``. Its neurons are the parallel fibres: the reference
        circuit connects ``output.neurons`` to the Purkinje cells through the learned
        PF->PC connection.

Parameters: a model's ``defaults`` may set any of its own keys, plus the reference keys
in ``params.OVERRIDABLE`` (n_granule, granule_noise_std, pfpc_learning_rate). Setting any
other reference key raises an error, so the downstream circuit is the same for every
model.
"""

from abc import ABC, abstractmethod

from .params import BASE_PARAMS, OVERRIDABLE


class ParamError(ValueError):
    """A model or a user tried to set a parameter it may not set."""


class GranuleLayer(ABC):
    """Base class for a granule(-Golgi) layer that plugs into the reference circuit."""

    name = 'base'
    defaults = {}

    @classmethod
    def own_keys(cls):
        """Keys that belong to this model rather than to the reference circuit."""
        return set(cls.defaults) - set(BASE_PARAMS)

    @classmethod
    def check_defaults(cls):
        locked = sorted((set(cls.defaults) & set(BASE_PARAMS)) - OVERRIDABLE)
        if locked:
            raise ParamError(
                f'{cls.name}: defaults set reference-circuit key(s) {locked}, which models '
                f'may not change. Allowed reference keys: {sorted(OVERRIDABLE)}.')

    @classmethod
    def merge_params(cls, base=None, overrides=None):
        """Reference params + this model's defaults + user overrides, with the whitelist
        enforced on both the defaults and the overrides."""
        cls.check_defaults()
        base = dict(BASE_PARAMS if base is None else base)
        p = {**base, **cls.defaults}
        overrides = dict(overrides or {})
        allowed = OVERRIDABLE | cls.own_keys() | {'seed'}
        unknown = sorted(set(overrides) - set(p))
        if unknown:
            raise ParamError(f'{cls.name}: unknown parameter(s) {unknown}')
        locked = sorted(set(overrides) - allowed)
        if locked:
            raise ParamError(f'{cls.name}: {locked} belong to the reference circuit and '
                             f'cannot be changed. Allowed: {sorted(allowed)}')
        p.update(overrides)
        return p

    @classmethod
    def overrides_vs_base(cls, p):
        """Reference keys whose value differs from BASE_PARAMS (logged with every run)."""
        return {k: dict(base=BASE_PARAMS[k], used=p[k])
                for k in sorted(OVERRIDABLE) if p.get(k) != BASE_PARAMS[k]}

    def __init__(self, p):
        self.p = p
        self.granule = None

    @abstractmethod
    def build(self, mf):
        """Build the layer inside the current nengo.Network and return its ensemble.

        ``mf.dimensions`` is 1 in v0.1 (channel 0 = CS)."""
