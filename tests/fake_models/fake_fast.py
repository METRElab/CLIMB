"""A model file for tests: never built (the fake simulator stands in for Nengo)."""
from cereb_platform import GranuleLayer


class FakeFast(GranuleLayer):
    name = 'fake_fast'
    defaults = dict(learn_trials=30, n_granule=500)

    def build(self, mf):
        raise RuntimeError('not used in tests')
