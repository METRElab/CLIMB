from cereb_platform import GranuleLayer


class BadLocked(GranuleLayer):
    name = 'bad_locked'
    defaults = dict(n_pkj=50)      # a reference-circuit key models may not change

    def build(self, mf):
        raise RuntimeError('not used in tests')
