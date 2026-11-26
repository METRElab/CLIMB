from cereb_platform import GranuleLayer


class FakeSlow(GranuleLayer):
    name = 'fake_slow'
    defaults = dict(learn_trials=90, pfpc_learning_rate=30)

    def build(self, mf):
        raise RuntimeError('not used in tests')
