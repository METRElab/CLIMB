import pytest

from cereb_platform import ParamError, protocols
from cereb_platform.loader import discover_models, load_model
from cereb_platform.params import BASE_PARAMS
from conftest import FAKE_MODELS


def test_default_learning_rate():
    assert BASE_PARAMS['pfpc_learning_rate'] == 2e-5


def test_model_may_override_whitelisted_keys():
    cls = load_model(FAKE_MODELS / 'fake_slow.py')
    p = cls.merge_params()
    assert p['pfpc_learning_rate'] == 30
    assert cls.overrides_vs_base(p) == {'pfpc_learning_rate': dict(base=2e-5, used=30)}


def test_model_may_not_override_locked_keys():
    with pytest.raises(ParamError, match='n_pkj'):
        load_model(FAKE_MODELS / 'bad_locked.py')


def test_user_overrides_follow_the_whitelist():
    cls = load_model(FAKE_MODELS / 'fake_fast.py')
    assert cls.merge_params(overrides={'n_granule': 800, 'learn_trials': 5})['n_granule'] == 800
    with pytest.raises(ParamError, match='cf_gain'):
        cls.merge_params(overrides={'cf_gain': 3.0})
    with pytest.raises(ParamError, match='unknown'):
        cls.merge_params(overrides={'nonsense': 1})


def test_discovery_skips_broken_models():
    models, errors = discover_models(FAKE_MODELS)
    assert set(models) == {'fake_fast', 'fake_slow'}
    assert 'bad_locked.py' in errors


def test_protocol_isi_range():
    assert protocols.resolve('delay', 1200)[0]['isi'] == 1.2
    assert protocols.resolve('delay')[0]['isi'] == 0.5
    for bad in (199, 2501):
        with pytest.raises(ValueError):
            protocols.resolve('delay', bad)
    with pytest.raises(ValueError, match='fixed'):
        protocols.resolve('trace', 500)
    with pytest.raises(ValueError, match='unknown'):
        protocols.resolve('two_isi')
    cfg = protocols.config()
    assert [p['id'] for p in cfg['protocols']] == ['delay', 'trace']
    assert cfg['isi'] == dict(min=200, max=2500, ideal=[350, 1600], default=500)
