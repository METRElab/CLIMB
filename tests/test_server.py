import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from cereb_platform.server import start_in_thread
from conftest import FAKE_MODELS

FACTORY = 'fake_sim:FakeExperiment'


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(FAKE_MODELS.parent))     # spawned workers find fake_sim
    app, httpd, url = start_in_thread(models_dir=FAKE_MODELS, out_root=tmp_path,
                                      factory=FACTORY)
    yield app, url, tmp_path
    app.shutdown()
    httpd.shutdown()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def listen(url, until='finished', timeout=120):
    events, done = [], threading.Event()

    def reader():
        with urllib.request.urlopen(url + 'api/events', timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith('data: '):
                    ev = json.loads(line[6:])
                    events.append(ev)
                    if ev['type'] == until:
                        done.set()
                        return
    threading.Thread(target=reader, daemon=True).start()
    time.sleep(0.2)
    return events, done


def test_page_and_config(server):
    app, url, _ = server
    with pytest.raises(urllib.error.HTTPError) as err:      # nothing to save yet
        urllib.request.urlopen(url + 'api/plot.png')
    assert err.value.code == 404
    html = urllib.request.urlopen(url).read().decode()
    assert 'Cerebellar testing platform' in html
    cfg = json.loads(urllib.request.urlopen(url + 'api/config').read())
    assert cfg['models'] == ['fake_fast', 'fake_slow']
    assert 'bad_locked.py' in cfg['model_errors']
    assert cfg['isi']['ideal'] == [350, 1600]
    assert cfg['blink_threshold'] == 0.1


def test_rejects_bad_requests(server):
    _, url, _ = server
    assert post(url + 'api/run', dict(model_a='fake_fast', protocol='delay', isi_ms=3000))[0] == 400
    assert post(url + 'api/run', dict(model_a='nope', protocol='delay'))[0] == 400
    assert post(url + 'api/run', dict(model_a='fake_fast', protocol='rhythm_v1'))[0] == 400


def test_two_models_live_end_to_end(server):
    _, url, out = server
    events, done = listen(url)
    status, body = post(url + 'api/run', dict(model_a='fake_fast', model_b='fake_slow',
                                              protocol='delay', isi_ms=800, speed=None))
    assert status == 200, body
    assert done.wait(120), [e['type'] for e in events][-5:]
    kinds = [e['type'] for e in events]
    assert kinds[0] == 'session'
    plots = [e for e in events if e['type'] == 'plot']
    assert [p['trials_done'] for p in plots][:15] == list(range(10, 151, 10))
    assert all(len(p['png']) > 1000 for p in plots)
    for label in 'AB':
        trials = [e for e in events if e['type'] == 'trial' and e['model'] == label]
        assert len(trials) == 150
        assert any(e['type'] == 'blink' and e['model'] == label for e in events)
        assert any(e['type'] == 'trace' and e['model'] == label for e in events)
    ready = [e for e in events if e['type'] == 'ready']
    assert len(ready) == 2 and ready[0]['input_fingerprint'] == ready[1]['input_fingerprint']
    assert ready[0]['duration'] == ready[1]['duration'] > 150 * 4
    fin = events[-1]
    assert fin['state'] == {'A': 'done', 'B': 'done'}
    session = out / fin['outdir'].split('/')[-1].split('\\')[-1]
    assert (session / 'learning_compare.png').exists()
    assert (session / 'session.json').exists()
    assert (session / 'A_fake_fast' / 'params.json').exists()
    meta = json.loads((session / 'B_fake_slow' / 'params.json').read_text())
    assert meta['isi_ms'] == 800
    fps = json.loads((session / 'session.json').read_text())['input_fingerprints']
    assert fps['A'] == fps['B'] == meta['input_fingerprint']
    # Save simulation: the latest plot at full resolution, as a named download
    with urllib.request.urlopen(url + 'api/plot.png') as r:
        png = r.read()
        disposition = r.headers['Content-Disposition']
    assert png[:8] == b'\x89PNG\r\n\x1a\n' and len(png) > 20000
    assert 'learning_delay_isi800_fake_fast_vs_fake_slow_trial150_' in disposition


def test_stop_and_speed(server):
    _, url, _ = server
    events, done = listen(url)
    assert post(url + 'api/run', dict(model_a='fake_fast', protocol='trace', speed=1.0))[0] == 200
    time.sleep(3)
    assert post(url + 'api/speed', dict(speed=5.0))[0] == 200
    assert post(url + 'api/stop', {})[0] == 200
    assert done.wait(30)
    assert events[-1]['state'] == {'A': 'stopped'}
    trials = [e for e in events if e['type'] == 'trial']
    assert 0 < len(trials) < 300


def test_different_inputs_stop_the_run(tmp_path, monkeypatch):
    from cereb_platform.server import start_in_thread as start
    monkeypatch.syspath_prepend(str(FAKE_MODELS.parent))
    app, httpd, url = start(models_dir=FAKE_MODELS, out_root=tmp_path,
                            factory='fake_sim:MismatchedInputExperiment')
    try:
        events, done = listen(url)
        assert post(url + 'api/run', dict(model_a='fake_fast', model_b='fake_slow',
                                          protocol='delay', speed=1.0))[0] == 200
        assert done.wait(60)
        errors = [e for e in events if e['type'] == 'error']
        assert errors and 'different inputs' in errors[0]['message']
        assert events[-1]['state'] == {'A': 'stopped', 'B': 'stopped'}
        assert len([e for e in events if e['type'] == 'trial']) <= 4   # stopped at once
    finally:
        app.shutdown()
        httpd.shutdown()


def test_pause_and_resume(server):
    _, url, _ = server
    events, done = listen(url)
    assert post(url + 'api/run', dict(model_a='fake_fast', model_b='fake_slow',
                                      protocol='delay', speed=5.0))[0] == 200
    time.sleep(3)
    assert post(url + 'api/pause', {})[0] == 200
    time.sleep(0.5)
    n_paused = len([e for e in events if e['type'] == 'trial'])
    time.sleep(2)
    assert len([e for e in events if e['type'] == 'trial']) == n_paused   # nothing moved
    assert post(url + 'api/resume', {})[0] == 200
    time.sleep(3)
    trials = [e for e in events if e['type'] == 'trial']
    assert len(trials) > n_paused
    for label in 'AB':                 # each model carried on without skipping a trial
        idx = [e['index'] for e in trials if e['model'] == label]
        assert idx == list(range(len(idx)))
    kinds = [e['type'] for e in events]
    assert kinds.count('paused') == 1 and kinds.count('resumed') == 1
    assert post(url + 'api/stop', {})[0] == 200
    assert done.wait(30)
    assert post(url + 'api/pause', {})[0] == 400      # nothing running
