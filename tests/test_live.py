import json
import threading
import time

import numpy as np

from cereb_platform.live import Lockstep, Pacer, drive
from cereb_platform.plots import plot_learning_compare
from conftest import FAKE_MODELS
from fake_sim import FakeExperiment


def run_fake(model='fake_fast.py', protocol='delay', isi_ms=None):
    exp = FakeExperiment(FAKE_MODELS / model, protocol=protocol, isi_ms=isi_ms)
    exp.build()
    events = []
    drive(exp, events.append)
    return exp, events


def test_drive_events_delay():
    exp, events = run_fake()
    kinds = [e['type'] for e in events]
    assert kinds.count('trial') == exp.n_trials == 150
    blocks = [e for e in events if e['type'] == 'block']
    assert [b['trials_done'] for b in blocks] == list(range(10, 151, 10))
    assert len(blocks[-1]['meas']) == 150
    # CS on/off for every trial, US only on paired trials
    stim = [e['what'] for e in events if e['type'] == 'stim']
    assert stim.count('cs_on') == 150 and stim.count('cs_off') == 150
    assert stim.count('us_on') == int(exp.trials.us_present.sum())
    # the fake learns: late blocks have more CRs than early ones
    m = blocks[-1]['meas']
    assert m[m.block == 14].cr.mean() > m[m.block == 0].cr.mean()
    # blinks are threshold crossings, at most a couple per trial
    blinks = [e for e in events if e['type'] == 'blink']
    assert 0 < len(blinks) <= 2 * exp.n_trials
    assert exp.n_done == exp.n_total


def test_blink_detector_rearms():
    exp, events = run_fake()
    trials_with_blink = {e['trial'] for e in events if e['type'] == 'blink'}
    # every paired trial has a UR, so it has at least one blink
    paired = set(np.flatnonzero(exp.trials.us_present.values))
    assert paired <= trials_with_blink


def test_partial_meas_matches_full():
    exp, events = run_fake()
    blocks = {e['trials_done']: e['meas'] for e in events if e['type'] == 'block'}
    full = blocks[150]
    part = blocks[50]
    assert part.cr.tolist() == full.cr.iloc[:50].tolist()


def test_trace_protocol_and_custom_isi():
    exp, _ = run_fake(protocol='trace')
    assert exp.n_trials == 300 and set(exp.trials.cs_duration) == {0.25}
    exp, _ = run_fake(isi_ms=1200)
    assert set(exp.trials.isi) == {1.2}


def test_finish_saves_outputs(tmp_path):
    exp, _ = run_fake('fake_slow.py')
    exp.finish(tmp_path)
    for f in ('trials.csv', 'summary.csv', 'learning_curve.csv', 'scalars.csv',
              'learning.png', 'aligned.png', 'params.json', 'traces.npz'):
        assert (tmp_path / f).exists(), f
    meta = json.loads((tmp_path / 'params.json').read_text())
    assert meta['overrides_vs_base'] == {'pfpc_learning_rate': {'base': 2e-5, 'used': 30}}
    assert meta['isi_ms'] == 500 and meta['protocol'] == 'delay'
    assert meta['input_fingerprint'] == exp.input_fingerprint()


def test_lockstep_keeps_models_on_the_same_step():
    cond, stop = threading.Condition(), threading.Event()
    progress = [-1, -1]
    exps = [FakeExperiment(FAKE_MODELS / f) for f in ('fake_fast.py', 'fake_slow.py')]
    log = []

    def worker(i):
        exp = exps[i]
        exp.build()
        step = exp.step

        def checked_step(n):
            # before a model simulates from step i0, its partner has reached i0 too
            log.append((i, exp.n_done, min(progress)))
            if i == 1:
                time.sleep(0.0001)        # model B is slower
            step(n)
        exp.step = checked_step
        lock = Lockstep(i, cond, progress, stop)
        drive(exp, lambda ev: None, label='AB'[i], lockstep=lock, block_size=1000)
        lock.leave()

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join(120) for t in ts]
    assert len(log) > 1000
    assert all(lowest >= i0 for _, i0, lowest in log if i0 > 0)
    assert exps[0].n_done == exps[1].n_done == exps[0].n_total


def test_same_seed_same_input_fingerprint():
    a = FakeExperiment(FAKE_MODELS / 'fake_fast.py', isi_ms=700)
    b = FakeExperiment(FAKE_MODELS / 'fake_slow.py', isi_ms=700)
    assert a.input_fingerprint() == b.input_fingerprint()
    assert a.input_fingerprint() != FakeExperiment(FAKE_MODELS / 'fake_fast.py', isi_ms=700,
                                                   seed=2).input_fingerprint()
    assert a.input_fingerprint() != FakeExperiment(FAKE_MODELS / 'fake_fast.py',
                                                   isi_ms=800).input_fingerprint()


def test_pacer_speed():
    class V:
        value = 10.0
    p = Pacer(V())
    t0 = time.perf_counter()
    for i in range(1, 11):
        p.wait(i * 0.1)            # 1 s of model time at 10x -> 0.1 s
    assert 0.08 < time.perf_counter() - t0 < 0.3
    V.value = 0.0                   # max: no waiting
    t0 = time.perf_counter()
    for i in range(11, 1000):
        p.wait(i * 0.1)
    assert time.perf_counter() - t0 < 0.05


def test_compare_plot_has_both_models():
    _, ev_a = run_fake('fake_fast.py')
    _, ev_b = run_fake('fake_slow.py')
    ma = [e for e in ev_a if e['type'] == 'block'][4]['meas']
    mb = [e for e in ev_b if e['type'] == 'block'][4]['meas']
    fig = plot_learning_compare({'A · fake_fast': ma, 'B · fake_slow': mb}, n_blocks=15)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ['A · fake_fast', 'B · fake_slow']
    assert fig.axes[0].get_xlim() == (-0.5, 14.5)


def test_pause_holds_and_resumes_from_the_same_step():
    exp = FakeExperiment(FAKE_MODELS / 'fake_fast.py')
    exp.build()
    paused, stop = threading.Event(), threading.Event()
    events = []
    t = threading.Thread(target=drive, args=(exp, events.append),
                         kwargs=dict(paused=paused, stop=stop))
    paused.set()                       # paused before the first step
    t.start()
    time.sleep(0.3)
    assert exp.n_done == 0
    paused.clear()
    time.sleep(0.05)
    paused.set()
    time.sleep(0.1)
    held_at = exp.n_done
    time.sleep(0.3)
    assert exp.n_done == held_at and 0 < held_at < exp.n_total
    paused.clear()
    t.join(60)
    assert exp.n_done == exp.n_total
    assert [e['index'] for e in events if e['type'] == 'trial'] == list(range(150))


def test_stop_while_paused():
    from cereb_platform.live import Stopped
    exp = FakeExperiment(FAKE_MODELS / 'fake_fast.py')
    exp.build()
    paused, stop = threading.Event(), threading.Event()
    paused.set()
    threading.Timer(0.2, stop.set).start()
    import pytest
    with pytest.raises(Stopped):
        drive(exp, lambda ev: None, paused=paused, stop=stop)


def test_trace_events_cover_the_whole_run():
    exp, events = run_fake()
    traces = [e for e in events if e['type'] == 'trace']
    assert traces and all(e['dt'] == 0.01 for e in traces)
    t = np.concatenate([e['t0'] + 0.01 * np.arange(len(e['y'])) for e in traces])
    y = np.concatenate([e['y'] for e in traces])
    assert np.allclose(np.diff(t), 0.01)                  # no gaps, no repeats
    assert abs(t[0] - 0.01) < 1e-9 and abs(t[-1] - exp.n_total * exp.dt) < 0.0101
    steps = np.round(t / exp.dt).astype(int)
    assert np.allclose(y, np.round(exp.eyelid[steps], 3))
    # ~100 ms per batch, so the page gets about 10 updates per second of model time
    assert len(traces) <= exp.n_total // 100 + exp.n_trials + 2


def test_us_has_on_and_off():
    exp, events = run_fake()
    stim = [e['what'] for e in events if e['type'] == 'stim']
    assert stim.count('us_on') == stim.count('us_off') == int(exp.trials.us_present.sum())


def test_headless_run_sends_no_traces(tmp_path):
    from cereb_platform.live import run_headless
    exp = FakeExperiment(FAKE_MODELS / 'fake_fast.py')
    lines = []
    run_headless(exp, tmp_path, log=lines.append)
    assert (tmp_path / 'params.json').exists() and 'done in' in lines[-1]
