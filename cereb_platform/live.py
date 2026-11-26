"""Stepping an Experiment through its trials, with events, pacing and lockstep.

``drive`` is the loop. It is used by the live workers (one process per model) and by the
headless ``run`` command (no pacing, no partner).
"""

import importlib
import time
import traceback

from .params import BLINK_THRESHOLD
from .stimulus import falling_edges, rising_edges

BLOCK_SIZE = 10           # learning plot update interval (trials)
CHUNK_STEPS = 20          # 20 ms of simulated time per chunk
TRACE_EVERY = 10          # eyelid sample sent to the page every 10 ms of model time
TRACE_BATCH = 100         # ... in batches covering at least 100 ms of model time
TRACE_MIN_WALL = 0.05     # and at most one batch per 50 ms of real time, so fast runs
TRACE_MAX_BATCH = 5000    # send fewer, bigger batches (but never more than 5 s each)


class Stopped(Exception):
    pass


def hold_while_paused(paused, stop):
    """Block while ``paused`` is set. Returns True if it waited. Raises Stopped if the
    run is stopped while paused."""
    if paused is None or not paused.is_set():
        return False
    while paused.is_set():
        if stop is not None and stop.is_set():
            raise Stopped()
        time.sleep(0.05)
    return True


class Pacer:
    """Keeps simulated time at ``speed`` x wall-clock time. speed <= 0 means as fast as
    possible. ``speed`` is anything with a ``.value`` (a multiprocessing.Value) or None."""

    def __init__(self, speed=None, stop=None):
        self.speed, self.stop = speed, stop
        self.reanchor(0.0)

    def current(self):
        return 0.0 if self.speed is None else float(self.speed.value)

    def reanchor(self, sim_t):
        self.wall0, self.sim0, self.s = time.perf_counter(), sim_t, self.current()

    def wait(self, sim_t):
        s = self.current()
        if s != self.s:
            self.reanchor(sim_t)
            return
        if s <= 0:
            return
        delay = self.wall0 + (sim_t - self.sim0) / s - time.perf_counter()
        if delay < -0.5:               # can't keep up: don't try to catch up in a burst
            self.reanchor(sim_t)
            return
        while delay > 0:
            if self.stop is not None and self.stop.is_set():
                return
            time.sleep(min(delay, 0.05))
            delay = self.wall0 + (sim_t - self.sim0) / s - time.perf_counter()


class Lockstep:
    """Keeps the models on the same input at the same moment.

    After every chunk (20 ms of model time) each model waits until its partner has
    reached the same time step, so the same CS and US reach both models together.
    progress[i] holds the last time step model i finished; -2 means not built yet and
    -1 built. A finished or failed model sets it to a large number so its partner never
    waits for it."""

    DONE = 10 ** 9

    def __init__(self, index, cond, progress, stop):
        self.index, self.cond, self.progress, self.stop = index, cond, progress, stop

    def sync(self, n):
        """Record that this model has reached step ``n`` and wait for the others."""
        with self.cond:
            self.progress[self.index] = n
            self.cond.notify_all()
            while min(self.progress[:]) < n:
                if self.stop.is_set():
                    raise Stopped()
                self.cond.wait(0.25)

    def leave(self):
        with self.cond:
            self.progress[self.index] = self.DONE
            self.cond.notify_all()


def drive(exp, emit, label='A', speed=None, stop=None, lockstep=None, paused=None,
          blink_threshold=BLINK_THRESHOLD, chunk_steps=CHUNK_STEPS, block_size=BLOCK_SIZE,
          trace_every=TRACE_EVERY, trace_batch=TRACE_BATCH):
    """Run ``exp`` to the end. Emits dict events through ``emit``:
    trial (a trial starts), stim (cs_on / cs_off / us_on / us_off), blink, trace (the
    eyelid every ``trace_every`` steps, sent every ``trace_batch`` steps), block (every
    ``block_size`` trials, with the scored trials so far). Raises Stopped if ``stop`` is
    set. While ``paused`` is set, the simulation holds between chunks and then carries on
    from the same time step."""
    dt = exp.dt
    bounds = list(exp.start_steps) + [exp.n_total]
    stim_events = sorted(
        [(int(i), 'cs_on') for i in rising_edges(exp.stim['cs'])]
        + [(int(i), 'cs_off') for i in falling_edges(exp.stim['cs'])]
        + [(int(i), 'us_on') for i in rising_edges(exp.stim['us'])]
        + [(int(i), 'us_off') for i in falling_edges(exp.stim['us'])])
    se = 0                                         # next stim event to report
    armed = True                                   # blink detector
    traced = 0                                     # last step sent as a trace sample
    traced_wall = 0.0
    pacer = Pacer(speed, stop)
    trial_rows = exp.trials.reset_index()

    for k in range(-1, exp.n_trials):              # k = -1: lead-in before trial 0
        end = bounds[k + 1]
        if k >= 0:
            row = trial_rows.iloc[k]
            emit(dict(type='trial', model=label, index=k, n_trials=exp.n_trials,
                      trial_type=row.trial_type, phase=row.phase, t=exp.n_done * dt))
        while exp.n_done < end:
            if stop is not None and stop.is_set():
                raise Stopped()
            if hold_while_paused(paused, stop):
                pacer.reanchor(exp.n_done * dt)    # model time stood still while paused
            i0 = exp.n_done
            exp.step(min(chunk_steps, end - i0))
            i1 = exp.n_done
            while se < len(stim_events) and stim_events[se][0] <= i1:
                i, what = stim_events[se]
                emit(dict(type='stim', model=label, what=what, t=i * dt, trial=k))
                se += 1
            for i, x in zip(range(i0 + 1, i1 + 1), exp.eyelid[i0 + 1:i1 + 1]):
                if armed and x >= blink_threshold:
                    emit(dict(type='blink', model=label, t=i * dt, trial=k))
                    armed = False
                elif not armed and x < blink_threshold:
                    armed = True
            pending = i1 - traced
            if trace_every and (i1 == exp.n_total or pending >= TRACE_MAX_BATCH or (
                    pending >= trace_batch
                    and time.perf_counter() - traced_wall >= TRACE_MIN_WALL)):
                s0 = (traced // trace_every + 1) * trace_every
                if s0 <= i1:
                    ys = exp.eyelid[s0:i1 + 1:trace_every]
                    emit(dict(type='trace', model=label, t0=round(s0 * dt, 6),
                              dt=trace_every * dt, y=[round(float(v), 3) for v in ys]))
                traced, traced_wall = i1, time.perf_counter()
            if lockstep is not None:
                lockstep.sync(i1)
            pacer.wait(i1 * dt)
        if k >= 0:
            n_done = k + 1
            if n_done % block_size == 0 or n_done == exp.n_trials:
                emit(dict(type='block', model=label, trials_done=n_done,
                          meas=exp.partial_meas(n_done)))
                pacer.reanchor(exp.n_done * dt)     # scoring time isn't model time


def resolve_factory(path):
    """'package.module:Class' -> the class (lets tests swap in a fake simulator)."""
    mod, _, attr = path.partition(':')
    return getattr(importlib.import_module(mod), attr)


def worker_main(index, label, job, queue, speed, stop, cond, progress, paused=None):
    """Entry point of a live worker process (one per model)."""
    import matplotlib
    matplotlib.use('Agg')
    lock = Lockstep(index, cond, progress, stop)
    emit = queue.put
    exp = None
    try:
        factory = resolve_factory(job.get('factory', 'cereb_platform.experiment:Experiment'))
        exp = factory(job['model_spec'], protocol=job['protocol'], isi_ms=job.get('isi_ms'),
                      seed=job.get('seed', 1))
        emit(dict(type='status', model=label, message=f'building {exp.name}…'))
        t0 = time.perf_counter()
        exp.build()
        emit(dict(type='ready', model=label, name=exp.name, n_trials=exp.n_trials,
                  duration=round(exp.n_total * exp.dt, 3),
                  build_s=round(time.perf_counter() - t0, 2),
                  input_fingerprint=exp.input_fingerprint(),
                  overrides=exp.layer_cls.overrides_vs_base(exp.p)))
        # wait until every model is built, so the run starts together
        with cond:
            progress[index] = -1
            cond.notify_all()
            while min(progress[:]) < -1:
                if stop.is_set():
                    raise Stopped()
                cond.wait(0.25)
        drive(exp, emit, label=label, speed=speed, stop=stop, lockstep=lock, paused=paused)
        lock.leave()
        emit(dict(type='status', model=label, message='saving results…'))
        exp.finish(job['outdir'])
        emit(dict(type='done', model=label, outdir=str(job['outdir'])))
    except Stopped:
        lock.leave()
        emit(dict(type='stopped', model=label))
    except Exception as e:
        lock.leave()
        emit(dict(type='error', model=label, message=f'{type(e).__name__}: {e}',
                  traceback=traceback.format_exc()))
    finally:
        if exp is not None:
            try:
                exp.close()
            except Exception:
                pass


def run_headless(exp, outdir, log=print):
    """Run one experiment as fast as possible and save its outputs (the `run` command)."""
    t0 = time.perf_counter()
    exp.build()
    log(f'built {exp.name} in {time.perf_counter() - t0:.1f} s; '
        f'{exp.n_trials} trials, {exp.duration:.0f} s simulated')

    def emit(ev):
        if ev['type'] == 'block':
            m = ev['meas']
            last = m[m.block == m.block.max()]
            log(f"  trials {ev['trials_done']:4d}/{exp.n_trials}: "
                f"{100 * last.cr.mean():5.1f}% CR in the last block")
    drive(exp, emit, trace_every=None)          # no page to draw traces on
    meas = exp.finish(outdir)
    log(f'done in {time.perf_counter() - t0:.0f} s; results in {outdir}')
    return meas
