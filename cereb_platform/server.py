"""The live panel's server: a browser page talking to Python.

Standard library only. The page gets events over server-sent events (GET /api/events)
and sends its controls with POST. Each model runs in its own worker process.
"""

import base64
import io
import json
import math
import multiprocessing as mp
import queue as queue_mod
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__, protocols
from .experiment import timestamp
from .live import worker_main
from .loader import discover_models, load_model
from .params import BLINK_THRESHOLD

WEB = Path(__file__).parent / 'web'
PLOT_LOCK = threading.Lock()        # matplotlib's pyplot isn't thread-safe
SAVE_DPI = 150
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.1, 10.0, 1.0
LABELS = ('A', 'B')


class Broadcaster:
    """Fans events out to every connected page; replays the latest state to new ones."""

    REPLAY = ('session', 'ready', 'plot', 'paused', 'resumed', 'finished', 'error')

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()
        self.history = []

    def subscribe(self):
        q = queue_mod.Queue()
        with self.lock:
            self.clients.add(q)
            for _, data in self.history:
                q.put(data)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.discard(q)

    def publish(self, ev):
        kind = ev.get('type')
        data = json.dumps(ev, default=str)
        with self.lock:
            if kind == 'session':
                self.history = []
            if kind in self.REPLAY:
                if kind == 'plot':       # only the latest plot is worth replaying
                    self.history = [h for h in self.history if h[0] != 'plot']
                self.history.append((kind, data))
            for q in self.clients:
                q.put(data)


def render_png(fig, dpi=80):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


class Session:
    """One live comparison: 1-2 models on one protocol."""

    def __init__(self, models, protocol, isi_ms, seed, speed, out_root, broadcaster,
                 factory='cereb_platform.experiment:Experiment'):
        self.models = models                      # [(label, display name, spec), ...]
        self.protocol, self.isi_ms, self.seed = protocol, isi_ms, seed
        self.bc = broadcaster
        self.factory = factory
        tag = f'{protocol}' + (f'_isi{isi_ms}' if isi_ms is not None else '')
        self.outdir = Path(out_root) / f'{timestamp()}_{tag}'
        ctx = mp.get_context('spawn')
        self.queue = ctx.Queue()
        self.stop_event = ctx.Event()
        self.pause_event = ctx.Event()            # set = paused
        self.speed = ctx.Value('d', speed)
        self.cond = ctx.Condition()
        self.progress = ctx.Array('i', [-2] * len(models))
        self.ctx = ctx
        self.procs = []
        self.state = {label: 'starting' for label, _, _ in models}
        self.blocks = {}                          # label -> {trials_done: meas}
        self.n_trials = None
        self.fingerprints = {}                    # label -> input fingerprint
        self.plotted = 0
        self.last_plot = None                     # (trials_done, {key: meas}) last drawn
        self.finished = False

    @property
    def keys(self):
        return {label: f'{label} · {name}' for label, name, _ in self.models}

    def start(self):
        self.bc.publish(dict(
            type='session', protocol=self.protocol, isi_ms=self.isi_ms, seed=self.seed,
            models=[dict(label=l, name=n) for l, n, _ in self.models],
            speed=self.speed.value, outdir=str(self.outdir)))
        for i, (label, name, spec) in enumerate(self.models):
            job = dict(model_spec=spec, protocol=self.protocol, isi_ms=self.isi_ms,
                       seed=self.seed, outdir=str(self.outdir / f'{label}_{name}'),
                       factory=self.factory)
            p = self.ctx.Process(target=worker_main, name=f'cereb-worker-{label}',
                                 args=(i, label, job, self.queue, self.speed,
                                       self.stop_event, self.cond, self.progress,
                                       self.pause_event),
                                 daemon=True)
            p.start()
            self.procs.append(p)
        threading.Thread(target=self._pump, name='cereb-pump', daemon=True).start()

    def set_speed(self, speed):
        self.speed.value = 0.0 if speed is None else float(speed)

    def stop(self):
        self.stop_event.set()

    def pause(self):
        if not self.finished and not self.pause_event.is_set():
            self.pause_event.set()
            self.bc.publish(dict(type='paused'))

    def resume(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.bc.publish(dict(type='resumed'))

    def running(self):
        return not self.finished

    # ------------------------------------------------------------------ events
    def _pump(self):
        while True:
            try:
                ev = self.queue.get(timeout=0.25)
            except queue_mod.Empty:
                if all(not p.is_alive() for p in self.procs):
                    for label, st in self.state.items():   # a worker died without a word
                        if st not in ('done', 'error', 'stopped'):
                            self.state[label] = 'error'
                            self.bc.publish(dict(type='error', model=label,
                                                 message='worker process exited unexpectedly'))
                    break
                continue
            self._handle(ev)
            if all(st in ('done', 'error', 'stopped') for st in self.state.values()):
                break
        self._finish()

    def _handle(self, ev):
        kind, label = ev.get('type'), ev.get('model')
        if kind == 'block':
            self.blocks.setdefault(label, {})[ev['trials_done']] = ev['meas']
            self._maybe_plot()
            return
        if kind == 'ready':
            self.state[label] = 'running'
            self.n_trials = ev['n_trials']
            self.fingerprints[label] = ev.get('input_fingerprint')
            if len(set(self.fingerprints.values())) > 1:
                self.bc.publish(ev)
                self.bc.publish(dict(type='error', model=None, message=(
                    'the models received different inputs '
                    f'({", ".join(f"{l}: {f}" for l, f in self.fingerprints.items())}); '
                    'stopping so they are not compared on different trials')))
                self.stop()
                return
        elif kind in ('done', 'error', 'stopped'):
            self.state[label] = kind
            if kind != 'done':
                self.blocks.pop(label, None)
            self._maybe_plot()
        self.bc.publish(ev)

    def _maybe_plot(self, final=False):
        """Plot every trial count that all models still running have reported.

        Workers' messages can arrive out of order, so blocks are kept per trial count.
        If the plot falls more than two updates behind, it skips to the latest."""
        live = [l for l, st in self.state.items() if st in ('running', 'done')]
        if not live or any(l not in self.blocks for l in live):
            return
        common = set.intersection(*(set(self.blocks[l]) for l in live))
        pending = sorted(k for k in common if k > self.plotted)
        if final and common:
            pending = pending or [max(common)]
        if len(pending) > 2:
            pending = pending[-1:]
        for k in pending:
            self._plot(k, live, save=final and k == max(common))
        for l in live:                  # drop what's no longer needed
            for k in [k for k in self.blocks[l] if k < self.plotted]:
                del self.blocks[l][k]

    def _figure(self, meas):
        from .plots import plot_learning_compare
        n_blocks = math.ceil(self.n_trials / 10) if self.n_trials else None
        isi = f', ISI {self.isi_ms} ms' if self.isi_ms is not None else ''
        title = f'{protocols.PROTOCOLS[self.protocol]["label"]} conditioning{isi}'
        return plot_learning_compare(meas, title=title, n_blocks=n_blocks, figsize=(9, 8))

    def _plot(self, k, live, save=False):
        self.plotted = max(self.plotted, k)
        meas = {self.keys[l]: self.blocks[l][k] for l in live}
        self.last_plot = (k, meas)
        with PLOT_LOCK:
            fig = self._figure(meas)
            if save:
                self.outdir.mkdir(parents=True, exist_ok=True)
                fig.savefig(self.outdir / 'learning_compare.png', dpi=SAVE_DPI)
            png = render_png(fig)
        self.bc.publish(dict(type='plot', trials_done=k, n_trials=self.n_trials, png=png))

    def saved_plot(self, dpi=SAVE_DPI):
        """The latest learning plot as PNG bytes at full resolution, and a file name."""
        if self.last_plot is None:
            return None, None
        k, meas = self.last_plot
        with PLOT_LOCK:
            fig = self._figure(meas)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi)
            import matplotlib.pyplot as plt
            plt.close(fig)
        names = '_vs_'.join(name for _, name, _ in self.models)
        isi = f'_isi{self.isi_ms}' if self.isi_ms is not None else ''
        stamp = self.outdir.name.split('_')[0]
        return buf.getvalue(), f'learning_{self.protocol}{isi}_{names}_trial{k}_{stamp}.png'

    def _finish(self):
        for p in self.procs:
            p.join(timeout=5)
        if any(st == 'done' for st in self.state.values()):
            try:
                self._maybe_plot(final=True)
            except Exception as e:      # the per-model results are already saved
                self.bc.publish(dict(type='error', model=None, message=f'final plot: {e}'))
            self.outdir.mkdir(parents=True, exist_ok=True)
            (self.outdir / 'session.json').write_text(json.dumps(dict(
                version=__version__, protocol=self.protocol, isi_ms=self.isi_ms,
                seed=self.seed, models=[dict(label=l, name=n, file=s) for l, n, s in self.models],
                input_fingerprints=self.fingerprints, outcome=self.state), indent=2))
        self.finished = True
        self.bc.publish(dict(type='finished', state=self.state,
                             outdir=str(self.outdir) if self.outdir.exists() else None))


class App:
    def __init__(self, models_dir='models', out_root='results',
                 factory='cereb_platform.experiment:Experiment'):
        self.models_dir = Path(models_dir)
        self.out_root = Path(out_root)
        self.factory = factory
        self.bc = Broadcaster()
        self.session = None
        self.lock = threading.Lock()
        self.refresh_models()

    def refresh_models(self):
        self.models, self.model_errors = discover_models(self.models_dir)

    def config(self):
        self.refresh_models()
        return dict(version=__version__, models=sorted(self.models),
                    model_errors=self.model_errors, blink_threshold=BLINK_THRESHOLD,
                    speed=dict(min=SPEED_MIN, max=SPEED_MAX, default=SPEED_DEFAULT),
                    running=bool(self.session and self.session.running()),
                    **protocols.config())

    def start(self, req):
        names = [req.get('model_a'), req.get('model_b')]
        if not names[0]:
            raise ValueError('choose model A')
        chosen = []
        for label, name in zip(LABELS, names):
            if not name:
                continue
            if name not in self.models:
                raise ValueError(f'unknown model {name!r}')
            load_model(self.models[name]).merge_params()     # whitelist check up front
            chosen.append((label, name, self.models[name]))
        protocol = req.get('protocol', 'delay')
        isi_ms = req.get('isi_ms') if protocols.PROTOCOLS.get(protocol, {}).get('isi_settable') else None
        protocols.resolve(protocol, isi_ms)                  # validates protocol and ISI
        speed = req.get('speed', SPEED_DEFAULT)
        speed = 0.0 if speed is None else min(max(float(speed), SPEED_MIN), SPEED_MAX)
        with self.lock:
            if self.session and self.session.running():
                self.session.stop()
                for p in self.session.procs:
                    p.join(timeout=10)
            self.session = Session(chosen, protocol, isi_ms and int(isi_ms),
                                   int(req.get('seed', 1)), speed, self.out_root, self.bc,
                                   factory=self.factory)
            self.session.start()
        return dict(ok=True, outdir=str(self.session.outdir))

    def set_speed(self, req):
        speed = req.get('speed')
        if speed is not None:
            speed = min(max(float(speed), SPEED_MIN), SPEED_MAX)
        if self.session:
            self.session.set_speed(speed)
        return dict(ok=True, speed=speed)

    def stop(self):
        if self.session:
            self.session.stop()
        return dict(ok=True)

    def pause(self):
        if not (self.session and self.session.running()):
            raise ValueError('nothing is running')
        self.session.pause()
        return dict(ok=True, paused=True)

    def resume(self):
        if self.session:
            self.session.resume()
        return dict(ok=True, paused=False)

    def shutdown(self):
        if self.session:
            self.session.stop()
            for p in self.session.procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = f'cereb-platform/{__version__}'

        def log_message(self, fmt, *args):      # keep the terminal quiet
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj, default=str).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                body = (WEB / 'index.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/api/config':
                self._json(app.config())
            elif self.path == '/api/events':
                self._events()
            elif self.path.split('?')[0] == '/api/plot.png':
                png, name = app.session.saved_plot() if app.session else (None, None)
                if png is None:
                    self._json(dict(error='no learning plot yet'), 404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(png)))
                self.send_header('Content-Disposition', f'attachment; filename="{name}"')
                self.end_headers()
                self.wfile.write(png)
            else:
                self._json(dict(error='not found'), 404)

        def do_POST(self):
            try:
                n = int(self.headers.get('Content-Length') or 0)
                req = json.loads(self.rfile.read(n) or b'{}')
                if self.path == '/api/run':
                    self._json(app.start(req))
                elif self.path == '/api/speed':
                    self._json(app.set_speed(req))
                elif self.path == '/api/stop':
                    self._json(app.stop())
                elif self.path == '/api/pause':
                    self._json(app.pause())
                elif self.path == '/api/resume':
                    self._json(app.resume())
                else:
                    self._json(dict(error='not found'), 404)
            except Exception as e:
                self._json(dict(error=f'{e}'), 400)

        def _events(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            q = app.bc.subscribe()
            try:
                self.wfile.write(b': connected\n\n')
                self.wfile.flush()
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(f'data: {data}\n\n'.encode())
                    except queue_mod.Empty:
                        self.wfile.write(b': ping\n\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                app.bc.unsubscribe(q)

    return Handler


def serve(models_dir='models', out_root='results', host='127.0.0.1', port=8000,
          open_browser=True, factory='cereb_platform.experiment:Experiment'):
    app = App(models_dir, out_root, factory=factory)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    url = f'http://{host}:{httpd.server_address[1]}/'
    print(f'cereb-platform {__version__}: live panel at {url}')
    print(f'models from {app.models_dir.resolve()} ({len(app.models)} found)')
    for f, err in app.model_errors.items():
        print(f'  skipped {f}: {err}')
    print('press Ctrl+C to stop')
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nstopping…')
    finally:
        app.shutdown()
        httpd.server_close()
    return app


def start_in_thread(**kw):
    """For tests: run the server in a background thread; returns (app, httpd, url)."""
    app = App(kw.get('models_dir', 'models'), kw.get('out_root', 'results'),
              factory=kw.get('factory', 'cereb_platform.experiment:Experiment'))
    httpd = ThreadingHTTPServer(('127.0.0.1', kw.get('port', 0)), make_handler(app))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.05)
    return app, httpd, f'http://127.0.0.1:{httpd.server_address[1]}/'
