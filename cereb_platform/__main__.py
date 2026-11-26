import argparse
import ast
import sys
from pathlib import Path


def parse_set(items):
    out = {}
    for item in items:
        key, _, val = item.partition('=')
        try:
            out[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            out[key] = val
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m cereb_platform',
                                 description='Cerebellar testing platform: eyeblink conditioning.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('serve', help='open the live panel in your browser')
    s.add_argument('--models', default='models', help='folder of granule-layer model files')
    s.add_argument('--out', default='results', help='where results are saved')
    s.add_argument('--port', type=int, default=8000)
    s.add_argument('--no-browser', action='store_true', help="don't open a browser tab")

    r = sub.add_parser('run', help='run one model without the panel, as fast as possible')
    r.add_argument('protocol', choices=['delay', 'trace'])
    r.add_argument('model', help='model file (path.py or path.py:ClassName)')
    r.add_argument('--isi', type=float, default=None, help='ISI in ms (delay only, 200-2500)')
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help='n_granule, granule_noise_std, pfpc_learning_rate or a model key')
    r.add_argument('--out', default='results')

    c = sub.add_parser('check', help='quick smoke test of a model file')
    c.add_argument('model')
    c.add_argument('--seed', type=int, default=1)

    args = ap.parse_args(argv)

    if args.cmd == 'serve':
        from .server import serve
        if not Path(args.models).is_dir():
            ap.error(f'models folder not found: {args.models}')
        serve(models_dir=args.models, out_root=args.out, port=args.port,
              open_browser=not args.no_browser)
    elif args.cmd == 'run':
        from .experiment import Experiment, timestamp
        from .live import run_headless
        exp = Experiment(args.model, protocol=args.protocol, isi_ms=args.isi, seed=args.seed,
                         overrides=parse_set(args.overrides))
        tag = args.protocol + (f'_isi{exp.isi_ms}' if args.protocol == 'delay' else '')
        run_headless(exp, Path(args.out) / f'{timestamp()}_{tag}_{exp.name}')
    elif args.cmd == 'check':
        from .check import run_check
        sys.exit(0 if run_check(args.model, seed=args.seed) else 1)


if __name__ == '__main__':
    main()
