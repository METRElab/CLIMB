"""Figures (ported from eyeblink/plots.py).

plot_learning keeps the original three panels. plot_learning_compare is the wrapper that
draws several models on the same three panels, one colour per model.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import trial_traces  # noqa: E402

MODEL_COLORS = ('#c8553d', '#2f6db5', '#3c9d5d', '#8a5ab0')   # A, B, ...


def plot_aligned(res, meas, n=10, pre=0.2):
    groups = {}
    paired = np.flatnonzero((meas.trial_type == 'paired').values)
    unmasked = np.flatnonzero(meas.trial_type.isin(['probe', 'cs_alone']).values)
    groups[f'first {n} paired'] = (paired[:n], 'tab:gray')
    groups[f'last {n} paired'] = (paired[-n:], 'k')
    if unmasked.size:
        groups[f'last {min(n, unmasked.size)} probe/CS-alone'] = (unmasked[-n:], 'tab:green')
    signals = [('pkj', 'Purkinje'), ('cn', 'CN'), ('io', 'IO error (PF→PC)'), ('eyelid', 'Eyelid (FEC)')]
    isis = np.unique(meas.isi)
    fig, axes = plt.subplots(1, len(signals), figsize=(4.2 * len(signals), 3.6), sharex=True)
    for ax, (sig, name) in zip(axes, signals):
        rel, tr = trial_traces(res, sig, pre=pre)
        for lab, (idx, col) in groups.items():
            if len(idx):
                ax.plot(rel * 1000, tr[idx].mean(0), color=col, lw=1.3, label=lab)
        for isi in isis:
            ax.axvline(isi * 1000, color='red', ls='--', lw=0.8)
        ax.set_title(name); ax.set_xlabel('Time from CS onset (ms)')
        if sig == 'io':
            ax.set_ylim(-1.1, 1.1); ax.set_title(name, fontsize=10)
    axes[0].legend(fontsize=7)
    plt.tight_layout(); return fig


def _draw_learning(axes, meas, color=None, label=None, show_isi=True):
    """Draw one model's learning data onto the three axes of a learning figure.

    With ``color`` set, every phase uses that colour (comparison plots); otherwise each
    phase gets the next colour in the cycle, as in the original plot."""
    phases = list(dict.fromkeys(meas['phase']))
    x0 = 0
    for ph in phases:
        m = meas[meas.phase == ph]
        pct = m.groupby('block')['cr'].mean() * 100
        x = x0 + pct.index.values
        name = ph if label is None else (label if len(phases) == 1 else f'{label} · {ph}')
        line, = axes[0].plot(x, pct.values, 'o-', label=name, color=color)
        c = line.get_color()
        paired = m[m.trial_type == 'paired']
        unm = m[m.trial_type != 'paired']
        if show_isi:   # grey dash = US time (ISI)
            axes[1].plot(x0 + m['trial_in_phase'] / 10, m['isi'] * 1000, '_', ms=5,
                         color='0.6', zorder=0)
        axes[1].plot(x0 + paired['trial_in_phase'] / 10, paired['onset'] * 1000,
                     '.', ms=4, color=c, label=name)
        axes[1].plot(x0 + unm['trial_in_phase'] / 10, unm['onset'] * 1000,
                     '^', ms=4, color=c)
        axes[1].plot(x0 + unm['trial_in_phase'] / 10, unm['peak_time'] * 1000,
                     'D', ms=4, mec=c, mfc='none')
        axes[2].plot(x0 + paired['trial_in_phase'] / 10, paired['amp_at_us'], '.', ms=4,
                     color=c, alpha=0.6)
        axes[2].plot(x0 + unm['trial_in_phase'] / 10, unm['peak_amp'], 'D', ms=4,
                     mec=c, mfc='none')
        x0 = (x[-1] + 1) if len(x) else x0


def _learning_axes(figsize=(10, 9)):
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    axes[0].set_ylabel('% CR (blocks of 10)'); axes[0].set_ylim(-5, 105)
    axes[1].set_ylabel('CR time from CS onset (ms)\n• onset (paired)   ▲ onset (probe)\n'
                       '◇ peak (probe)   – US time (ISI)')
    axes[2].set_ylabel('CR amplitude (FEC)\n• at US onset (paired)\n◇ peak (probe/CS-alone)')
    axes[-1].set_xlabel('Block of 10 trials')
    return fig, axes


def plot_learning(meas):
    fig, axes = _learning_axes()
    _draw_learning(axes, meas)
    axes[0].legend()
    fig.tight_layout(); return fig


def plot_learning_compare(meas_by_model, title=None, n_blocks=None, figsize=(10, 9)):
    """Several models' learning data on one figure: {label: meas}. ``n_blocks`` fixes the
    x-axis to the whole experiment, so the plot doesn't rescale as trials come in."""
    fig, axes = _learning_axes(figsize)
    for i, (label, meas) in enumerate(meas_by_model.items()):
        if meas is None or len(meas) == 0:
            continue
        _draw_learning(axes, meas, color=MODEL_COLORS[i % len(MODEL_COLORS)], label=label,
                       show_isi=(i == 0))
    if n_blocks:
        axes[-1].set_xlim(-0.5, n_blocks - 0.5)
    if any(m is not None and len(m) for m in meas_by_model.values()):
        axes[0].legend(loc='lower right')
    if title:
        axes[0].set_title(title)
    fig.tight_layout(); return fig


def plot_probe_traces(meas, rel, amp, n_groups=4, types=('probe', 'cs_alone')):
    phases = [ph for ph in dict.fromkeys(meas['phase'])
              if meas[(meas.phase == ph) & meas.trial_type.isin(types)].shape[0] > 0]
    if not phases:
        return None
    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 4), sharey=True,
                             squeeze=False)
    for ax, ph in zip(axes[0], phases):
        idx = np.nonzero(((meas.phase == ph) & meas.trial_type.isin(types)).values)[0]
        colors = plt.cm.viridis(np.linspace(0, 0.9, n_groups))
        for g, chunk in enumerate(np.array_split(idx, n_groups)):
            if len(chunk):
                ax.plot(rel * 1000, amp[chunk].mean(0), color=colors[g],
                        label=f'trials {chunk[0]}-{chunk[-1]}')
        for isi in np.unique(meas.loc[meas.phase == ph, 'isi']):
            ax.axvline(isi * 1000, color='red', ls='--', lw=0.8)
        ax.set_title(ph); ax.set_xlabel('Time from CS onset (ms)')
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel('FEC above baseline')
    plt.tight_layout(); return fig
