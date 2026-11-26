# CLIMB: Cerebellar Learning and Interchangeable Models' Benchmark

Test granule-layer models inside a fixed reference cerebellar circuit on eyeblink conditioning, and watch two models learn side by side.

Models are built in [Nengo](https://www.nengo.ai/).


## Install

Python 3.13.

```
pip install -e .
```

## Live panel

```
python -m cereb_platform serve
```

This opens the panel in your browser. Pick model A and, optionally, model B from the `models/` folder, pick delay or trace conditioning, set the ISI for delay (200–2500 ms; 350–1600 ms is the ideal cerebellar range) and press **Run**.

- Both models get exactly the same input. The input panel shows it as a timeline of the whole experiment: the CS (tone), the US (air puff) and each model's eyelid with its blinks. 
- Drag the highlighted window, or click anywhere on the timeline, to look at that  moment zoomed in underneath; the Window slider sets how wide it is.
- Each model gets an eye. The eye blinks whenever the model's eyelid
  output crosses 0.1.
- The speed slider runs from Slowest (0.1× real time) through 1× to Max. It slows a run down so you can watch it; it can't make a run faster than your computer can simulate it.
- Both models get the same seed, so the same trial schedule, CS, US and mossy-fibre noise.

Results are saved in `results/<time>_<protocol>/`

Options: `--models DIR`, `--out DIR`, `--port N`, `--no-browser`.

## Custom granule-layer model

A model is a `.py` file in `models/` with one subclass of `GranuleLayer`:

```python
import nengo
from cereb_platform import GranuleLayer

class MyLayer(GranuleLayer):
    name = 'my_layer'
    defaults = dict(my_tau=0.05, n_granule=1000)

    def build(self, mf):
        p = self.p
        gc = nengo.Ensemble(p['n_granule'], dimensions=1)
        nengo.Connection(mf, gc, synapse=p['my_tau'])
        return gc
```

Conditions of layer:
- **Input.** `mf` is a 1-D `nengo.Ensemble`. Its decoded value is the CS (1 while the tone is on), delayed by 100 ms, plus low-pass filtered noise.
- **Output.** Return a `nengo.Ensemble`. Its neurons are the parallel fibres: the circuit connects `output.neurons` to the Purkinje cells through the learned PF→PC connection.
- **Parameters.** `defaults` may set the model's own keys, plus three reference keys:
  `n_granule`, `granule_noise_std` and `pfpc_learning_rate` (default `2e-5`).
- **Seeds.** `self.p['seed']` is the run's seed. Default=1

Check a model file before using it:

```
python -m cereb_platform check models/my_layer.py
```

## Panel-less run

To run faster without the panel:

```
python -m cereb_platform run delay models/lmu.py --isi 500
python -m cereb_platform run trace models/reservoir_single.py
```

## Included models

- `lmu`: Legendre delay network (LMU) granule layer inspired by Stockel et al. (2021)
- `reservoir_single`: single recurrent inhibitory reservoir inspired by Rossert et al. (2015)


## Changes

See [CHANGELOG.md](CHANGELOG.md).

## AI Usage disclosure
Graph displays and front-ends were built using Claude. No part of the models or tasks were built with generative AI

## Licence

MIT.
