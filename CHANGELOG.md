# Changelog

## v0.2.0 (2026-11-22)

### Added

- **Pause / Resume.** A paused run is held in memory, so closing the server
  ends it.
- **Save simulation.** A button at the top right of the Learning box downloads the
  current learning plot as a PNG, drawn at full resolution (150 dpi) rather than the
  on-page preview. The file name gives the input, ISI, models and trial count, for
  example `learning_delay_isi500_lmu_vs_reservoir_single_trial40_20261125-131100.png`.

### Changed

- **Input timeline.** The CS and US lamps in the input panel are replaced by a timeline
  of the whole experiment, from 0 to the run's full length, that fills in as the run goes.
  It shows the CS, the US and each model's eyelid trace with its blinks, and numbers the
  trials. A highlighted window on the timeline (about the ISI plus 1.2 s wide, at least
  2 s) is drawn zoomed in underneath, with the blink threshold and times in ms from the CS
  onset. Drag the window, or click anywhere on the timeline, to look at any moment of the
  run so far. Set its width with the Window slider (0.5 to 30 s), or drag its edges when
  it's wide enough to grab. "Follow live" or a double-click sends it back to the present.
  Pausing freezes the timeline so you can look through it, and it stays available after
  the run ends.
- The page now gets each model's eyelid trace (one sample every 10 ms of model time),
  the end of each US and the run's total length, to draw the timeline. Trace samples are
  sent in batches at most every 50 ms of real time, so fast runs don't flood the page.

## v0.1.1 (2026-11-22)

### Changed

- The simulation speed slider's labels now read **Slowest**, **1×** and **Max**
  (previously 0.1×, 1×, 10× and max). The value shown beside the slider reads "Max" at
  the top of the range.
- Both models now get the same input at the same moment. They wait for each other every
  20 ms of model time instead of only at the end of each trial, so the same CS and US
  reach both models together and the two eyes can be compared blink by blink.
- One shared input panel above the eyes shows the CS and US lamps and the trial counter
  for both models. It replaces the separate CS/US lamps and trial counters on each eye.

### Added

- Input check. Each model fingerprints everything it receives (trial schedule, CS, US,
  climbing-fibre burst and mossy-fibre noise) when it's built. If the two models'
  fingerprints differ, the run stops with an error instead of comparing them on different
  input. The fingerprint is saved in each model's `params.json` and in `session.json`.
