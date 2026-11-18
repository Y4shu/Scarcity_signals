# Scarcity Signals and Stranded Compute

A simulation study of when a firm should buy AI accelerators ahead of demand and when it should wait for its data and power capacity to catch up. The model follows one enterprise over 48 months. It compares four procurement policies under uncertain demand, GPU supply shortages, and limited access to cloud compute.

The headline result from the baseline simulation: **gating purchases on readiness raises fleet utilization from 54.8% to 82.2% and reduces peak excess capacity from 284.2 to 115.4 GPU equivalents**, compared with the reactive hardware-first policy. These are results from the model's assumptions, not observed results for a real firm. See [results and limitations](docs/RESULTS.md).

## Run the project

Use Python 3.11 or newer, and run these commands from the directory containing this README:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# On Windows Git Bash, use: source .venv/Scripts/activate
python -m pip install -r requirements.txt
python sanity.py
python experiments.py
```

`sanity.py` checks eight limiting cases, including no shortages, ample readiness, cheap unlimited cloud, and zero lead times. `experiments.py` regenerates eight figures and the baseline table in `out/`. The archived outputs are included so you can read the results without installing Python. Rerunning the experiments replaces those output files in your local checkout.

## How the model works

Demand is a noisy S-curve. Prepared data limits how much of that demand can run. Power limits how many owned GPUs can operate. Cloud can serve some remaining ready workload, but its availability drops during shortages. The simulated supply market moves between normal and shortage regimes; shortages raise prices, lengthen lead times and leave part of an order backlogged.

| Policy | Ordering rule |
|---|---|
| Hardware-first | Follows the forecast, inflates shortage orders, and builds data and power reactively. |
| Buy-ahead | Builds data and power ahead, orders to the full forecast, and does not inflate shortage orders. |
| Gated | Builds readiness ahead and caps GPU orders at the workload and power expected to be ready when they arrive. |
| Cloud-first | Builds data readiness and serves workload using cloud rather than owning GPUs. |

Buy-ahead is the closest comparison for isolating the effect of the cap. Hardware-first is useful for showing the combined effect of stopping shortage gaming, building readiness earlier, and limiting orders. The cost and carbon results use scenario assumptions. The single firm does not feed its orders back into market shortage risk.

## Files

| Path | Contents |
|---|---|
| [`model.py`](model.py) | Assumptions, demand and shortage paths, policies, and simulation. |
| [`sanity.py`](sanity.py) | Checks on accounting and policy behavior. |
| [`experiments.py`](experiments.py) | Baseline, policy decomposition, sensitivity sweeps, and plotting. |
| [`out/`](out/) | Baseline CSV and all eight figures. |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Main readings, how to interpret the plots, and limitations. |
| [`papers/`](papers/) | Supplied course report and conference-format manuscript. Author and affiliation fields are placeholders in the original PDFs. |

The Git history can be reconstructed from the supplied snapshot. The milestone dates in that import are editorial dates used to organize the source, not a record of original Git activity.
