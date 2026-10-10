# Primary evaluation ownership boundary

The original input-only execution adapter reserved the 2080 for all 1,535 rows,
including CPU-only native controls and common-pool construction. Its 114,240
seconds exceeded 70% of the 40-hour repair window (100,800 seconds).

The successor preserves the scientific schedule and separates each four-case
chunk into pool, native-control and inference ownership. Conference runs first.
Within each cohort all pool producers complete before two serial consumer lanes
begin; both lanes complete before the next cohort. This DAG needs at most six
CPUs, 40,000 MiB and one exclusively owned 2080. The inference lane retains every
E3 arm, both paired E1/E2 arms, and E4/E5 baselines and interventions on the same
UUID/resource profile. The 5090 is not counted as another inference worker.

| Reservation | Seconds |
| --- | ---: |
| Pool producers, including 30 seconds per shard | 5,520 |
| Native consumers, including shard overhead | 35,400 |
| GPU consumers, including shard overhead | 74,400 |
| Longest dependency path | 79,920 |
| Conservative dispatch allowance, 300 seconds × 54 descriptors | 16,200 |
| Elapsed projection including dispatch allowance | 96,120 |
| Worker service, counted once | 115,320 |

The last eight evaluation hours remain reserved. Conference keeps at least its
predecessor share of the two-worker envelope (and no less than one third).
Actual first admission, prior charges and the measurement deadline must still
clip these projections. The dispatcher normally checks prepared launches every
15 seconds; the projection nevertheless reserves an entire 300-second supervisor
interval per descriptor. This allowance is elapsed headroom, not additional
scientific work, GPU occupancy, or a larger case budget.

A consumer binds its producer manifest, job, dispatch nonce, attempt and output
location before launch. It requires successful terminal completion at that
nonce/step, a frozen producer command, the exact row identity, unchanged public
inputs/protocol, and matching terminal output digests for the row and every
consumed artifact. A producer timeout leaves consumers unavailable. It never
causes regeneration. Pool costs remain at the producer; consumers and comparison
rows reference them. Already committed rows and interrupted case reservations
retain the adapter's no-replay rules.

Native workers require zero Slurm-visible GPUs. Inference requires the admitted
2080 UUID, exact charged owner/nonce and both evaluation/GPU stage memberships.
There is no live ownership transfer. A future oversized route requires a separate
matched, budget-preserving successor; this preparation does not silently route
or retry a terminal scientific row on another GPU.

These are disabled contracts and a reservation calculation, not measurements or
TEST admission. A source-bound qualification of CPU generation/native work and
the exact concurrent inference profile remains mandatory; the old fit-plus-DEV
load measurements do not establish this new profile. Models, common fitting
endpoint, qualified teacher, weak labels and scale remain independently gated.
The old 99-update engineering result, recurring checkpoint overhead, previous
budgets, all TEST declarations/semantic slots and teacher incident are unchanged.
No TEST payload/evaluator, primary optimizer or hosted client is used by the
preparation or its fixture audit.
