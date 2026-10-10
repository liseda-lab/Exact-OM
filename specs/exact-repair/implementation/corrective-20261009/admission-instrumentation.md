# Engineering admission instrumentation

The previous 99-update TRAIN probe spent 605.147731 process seconds and
271.217156 seconds inside measured updates. Its 333.930575-second gap includes
recurring checkpoint work and cannot be assigned to one-time setup. The original
128-case denominator (99 measured cached, 27 unmeasured cached, two missing),
660-second allowance, terminal resource limitation and 650.280170 charged GPU
seconds remain unchanged. This change does not resume that probe.

`EXACT_REPAIR_PHASE_LOG` enables append-only invocation/span records around the
existing trainer. Named phases cover graph/grammar/pair setup, model creation,
source identity and optimizer setup, checkpoint snapshot construction,
serialization, file flush/fsync, replacement/directory fsync, read/deserialization
and full state restoration. Rows bind the timing/trainer source hashes and exact
checkpoint identity, epoch, minibatch offset and update count. Each records wall
and process CPU time, peak RSS, I/O blocks and serialized/read bytes. Spans are
inclusive and nested; they must not be summed as campaign costs. The existing
ledger remains authoritative. A start without an end is incomplete. Trace writes
are flushed; checkpoint durability retains its existing fsync sequence.

The instrumentation does not change architecture, labels, update opportunities,
optimizer, RNG, scientific endpoint or checkpoint state. Tests interrupt after a
committed minibatch and compare resumed tensors, optimizer, RNG and next position
with uninterrupted training. An injected serialization failure must preserve
the old checkpoint bytes and clean the unpublished temporary file. The trainer
source identity changes, so deployment uses a new committed snapshot; old source
checkpoints do not acquire new resume permission.

A separate CPU-only diagnostic reads the authenticated terminal engineering
state, serializes a new copy, reads it back and compares every tensor/primitive.
It never constructs a model or runs an optimizer. This measures current CPU I/O
and cannot retrospectively attribute the old GPU process gap or admit a fitting
endpoint.

The finite component probe uses two independently charged Slurm workers, each
with three CPUs and 20,000 MiB: native checks have no visible GPU and inference
exclusively owns the declared 2080 UUID. Authored TRAIN fixtures include corrupted
and coherent range/interaction cases. Inference uses the installed 13-node-type,
553-relation HGT declaration, three layers, width 128 and four heads with fresh
disposable weights. It performs no optimizer update, hosted request or TEST
input/evaluator read. Readiness nonces and row intervals expose actual concurrent
load; visibility mocks and arithmetic projections are not capacity evidence.
All scheduled rows, including unavailable tails, remain in the report.

The audit reserves 180 seconds in the already started learning stage; native and
inference each reserve 360 seconds there, with inference also charged to the
already started secondary-learning GPU stage. Preparation checks remaining
capacity at 70% without starting or renewing clocks; the dispatcher clips actual
admission again. Nested native calls use the remaining case/stage allowance with
cleanup reserved. The existing deterministic dispatcher/tmux owner launches the
audit, then the two component workers concurrently.

Component results do not qualify representative actual-case/combined-loss
throughput, generation, complete E3/E4/E5 execution or six selected models. All 54
scientific descriptors stay disabled. The teacher amendment incident, frozen
scientific rows, proposed DEV schedule and endpoint gates remain unchanged.
