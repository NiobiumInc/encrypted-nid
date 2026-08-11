# Encrypted Network Intrusion Detector (NID) over the niobium-client FHETCH transport

This document describes how the FHE Network Intrusion Detection (NID/KitNET)
workload runs its server-side homomorphic evaluation over the niobium-client
FHETCH transport. This is the same client-over-HTTP path used by other Niobium FHE
workloads. The server's `replay()` is shipped over HTTP to a
`nbcc_fhetch_replay_server`, which `--exec`s a Niobium replay backend (the released
SDK) and returns the result ciphertext.

It covers the integration steps and changes. It does not cover Niobium's internals.

## What the integration gives you

Niobium records the server-side encrypted KitNET computation once as a portable
trace, then replays that trace. Replay runs either locally (for debugging) or on
the Niobium backend (accelerated). You keep the normal OpenFHE implementation of the
workload. Niobium wraps the part you want it to execute.

The trace captures the computation graph, not the data. A recorded trace can be
replayed with new keys and new inputs without re-recording, as long as the crypto
parameters (ring dimension, modulus chain) are unchanged.

## How the server replays

The compute binary `server_standalone_sdk` links the SDK's `libnbfhetch`. Its
`niobium::compiler().replay()` dispatches on `--target`. For `local` it runs the
in-process `fhetch_sim`. For `FOG` and other targets it becomes the FHETCH transport,
shipping the trace (with keys and inputs) over HTTP to a replay server or the Fog.
Recording always runs locally on CPU. Only replay uses a backend.

## How the SDK compute stage works

`Server/server_standalone_sdk.cpp` runs the same encrypted KitNET circuit
(`KitNET::execute_ckks`) and uses the explicit niobium-fhetch API. The mapping from a
plain compiler record/replay app to this SDK stage:

| plain compiler app | this SDK stage (`server_standalone_sdk`) |
|----------------------------------------|-------------------------------|
| `cached_key(relin, EvalMult)` | `tag_keys(cc)`. Captures the keys already loaded in the context. |
| `global_key_cache(...)` | dropped. The transport is stateless; keys are captured from the loaded context and packed into the request. |
| `niobium_hw(bool)` / `--niobium_hw` | dropped. The hardware data format is auto-enabled by the replay `--target` (e.g. `FOG`). |
| `if (!cache_valid) {record} else {replay}` | kept. Record XOR replay. |

`server_standalone_sdk` is record XOR replay. A cache-miss records only and writes the
OpenFHE (CPU) result to `Mirai_Workload_Inputs/score_ciphertext_fhe.bin`, so decrypt
CPU-verifies the record phase with no backend (record runs on the client). A cache-hit
replays over the transport and writes `score_ciphertext_replay.bin`. Tagging order is
`capture_crypto_context(cc)`, then `tag_input(input_i)` for each feature ciphertext,
then `tag_keys(cc)`, then `start` / `execute_ckks` / `probe` / `stop` on a cache-miss,
or `replay()` + `result()` on a cache-hit. The KitNET weight plaintexts are tagged
inline by `execute_ckks`. That code is shared, so the SDK target compiles `Kitnet.cpp`,
`dA.cpp`, and `Packetdata.cpp` with `OPENFHE_CPROBES` and the niobium-fhetch include
path first (its `niobium::compiler()` also resolves to the SDK).

`OPENFHE_CPROBES` is required on the circuit TUs so the ciphertext-ciphertext
`EvalMult` hooks fire. Without it the recorder sees an untracked context and emits 0
`mul` instructions.

## The pieces that changed

| File | Change |
|------|--------|
| `Server/server_standalone_sdk.cpp` | New. The one file with real transport integration: the explicit niobium-fhetch record/replay lifecycle around `KitNET::execute_ckks`, behind `#ifdef NIOBIUM_COMPILER`. |
| `CMakeLists.txt` (root) | New. `-DNIOBIUM_SDK_BUILD=ON` builds the four stages against the client's bundled OpenFHE and `libnbfhetch`. The compute stage is built with `-DNIOBIUM_COMPILER -DOPENFHE_CPROBES`. |
| `scripts/build_task.sh` | New. Self-contained build: the `niobium-client` submodule builds its own OpenFHE, `libnbfhetch`, and transport, then the NID stages build against that. |
| `harness/run_submission.py` | New. Server-agnostic driver: keygen, encrypt, compute (record if no trace, else replay), decrypt/verify. Takes `--target`, `--profile`, `-O`. |
| `niobium-client` | New git submodule. The Niobium client SDK, with its cooperative record/replay transport. |

The existing `Server/`, `src/`, `assets/`, and `profiles.json` are unchanged.

## Building (self-contained, no compiler checkout)

```bash
git lfs install && git lfs pull  # committed datasets/pcaps are Git-LFS tracked
git submodule update --init niobium-client
scripts/build_task.sh            # heavy the first time (compiles OpenFHE once)
```

This outputs `keygen`, `encrypt_mirai`, `decrypt_probe`, and `server_standalone_sdk`
in `build/`. It also builds the transport server (`nbcc_fhetch_replay_server`) and the
forwarder (`nbcc_fhetch_replay`) under `niobium-client/build/src/fhetch_transport/`,
and `libnbfhetch` under the client's build tree.

## Running it: record, then replay over the transport

`harness/run_submission.py` is server-agnostic. It records if there is no trace cache
and replays if one exists. Run it twice around a transport server. The repo does not
start the server; that is the caller's or CI's job.

```bash
# zero setup: record once, then replay in-process (target=local, the default).
python3 harness/run_submission.py --profile full

# replay on the real FPGA via the Fog (jobs-as-a-service):
python3 harness/run_submission.py --profile full --target FOG
```

- Backends (`--target`). `local` (the default) replays in-process, with no server and
  no credentials, so a fresh clone runs out of the box. `FHE_SIM` is OpenFHE software
  over a transport server you run (needs `NBCC_FHETCH_SERVER`). `FOG` runs the real
  FPGA via the Fog (jobs-as-a-service), with no local card needed. A directly-attached
  card is an advanced `--target` path for hosts that have one, and is left
  undocumented. The software sims (`local` and `FHE_SIM`) materialize the full
  polynomial state in host RAM and are heavy at ring 2^16, so `FOG` is the recommended
  path for NID. Switching `--target` does not change the workload code.
- Record once, replay with new keys. The first run records the trace. Later runs with
  regenerated keys and data replay it (Niobium refreshes the changed inputs
  automatically). Keep the recorded trace directory (`NID_Mirai_<profile>_workload/`)
  between runs. Delete it to force a fresh record after changing the computation or
  crypto parameters.
- Pass the optimization level as `--opt-level O3` (the harness default). `init()`
  forwards `--opt-level` to the replay backend so its optimizations apply. A bare `-O3`
  is silently ignored and leaves large traces unoptimized.
- ⚠️ Build the SDK stages against the pinned `niobium-client/vendor/niobium-fhetch`,
  the version this repo is validated against. An older niobium-fhetch may not forward
  `--opt-level` to the replay backend. When that happens, `--opt-level=O3` is silently
  dropped, the replay runs unoptimized, and the FPGA result collapses to a near-constant
  (it looks correct on the software backend and is wrong on the device).
  `scripts/build_task.sh` gets this right, and the harness asserts the replay dispatch
  line carries `--opt-level` and fails fast if it does not.
- `--target FOG` runs the real FPGA over the Fog. The backend is chosen entirely by
  `--target`. For `FOG`, instead of an `NBCC_FHETCH_SERVER` you run yourself, the
  harness submits each batch's replay via the client's Fog CLI
  (`niobium-client/scripts/fog submit`) to the Niobium Fog. Authenticate it first with
  `niobium-client/scripts/fog init`, then `login`. The transport is stateless and
  upload-bound. The Fog account enforces an in-flight-job quota, so keep the default
  `--jobs 1` (serial) unless you are on a fast, co-located uplink. See Choosing how to
  submit below.

## Profiles

All profiles share one CKKS config (ring 65536, depth 22, HEStd_128_classic), are all
50-feature, and score batches of 32,768 packets. Correctness is CPU-verified by
decrypting the replayed MSE. They differ in where the data comes from and how many
batches they run.

| Profile | Data source | Batches | Role |
|---------|-------------|---------|------|
| `full`  | a pre-extracted single batch in `assets/` (`Mirai_first_batch_32K.bin` + `Mirai_model_FULL.bin`) | 1 | single-batch accuracy baseline (lightest run) |
| `unit`  | pcap-derived from `Datasets/Unit/` | 2 | small multi-batch demo |
| `mirai` | pcap-derived from `Datasets/Mirai/` | 21 | full multi-batch detection run |

`profiles.json` fields differ per profile. Each profile carries only what its data path
needs.
- `full` carries explicit `model`, `norm`, and `data_file` because it points at the
  pre-extracted `assets/` batch (a distinct model file from the `Datasets/` one).
- `unit` and `mirai` carry `data_source` (resolving to `Datasets/<Source>/`),
  `warmup_skip` (packets skipped during batch extraction, matching the CPU path), and a
  tuned `cfar_threshold` used by the report's anomaly decisions.
- `full` has no `cfar_threshold`. Its single batch is benign (pre-attack), so there is
  no tuned detection threshold. The report falls back to the 99.9th percentile of
  normal traffic for `full`.

Multi-batch profiles score independent batches (same circuit, different data, no
cross-batch FHE reduce). The trace is recorded once and each batch is replayed as its
own request. `--batches N` caps the count (for example `--batches 8` runs 8 of mirai's
21). `--retries N` retries a batch whose replay drops mid-upload.

Data prep is automatic. `full` and `unit` ship committed `.bin` batch fixtures. Mirai's
21 batches (~273 MB) are not committed. The harness extracts any missing batch files on
demand from `Datasets/<src>/<src>_pcap.pcap` (via
`training/extract_dataset_batches.py`, using the profile's `warmup_skip`), so
`--profile mirai` works from a fresh clone with no manual step. If a committed fixture
is missing, the harness reports the missing file and stops before encryption.

Timing is not stated per profile. It depends on the target (FHE_SIM, FPGA, or Fog) and,
for the transport, the uplink bandwidth. The rough shape is one record (real CPU-FHE,
roughly constant across profiles) plus `batch_limit` replays, so `full` (1 batch) is
the quickest end-to-end.

## Run artifact and reports (charts)

The harness produces data. The report and plot tools consume it. Run and report stay
decoupled: the runner never depends on matplotlib, and the same artifact feeds any
consumer whether it was produced locally or on the Fog.

```
runs/<profile>_<target>_<timestamp>/
  scores_batch{N}.csv     # FPGA/replay (FHE) anomaly scores (Packet,MSE,RMSE), one per batch
  plaintext_batch{N}.csv  # cleartext-CPU ground truth (Packet,MSE,RMSE), paired 1:1
  batch{N}.log            # per-batch replay and decrypt log
  run.json                # manifest: profile, target, opt_level, cfar, per-batch
                          #   {batch, replay_ms, wall_s, passed}, totals
```

`run_submission.py` prints the artifact path and the exact `make report` line at the
end of a run. Generate charts from it as a separate step.

```bash
# FPGA anomaly-detection charts (from scores_batch*.csv) -> runs/<id>/graphs/
make report RUN=runs/<id> PROFILE=mirai

# add the CPU-vs-FPGA comparison chart -> runs/<id>/plots/
make report RUN=runs/<id> PROFILE=mirai CPU=<cpu_scores.csv>
```

The plaintext-vs-FPGA comparison (`scripts/plot_comparison.py`) runs by default. The
harness writes `plaintext_batch*.csv` (cleartext-CPU ground truth) next to
`scores_batch*.csv` (FPGA FHE), and the plot compares them. Pass `CPU=<cpu_scores.csv>`
(from the `networkmonitor.py <dataset>` reference path) to overlay a CPU-FHE series as
an optional third line.

## Choosing how to submit (the transport is stateless)

The niobium-client transport is stateless. Every replay request is self-contained (keys,
trace, inputs) and the worker keeps nothing afterward. This suits one-shot use. It gets
more wasteful the more requests you make, because the trace and keys are re-sent every
time. Rules of thumb:

- One-shot or occasional: just run it. Record once, replay once. The stateless re-upload
  is irrelevant at N=1.
- Independent multi-batch: record once, replay per batch (what the harness does, serially
  by default). On a fast, co-located uplink, add `--jobs N` to submit N batches
  concurrently across workers (each batch runs in its own isolated working dir under
  `.concurrent/`). On a slow uplink they serialize on your one pipe, so leave the default
  `--jobs 1`. `--jobs > 1` is opt-in and best validated on CI. Measured caveat: the Fog
  account enforces a concurrency quota (for example 4 in-flight jobs), and `fog submit`
  reserves a job for the whole replay. A failed or aborted replay leaves it "assigned"
  (orphaned) until it times out, so a naive `--jobs N` can exhaust the quota and cascade
  into HTTP 429s (`fog cancel --pending` clears orphans). Serial stays within quota.
- Map-reduce (cross-batch dependency): bundle all batches and the reduce into one recorded
  trace, or pull batch outputs back to the client and re-`tag_input()` them into a
  separate reduce request (client-mediated state).
- Heavy, repeated, or streaming: run from a host co-located with the Fog so the multi-GB
  uploads take seconds.

For one 50-feature `mirai`/`unit`/`full` request (~1.47 GB): ~1.2 GB is the batch's own
input ciphertexts (unavoidable, they change every batch), and ~266 MB (~18%) is the trace
and keys re-sent each time. The input bytes are inherent to FHE.
