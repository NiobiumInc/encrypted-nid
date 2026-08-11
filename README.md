# Encrypted Network Intrusion Detector

Run a real neural-network intrusion detector on traffic that stays encrypted the whole time.

This demo runs the KitNET anomaly detector, an ensemble of autoencoders, under Fully
Homomorphic Encryption with OpenFHE. The client encrypts network features and sends
them to the server. The server scores every packet for anomalies on ciphertext it
cannot read, then returns encrypted scores. Only the client can decrypt them.

The server holds the model and the evaluation keys. It never receives the secret key.
Decryption happens only on the client. The workload runs on a Niobium FPGA over the
Fog, so no local accelerator is needed.

The decrypted FPGA scores match the cleartext model to within floating-point precision.
A sample run is shown below.

## Run it

Prerequisites: Python 3.12 or newer, [git-lfs](https://git-lfs.com) (the datasets are
LFS-tracked), and about 6 GB of RAM for the local path. The FPGA-over-Fog path also
needs a Niobium Fog account.

```bash
# one-time setup
git lfs install && git lfs pull                     # datasets/pcaps (Git-LFS)
git submodule update --init niobium-client          # the SDK (FHETCH transport)
python3 -m venv .venv && source .venv/bin/activate  # Python 3.12+
pip install -r requirements.txt                     # numpy + matplotlib
scripts/build_task.sh                               # builds the SDK and the stages

# record once, then replay in-process. no server, no account.
python3 harness/run_submission.py --profile unit

# replay on a real FPGA over the Fog (needs a Niobium account;
# authenticate once with niobium-client/scripts/fog init, then login):
python3 harness/run_submission.py --profile mirai --target FOG --batches 4

# charts and detection metrics from a run
make report RUN=runs/<id> PROFILE=mirai
```

Three profiles share the same binaries and CKKS config. They differ in data and batch
count.

| Profile | Data | Batches |
|---------|------|---------|
| `full`  | one pre-extracted batch in `assets/` | 1 |
| `unit`  | two committed batches in `Datasets/Unit/` | 2 |
| `mirai` | the Mirai capture in `Datasets/Mirai/` | 21 |

`make report` targets the `mirai` profile. `full` and `unit` ship ready-to-run
fixtures. `mirai` extracts its 21 batches on first use. Each batch is an independent
32K-packet replay, so `--batches N` runs a subset.

To use your own data, drop a `.pcap` into `Datasets/` and re-run. The same
feature-extraction and encryption pipeline handles it.

## What happened

The server-side computation is recorded once, then replayed. The local path replays
in-process for a CPU reference. The Fog path replays on the FPGA. Each 32K-packet batch
is scored under encryption. The client decrypts the result and compares it to the
plaintext model.

```console
$ python3 harness/run_submission.py --profile mirai --target FOG

# run 1: record the trace once, locally (a few seconds, no server)
[compute-sdk] recording...
[TIMING] compute_ms: 6855
[compute-sdk] record done

# run 2: replay a 32K-packet batch on the FPGA over the Fog, then verify
[nbcc_fhetch_replay] POSTing 1474711778 bytes (streamed, target=FOG) -> Fog worker
[TIMING] replay_ms: 392475

=== METRICS FOR WORKFLOW ===
FHE RMSE (sqrt(MSE)):  1.291013183629697e-02
Plaintext RMSE:        1.291013184117146e-02
MSE relative error:    0.0000%
✅ VALIDATION PASSED (mean error: 0.00% ≤ 15.00%)
[harness] batch 0: Encrypted computation completed (elapsed: 392.5s) -> PASS
```

The [sample plot](docs/sample_plaintext_vs_encrypted.png) draws the FPGA (encrypted)
scores and the plaintext scores on the same axes. The Mirai attack rises above the
detection threshold in both panels, and the two panels overlap.

`make report` produces the charts and the detection metrics: precision, recall/TPR,
FPR, and F1 against the CFAR threshold.

- `scripts/plot_comparison.py`: plaintext-vs-FPGA score timelines, CFAR decisions, and
  the agreement scatter.
- `scripts/plot_fpga_results.py`: FPGA anomaly-score plots, labeled.
- `scripts/demo_report.py`: the full run report.

## More information

- [NIOBIUM_INTEGRATION.md](NIOBIUM_INTEGRATION.md): the build and run guide. Covers the
  record/replay model, the backends (`local`, `FHE_SIM`, `FOG`), and the transport.
- [docs/Encrypted_NID_Design.pdf](docs/Encrypted_NID_Design.pdf): the application and
  ML-model design document (CC BY 4.0).
- [Security and parameters](#security-and-parameters): the CKKS configuration.
- [Repository layout](#repository-layout): where the code lives.
- Kitsune: the detector this derives from. See [Acknowledgements](#acknowledgements).

### Security and parameters

All three profiles use the same CKKS configuration (OpenFHE). The parameters fit the
KitNET inference, including the Chebyshev polynomial approximations of the sigmoid and
tanh activations, within the multiplicative budget at a 128-bit security level.

- Ring dimension `N = 2^16 = 65536`. Each ciphertext packs `2^15 = 32768` slots, one
  per packet, so one 32K-packet batch fits in one ciphertext.
- Multiplicative depth 22. This yields a 23-limb RNS modulus chain (the captured
  context reports `moduli=23`). The KitNET Chebyshev activation approximations consume
  this depth.
- 54-bit scaling modulus with the FLEXIBLEAUTO scaling technique.
- UNIFORM_TERNARY secret-key distribution and OpenFHE's default hybrid key switching.
- Security level `HEStd_128_classic`. OpenFHE sizes the modulus chain so the total
  `logQ` stays within the [HomomorphicEncryption.org](https://homomorphicencryption.org)
  standard's bound for `N = 2^16`, which gives at least 128 bits of classical security.

The OpenFHE `CCParams`, from [src/keygen.cpp](src/keygen.cpp):

```cpp
CCParams<CryptoContextCKKSRNS> parameters;
parameters.SetSecretKeyDist(UNIFORM_TERNARY);
parameters.SetScalingModSize(54);
parameters.SetScalingTechnique(FLEXIBLEAUTO);
parameters.SetSecurityLevel(HEStd_128_classic);
parameters.SetMultiplicativeDepth(22);   // KitNET Chebyshev activation budget
parameters.SetRingDim(65536);            // N = 2^16
parameters.SetBatchSize(32768);          // 2^15 slots (one per packet)
```

### Repository layout

```
encrypted-nid/
├─ README.md                # this file
├─ NIOBIUM_INTEGRATION.md   # build and run guide (record/replay, backends, Fog)
├─ CONTRIBUTING.md
├─ LICENSE                  # Apache-2.0
├─ NOTICE                   # third-party attributions (Kitsune, etc.)
├─ profiles.json            # workload profiles: full / unit / mirai
├─ requirements.txt         # Python deps (numpy, matplotlib)
├─ Makefile  CMakeLists.txt # convenience targets, C++ stage build
│
├─ networkmonitor.py        # Python driver: in-the-clear model training
├─ harness/                 # run_submission.py: record, replay, verify
├─ scripts/                 # build_task.sh, report and plot tools
│
├─ Server/                  # C++ FHE server: KitNET inference under encryption
├─ src/                     # C++ stages: keygen, encrypt_mirai, decrypt_probe
├─ KitNET/                  # KitNET anomaly detector (Kitsune-derived)
├─ Features/                # PCAP to feature-vector extraction (Kitsune-derived)
├─ training/                # offline training, on-demand batch preparation
│
├─ Datasets/                # trained models, normalization, labels (Mirai, Unit)
├─ assets/                  # committed batch fixtures, the FULL model
├─ docs/                    # design document (PDF)
│
└─ niobium-client/          # Niobium SDK (FHETCH transport), git submodule
```

## Acknowledgements

This project builds on Kitsune. The KitNET anomaly detector, the feature extraction,
and the Mirai capture used here come from the Kitsune project by Yisroel Mirsky, Tomer
Doitshman, Yuval Elovici, and Asaf Shabtai, "Kitsune: An Ensemble of Autoencoders for
Online Network Intrusion Detection", NDSS 2018. The reference implementation is at
https://github.com/ymirsky/Kitsune-py.
