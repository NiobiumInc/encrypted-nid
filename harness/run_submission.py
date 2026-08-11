#!/usr/bin/env python3
# Copyright 2025-present Niobium Microsystems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
run_submission.py — drive the NID (Network Monitor) workload, FBS-submission style.

Flow:  (build) -> keygen(once) -> encrypt(each batch) -> RECORD once (local) ->
       REPLAY each batch -> decrypt/verify each batch -> aggregate.

Batches: NID batches are INDEPENDENT (same circuit, different packet data, no
cross-batch FHE reduce), so the trace is recorded ONCE and each batch is replayed
as a SELF-CONTAINED request. This matches the niobium-client transport being
STATELESS — the worker holds nothing between requests, so every replay ships the
whole trace + keys + that batch's inputs. `profiles.json` `batch_limit` sets the
batch count (full = 1; unit = 2).

Replay backends (--target):
  local (default)   in-process simulator — no server, no credentials, runs out of the box
  FOG               real FPGA on the Fog: jobs-as-a-service (auto-provisions a worker),
                    or your own replay server if NBCC_FHETCH_SERVER is set
  FHE_SIM             software (OpenFHE) over a transport server you run (set NBCC_FHETCH_SERVER)
  (record always runs locally on CPU, no backend needed. The FPGA is reached via
   the Fog — use FOG, not a directly-attached card.)

    python3 harness/run_submission.py --profile unit                       # local, zero setup
    python3 harness/run_submission.py --profile unit --target FOG --opt-level O3
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # repo root
BUILD = ROOT / "build"                                    # scripts/build_task.sh output
CLIENT = ROOT / "niobium-client"
INPUTS = ROOT / "Mirai_Workload_Inputs"                   # shared keys/cc + result (per-batch, serial)
FOG = CLIENT / "scripts" / "fog"


def profiles() -> dict:
    return json.loads((ROOT / "profiles.json").read_text())["profiles"]


def model_path(profile: str, cfg: dict) -> str:
    if cfg.get("model"):
        return cfg["model"]
    src = cfg.get("data_source", "Mirai")                 # unit/mirai/... derive from Datasets/
    return f"Datasets/{src}/{src}_model.bin"


def batch_files(profile: str, cfg: dict, limit=None) -> list:
    """(input_data_file, encrypted_output_dir) per batch. `limit` caps the
    profile's batch_limit (e.g. run 8 of mirai's 21)."""
    n = cfg.get("batch_limit", 1)
    if limit is not None:
        n = min(n, limit)
    if n <= 1:
        df = cfg.get("data_file", "assets/datasets/Mirai_first_batch_32K.bin")
        return [(ROOT / df, INPUTS)]
    src = cfg.get("data_source", "Mirai")
    return [(ROOT / f"assets/datasets/{src}_batch_{b}.bin",
             ROOT / f"Mirai_Workload_Inputs_batch{b}") for b in range(n)]


def ensure_batches(profile: str, cfg: dict, batches: list) -> None:
    """Make the per-batch data files present before encryption.

    full/unit ship committed fixtures in assets/datasets/. Multi-batch profiles
    (e.g. mirai's 21 batches, ~273 MB) are NOT committed — the harness extracts
    them on demand from Datasets/<src>/<src>_pcap.pcap via
    training/extract_dataset_batches.py (same "harness owns dataset prep" model
    as the FBS submission). This makes every profile just work from a fresh
    clone instead of failing cryptically on a missing input file."""
    missing = [df for df, _ in batches if not df.exists()]
    if not missing:
        return
    n, src = len(batches), cfg.get("data_source", "Mirai")
    if n > 1:
        print(f"[harness] {len(missing)}/{n} batch file(s) missing — extracting "
              f"{n} '{src}' batch(es) (training/extract_dataset_batches.py) ...")
        sh(["python3", str(ROOT / "training" / "extract_dataset_batches.py"), src, str(n)])
        still = [df for df, _ in batches if not df.exists()]
        if still:
            raise SystemExit(f"[harness] extraction did not produce {still[0]} "
                             f"(check Datasets/{src}/{src}_pcap.pcap exists)")
    else:
        raise SystemExit(f"[harness] missing committed batch data {missing[0]} — "
                         f"the repo fixture is absent; re-checkout the dataset.")


def lib_env() -> dict:
    """Runtime env for the stage binaries: the client's bundled OpenFHE +
    libnbfhetch. Honors a caller-supplied DYLD/LD path (used by the reuse-a-
    prebuilt-OpenFHE flow). Points NBCC_FHETCH_REPLAY_BIN at the client forwarder."""
    env = os.environ.copy()
    libs = [CLIENT / "vendor" / "lib" / "openfhe" / "lib"]
    for d in (CLIENT / "build" / "vendor" / "niobium-fhetch",
              CLIENT / "build" / "_deps" / "niobium-fhetch-build"):
        if d.exists():
            libs.append(d)
    libp = os.pathsep.join(str(x) for x in libs)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[var] = libp + (os.pathsep + env[var] if env.get(var) else "")
    fwd = CLIENT / "build" / "src" / "fhetch_transport" / "nbcc_fhetch_replay"
    if fwd.exists() and not env.get("NBCC_FHETCH_REPLAY_BIN"):
        env["NBCC_FHETCH_REPLAY_BIN"] = str(fwd)
    if env.get("NBCC_FHETCH_SERVER") and not env.get("NBCC_FHETCH_REPLAY"):
        env["NBCC_FHETCH_REPLAY"] = str(fwd)
    return env


def cache_exists(profile: str) -> bool:
    # server_standalone_sdk: program dir "NID_Mirai_<p>_workload_profile_<p>".
    return any(ROOT.glob(f"NID_Mirai_{profile}_workload*"))


def sh(cmd, env=None, capture=False, cwd=None):
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd or ROOT),
                          check=not capture, capture_output=capture, text=True, env=env)


def parse_output(text: str) -> dict:
    out = {}
    for key, pat in (("compute_ms", r"\[TIMING\] compute_ms:\s*(\d+)"),
                     ("replay_ms", r"\[TIMING\] replay_ms:\s*(\d+)"),
                     ("fpga_ms", r"fpga=(\d+)ms")):
        m = re.search(pat, text)
        if m:
            out[key] = int(m.group(1))
    return out


def compute_cmd(profile, input_dir, args):
    c = [BUILD / "server_standalone_sdk", f"--profile={profile}",
         f"--input-dir={input_dir}", f"--target={args.target}",
         "--opt-level", args.optimization]
    return c


def run_record(profile, input_dir, args, env):
    """Record the trace once, locally (no backend). Records iff no cache yet."""
    if cache_exists(profile):
        return None
    print(f"[harness] recording trace (local) from {input_dir.name} ...")
    # strip any transport server so this is a pure local record
    rec_env = dict(env)
    rec_env.pop("NBCC_FHETCH_SERVER", None)
    cp = sh(compute_cmd(profile, input_dir, args), env=rec_env, capture=True)
    (ROOT / "measurements").mkdir(exist_ok=True)
    (ROOT / "measurements" / f"{profile}_record.log").write_text(cp.stdout + cp.stderr)
    if cp.returncode != 0:
        print(cp.stdout + cp.stderr)
        raise SystemExit(f"[harness] record failed for {profile}")
    return cp


def assert_o3_dispatch(stdout: str) -> None:
    """Guardrail: the replay dispatch line MUST carry --opt-level. If the SDK
    stages were built against a niobium-fhetch that does not forward --opt-level,
    the flag is silently dropped and the replay runs unoptimized, which makes the
    result collapse to a near-constant (looks bit-exact on the software backend,
    wrong on the device). Fail fast instead of shipping a collapsed result.

    Fix: build the SDK stages against the pinned niobium-client/vendor/niobium-fhetch
    (the version this repo is validated against)."""
    for ln in stdout.splitlines():
        if "via:" in ln and "nbcc_fhetch_replay" in ln and "--opt-level" not in ln:
            raise SystemExit(
                "[harness] FATAL: replay dispatched WITHOUT --opt-level -> it will run "
                "unoptimized and the result collapses to a near-constant. Rebuild the SDK "
                "stages against the pinned niobium-client/vendor/niobium-fhetch.\n  dispatch: "
                + ln.strip())


def run_replay(profile, b, input_dir, args, env, cwd=ROOT):
    """Replay one batch, retrying transient transport failures. When the Fog
    jobs path is active (args.fog; set for --target FOG without a server) each
    batch is a dedicated `fog submit` job; else run server_standalone_sdk
    directly (in-process for local, or over NBCC_FHETCH_SERVER). `cwd` isolates the
    program dir + result when running batches concurrently.

    The result ct is removed first so a FAILED replay can't leave decrypt reading
    the PREVIOUS batch's stale ciphertext. Retries on a missing 'replay done'
    (covers dropped connections / job-deadline cancellations)."""
    if args.fog:
        cmd = [FOG, "submit", BUILD / "server_standalone_sdk",
               f"--profile={profile}", f"--input-dir={input_dir}",
               f"--target={args.target}", "--opt-level", args.optimization]
    else:
        cmd = compute_cmd(profile, input_dir, args)
    result_ct = Path(cwd) / "Mirai_Workload_Inputs" / "score_ciphertext_replay.bin"
    t0 = time.time()
    cp = None
    for attempt in range(1, args.retries + 2):
        result_ct.unlink(missing_ok=True)                 # no stale read on failure
        cp = sh(cmd, env=env, capture=True, cwd=cwd)
        assert_o3_dispatch(cp.stdout)                      # O0 -> collapsed result; fail fast
        if cp.returncode == 0 and "replay done" in cp.stdout and result_ct.exists():
            break
        if attempt <= args.retries:
            print(f"[harness] batch {b}: replay attempt {attempt} failed "
                  f"(transport?), retrying ...")
    return cp, round(time.time() - t0, 2)


def run_decrypt(profile, cfg, data_file, env, cwd=ROOT, scores_path=None,
                plaintext_path=None):
    cmd = [BUILD / "decrypt_probe", "--profile", profile,
           "--model-path", (ROOT / model_path(profile, cfg)),
           "--features", cfg["features"], f"--data-path={data_file}",
           "--use-replay"]
    if scores_path is not None:                 # emit scores_batch*.csv (FPGA FHE)
        cmd.append(f"--save-scores={scores_path}")
    if plaintext_path is not None:              # emit plaintext_batch*.csv (cleartext CPU)
        cmd.append(f"--save-plaintext-scores={plaintext_path}")
    return sh(cmd, env=env, capture=True, cwd=cwd)


def make_batch_workdir(profile, b):
    """Per-batch isolated cwd for CONCURRENT replay (--jobs > 1). server/decrypt
    resolve cwd/Mirai_Workload_Inputs (keys + result) and cwd/<program_dir>
    (trace + serialized_probes), so a private cwd isolates the mutable state that
    concurrent batches would otherwise clobber. Big immutable files (.fhetch,
    .mk) are symlinked; the rest is copied so libnbfhetch's in-place input
    rewrite + probe unpack stay private. Returns the workdir Path."""
    import shutil
    prog = f"NID_Mirai_{profile}_workload_profile_{profile}"
    w = ROOT / ".concurrent" / f"batch_{b}"
    if w.exists():
        shutil.rmtree(w)
    w.mkdir(parents=True)
    for name in ("assets", "Datasets", "profiles.json"):          # read-only, shared
        (w / name).symlink_to(ROOT / name)
    # keys (shared, read-only) + a private result slot
    wi = w / "Mirai_Workload_Inputs"; wi.mkdir()
    for k in ("cryptocontext.bin", "relinearization_key.bin",
              "secret_key.bin", "public_key.bin"):
        if (INPUTS / k).exists():
            (wi / k).symlink_to(INPUTS / k)
    # private program dir: symlink the big immutable trace/key blobs, copy the rest
    src, dst = ROOT / prog, w / prog; dst.mkdir()
    big = {f"{prog}.fhetch", f"{prog}.mk.bin", f"{prog}.mk.ids"}
    for p in src.iterdir():
        if p.name == "serialized_probes":
            (dst / p.name).mkdir()                                # private, empty
        elif p.name in big:
            (dst / p.name).symlink_to(p)
        elif p.is_dir():
            shutil.copytree(p, dst / p.name)
        else:
            shutil.copy2(p, dst / p.name)
    return w


def process_batch(b, data_file, indir, args, cfg, env, cwd, run_dir):
    """Replay + decrypt one batch into the run artifact, return metrics.

    Writes runs/<id>/scores_batch{b}.csv (FPGA-FHE anomaly scores) +
    runs/<id>/plaintext_batch{b}.csv (cleartext-CPU ground truth, paired 1:1) +
    runs/<id>/batch{b}.log. The report compares plaintext vs FPGA from these."""
    cp, wall = run_replay(args.profile, b, indir, args, env, cwd=cwd)
    scores = run_dir / f"scores_batch{b}.csv"
    plaintext = run_dir / f"plaintext_batch{b}.csv"
    dp = run_decrypt(args.profile, cfg, data_file, env, cwd=cwd,
                     scores_path=scores, plaintext_path=plaintext)
    log = cp.stdout + cp.stderr + "\n" + dp.stdout + dp.stderr
    passed = (cp.returncode == 0 and dp.returncode == 0 and
              "VALIDATION PASSED" in dp.stdout)
    m = parse_output(log)
    m.update(batch=b, wall_s=wall, passed=passed,
             scores_csv=(scores.name if scores.exists() else None),
             plaintext_csv=(plaintext.name if plaintext.exists() else None))
    (run_dir / f"batch{b}.log").write_text(log)
    verdict = "PASS" if passed else "FAIL"
    tail = "" if passed else "\n" + "\n".join(log.strip().splitlines()[-20:])
    rms = m.get('replay_ms')
    rstr = f"{rms / 1000:.1f}s" if isinstance(rms, (int, float)) else "-"
    print(f"[harness] batch {b}: Encrypted computation completed "
          f"(elapsed: {rstr}) -> {verdict}{tail}")
    return m


def main() -> int:
    p = argparse.ArgumentParser(description="Run the NID workload (FBS-submission style).")
    p.add_argument("--profile", default="full",
                   help="Workload profile (full|unit|mirai). Default full.")
    p.add_argument("--target", default="local",
                   help="Replay target: local (default; in-process simulator, no "
                        "backend/credentials — runs out of the box), FHE_SIM (software, "
                        "over a transport server), FOG (real FPGA on the Fog, jobs-as-a-service "
                        "— the way to use the accelerator). NOTE: the software sims "
                        "(local/FHE_SIM) materialize the full polynomial state in RAM — "
                        "heavy at ring 2^16; FOG is the validated FPGA path for NID.")
    p.add_argument("-O", "--opt-level", dest="optimization",
                   choices=["O0", "O1", "O2", "O3"], default="O3",
                   help="Replay opt level (default O3 — REQUIRED: O0 skips address "
                        "compaction and the FPGA result is wrong).")
    # Deprecated: FOG now implies jobs-as-a-service when no NBCC_FHETCH_SERVER is
    # set. Kept hidden so older `--fog` commands still work (forces the jobs path).
    p.add_argument("--fog", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--batches", type=int, default=None,
                   help="Cap the number of batches (e.g. 8 of mirai's 21). Default: profile batch_limit.")
    p.add_argument("--retries", type=int, default=2,
                   help="Per-batch replay retries on transient transport failure (default 2).")
    p.add_argument("--jobs", type=int, default=1,
                   help="Concurrent batch replays (default 1 = serial, the tested path). "
                        ">1 isolates each batch in its own working dir and submits them in "
                        "parallel across Fog workers — worth it only on a fast/co-located uplink.")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip scripts/build_task.sh (assume build/ is present).")
    args = p.parse_args()

    if not args.skip_build and not (BUILD / "server_standalone_sdk").exists():
        print("[harness] building (scripts/build_task.sh) ...")
        subprocess.run(["bash", str(ROOT / "scripts" / "build_task.sh")], check=True)

    cfg = profiles().get(args.profile)
    if cfg is None:
        p.error(f"unknown profile {args.profile!r}")

    env = lib_env()
    env["NIOBIUM_PROFILE"] = args.profile
    INPUTS.mkdir(parents=True, exist_ok=True)
    batches = batch_files(args.profile, cfg, args.batches)
    # Backend is selected solely by --target. For FOG: if you point
    # NBCC_FHETCH_SERVER at your own replay server, the trace POSTs there;
    # otherwise the harness uses
    # Niobium's jobs-as-a-service (scripts/fog submit) to provision a worker.
    # (The hidden --fog alias forces the jobs path even when a server is set.)
    fog_jobs = (args.target == "FOG") and (args.fog or not env.get("NBCC_FHETCH_SERVER"))
    args.fog = fog_jobs      # downstream (run_replay, run.json) keys off this
    print(f"\n[harness] === {args.profile}: {len(batches)} batch(es), "
          f"target={args.target}, opt={args.optimization}, fog_jobs={fog_jobs} ===")
    # Dispatch wiring:
    #   target=local          -> the client's in-tree fhetch_sim worker (NBCC_FHETCH_SIM).
    #   FOG without a server   -> scripts/fog submit provisions a worker (run_replay).
    #   any other non-local    -> a running nbcc_fhetch_replay_server (NBCC_FHETCH_SERVER).
    if args.target == "local":
        sim = CLIENT / "build" / "vendor" / "niobium-fhetch" / "fhetch_sim"
        if not sim.exists():
            p.error(f"--target local needs the fhetch_sim worker at {sim} — "
                    f"build the client first (scripts/build_task.sh).")
        env["NBCC_FHETCH_SIM"] = str(sim)
    elif not fog_jobs and not env.get("NBCC_FHETCH_SERVER"):
        p.error(f"--target {args.target} needs a transport server: set NBCC_FHETCH_SERVER "
                f"to a running nbcc_fhetch_replay_server, or use --target FOG "
                f"(Fog jobs-as-a-service), or --target local (in-process, no server).")

    ensure_batches(args.profile, cfg, batches)   # extract mirai batches on demand if absent

    # 1) keys once (shared crypto params across batches)
    if not (INPUTS / "cryptocontext.bin").exists():
        sh([BUILD / "keygen", "--profile", args.profile, "--output-dir", INPUTS], env=env)

    # 2) encrypt each batch to its own dir
    for data_file, outdir in batches:
        outdir.mkdir(parents=True, exist_ok=True)
        sh([BUILD / "encrypt_mirai", cfg["features"], f"--input-file={data_file}",
            f"--keys-dir={INPUTS}", f"--output-dir={outdir}"], env=env)

    # 3) RECORD the trace once (local), from batch 0's data
    run_record(args.profile, batches[0][1], args, env)

    # 4) REPLAY + decrypt each batch into a canonical run artifact (runs/<id>/).
    #    The artifact — scores_batch*.csv + run.json — is the contract consumed by
    #    the report/plot tools; the runner never plots. See NIOBIUM_INTEGRATION.md.
    run_dir = ROOT / "runs" / f"{args.profile}_{args.target}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    t_replay = time.time()
    if args.jobs <= 1:
        # SERIAL (default): one batch at a time in the shared dir. The tested path.
        results = [process_batch(b, data_file, indir, args, cfg, env, ROOT, run_dir)
                   for b, (data_file, indir) in enumerate(batches)]
    else:
        # CONCURRENT (opt-in): N batches at once, each in an isolated cwd so their
        # program dirs / results can't collide. Pays off only when the uplink is
        # NOT the bottleneck (co-located / CI). See NIOBIUM_INTEGRATION.md.
        import concurrent.futures as _f
        print(f"[harness] concurrent replay, up to {args.jobs} at a time "
              f"(isolated per-batch working dirs under .concurrent/)")

        def _one(item):
            b, (data_file, indir) = item
            return process_batch(b, data_file, indir, args, cfg, env,
                                 make_batch_workdir(args.profile, b), run_dir)

        with _f.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_one, list(enumerate(batches))))

    replay_wall = round(time.time() - t_replay, 1)
    nok = sum(1 for r in results if r["passed"])
    manifest = {
        "profile": args.profile, "target": args.target, "opt_level": args.optimization,
        "fog": args.fog, "jobs": args.jobs, "num_batches": len(results),
        "passed": nok, "replay_wall_s": replay_wall,
        "cfar_threshold": cfg.get("cfar_threshold"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batches": sorted(results, key=lambda r: r["batch"]),
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    rel = run_dir.relative_to(ROOT)
    print(f"\n[harness] {args.profile}: {nok}/{len(results)} batch(es) PASS "
          f"| replay-phase wall={replay_wall}s (jobs={args.jobs})")
    print(f"[harness] artifact: {rel}/  (scores_batch*.csv + plaintext_batch*.csv + run.json)")
    print(f"[harness] report:   make report RUN={rel} PROFILE={args.profile}")
    return 0 if nok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())