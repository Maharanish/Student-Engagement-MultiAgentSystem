# Student Engagement — Agentic AI

> A local-first, multi-agent system that observes a learner during an online study
> session, estimates engagement from webcam frames with an on-device TimeSformer
> model, maintains a Bayesian belief over hidden engagement states, and — through a
> utility-based orchestrator — offers gentle, optional, tiered prompts when attention
> appears to drift. Everything runs on the learner's own machine: no audio is
> captured, raw frames are never written to disk, and no data leaves the device.

> ⚠️ **Before any first-time run, read and complete
> [`docs/consent_form.md`](docs/consent_form.md).** Running the system captures
> webcam frames; informed consent is required. See also
> [`docs/ethics_statement.md`](docs/ethics_statement.md).

---

## 1. Project overview

**Research motivation.** Online study sessions invite quiet disengagement: students
drift, lose the thread, and recover only when an external cue arrives — often hours
too late. Existing engagement-monitoring tools tend to be either intrusive (focus
lock, screen monitoring), evaluative (attendance / grading), or both. This thesis
prototype asks: **can a fully local, ethically-bounded agentic AI offer timely,
optional, supportive prompts that respect learner autonomy while still helping
sustain attention?**

**Main contribution.** A utility-based agentic orchestrator that pairs a
DAiSEE-trained TimeSformer engagement classifier with a Bayesian belief over four
hidden states (`actively_dis`, `drifting`, `engaged`, `frustrated`) and decides
tiered interventions by expected-utility maximization with explicit fatigue,
cooldown, warmup, and budget governors. The ethical policy — *"do not pester a
frustrated learner"* — is encoded mathematically in a 4 × 4 utility matrix rather
than buried in conditional logic, making it auditable, perturbation-testable, and
reproducible.

## 2. Features

- **FER-based engagement detection.** TimeSformer model fine-tuned on DAiSEE; 4-way
  ordinal softmax (very low → very high engagement) at ~ 0.5 Hz.
- **Agentic intervention logic.** Four cooperating agents (Detection, Orchestrator,
  Intervention, Delivery) around a thread-safe blackboard.
- **Utility-aware intervention policy.** Bayesian belief + `EU(a) = Σ b·U − fatigue
  − cooldown`; sensitivity-tested against ±20 % perturbation of the utility matrix.
- **Ethical safeguards.** No frames on disk, no audio, no network, no grading, no
  attendance, no scoring; persistent Tier-3 widget with non-coercive options; bounded
  interruption rate (≤ 4 prompts / session, ≥ 180 s apart, 10-min warmup).
- **Local-only processing.** No cloud, no telemetry, no remote dependency. The
  application has no network client.
- **Dry-run mode.** `--dry-run` swaps the camera for a Dirichlet softmax generator,
  letting the full pipeline run in CI without a webcam.
- **Per-user adaptive profile.** Across sessions, the system aggregates the user's
  engagement-label distribution (Dirichlet update) and per-tier response rate (Beta
  posterior mean) into a portable `profiles/<user-id>.json`.

## 3. Quickstart

### 3.1 Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Place the model weights (optional)

Drop the trained checkpoint at `mas_engagement/data/E3_WCE_lr1e5.pth`. If absent,
detection silently falls back to a Dirichlet mock and every record carries
`"mock": true`; the rest of the pipeline runs unchanged.

### 3.3 Run — dry-run (no camera, CI-safe)

```bash
python main.py --dry-run --duration 60 --user-id ci_smoke
```

### 3.4 Run — real session with a user profile

```bash
python main.py --user-id alice --duration 1800
```

Stop at any time with **Ctrl-C** (graceful shutdown — camera released immediately).
The session log is written to `logs/session_<ts>.jsonl`; the profile is updated and
saved at `profiles/<user-id>.json`.

### 3.5 Run with OBS Virtual Camera

```bash
# 1. Install OBS Studio; add a Media Source pointing at a learner video.
# 2. Tools → Start Virtual Camera.
# 3. Run the session — OBS will be picked up automatically:
python main.py --user-id obs_demo --duration 720
```

Verify the detection agent logs `Camera opened on index N` at startup and that N
matches OBS.

### 3.6 Run on pre-recorded video

Either route video through OBS Virtual Camera (recommended), or monkey-patch
`_open_camera` to `cv2.VideoCapture('path/to/video.mp4')`. The latter is a one-line
edit (not exposed as a flag) intended for short debugging sessions.

### 3.7 Evaluate a session

```bash
python scripts/validate_logs.py logs/<session>.jsonl
python scripts/compute_metrics.py logs/<session>.jsonl --out metrics.csv
python scripts/sensitivity_analysis.py logs/<session>.jsonl --out sensitivity.csv \
    --n-perturbations 50
```

### 3.8 Build a distributable release (`EduAgent_Release.zip`)

```bash
python build_release.py
```

Produces `EduAgent_Release.zip` (≈ 600–800 MB) containing a PyInstaller
`--onedir` bundle, the message bank, the model weights, empty `profiles/` and
`logs/` directories, and a `README_User.txt` quickstart for end users. Build
takes ≈ 5–15 minutes; the script self-cleans `build/`, `dist/`, and the
`.spec` file on exit. See `docs/CHANGES_post_validation.md` § 8 for the
rationale (`--onedir` vs `--onefile` + torch DLLs).

## 4. Architecture

Four cooperating agents communicate **only** through a thread-safe blackboard
(`SharedState`); no agent calls another directly.

```
        webcam frames                                  learner responses
             │                                                  ▲
             ▼                                                  │
    ┌──────────────────┐                            ┌──────────────────────┐
    │  Detection Agent │                            │   Delivery Agent     │
    │  TimeSformer →   │                            │  toast / persistent  │
    │  softmax label   │                            │  Tier-3 widget       │
    └────────┬─────────┘                            └──────────▲───────────┘
             │ append_engagement                              │ consume_pending_message
             ▼                                                 │
   ┌──────────────────────────────────────────────────────────────────────┐
   │                       SharedState  (blackboard)                       │
   │  engagement_history · belief_state · pending_action ·                 │
   │  pending_message · interventions · response_history · flags          │
   └──────────────────────────────────────────────────────────────────────┘
             │ get_belief / set_belief                        ▲
             ▼ set_pending_action                              │ set_pending_message
    ┌──────────────────┐  pending_action  ┌────────────────────────────┐
    │ Orchestrator     │ ───────────────► │  Intervention Agent        │
    │ Bayesian belief  │                  │  tier_1/2: random + no-rep │
    │ + utility choice │                  │  tier_3: persistent widget │
    └──────────────────┘                  └────────────────────────────┘
```

Full architectural reference: **[`docs/system_architecture.md`](docs/system_architecture.md)**.
Per-tick chronological walkthrough: **[`docs/runtime_flow.md`](docs/runtime_flow.md)**.

## 5. Documentation index

| Document | Purpose |
|---|---|
| [`docs/consent_form.md`](docs/consent_form.md) | Participant informed-consent template; read before first run. |
| [`docs/ethics_statement.md`](docs/ethics_statement.md) | Privacy, function-creep prevention, autonomy, bias, ground-truth limits, the utility matrix as ethical artifact, safety failures, human oversight. |
| [`docs/PEAS.md`](docs/PEAS.md) | PEAS specification for the whole agent + per-agent decomposition. |
| [`docs/system_architecture.md`](docs/system_architecture.md) | High-level architecture, file-by-file reference, data flow, agent loop, configuration system, failure modes, runtime states, ASCII diagrams, reproducibility notes, OBS testing procedure. |
| [`docs/runtime_flow.md`](docs/runtime_flow.md) | Chronological execution walkthrough — what happens from startup to shutdown, second by second. |
| [`docs/parameters_reference.md`](docs/parameters_reference.md) | Every configurable parameter with default, range, effect, and tuning notes. |
| [`docs/CHANGES_post_validation.md`](docs/CHANGES_post_validation.md) | Single canonical diff index of every code-affecting change since the initial teacher-validated build (timing changes, Tier-3 persistence, intent-aware Bahasa rendering, PyInstaller-aware paths, release build script). |

## 6. Privacy & ethics

**Before first run: read [`docs/consent_form.md`](docs/consent_form.md).**

Highlights of the privacy-by-design contract — all auditable in the source:

- **Raw webcam frames are NEVER written to disk.** Frames live in an in-memory ring
  buffer (`mas_engagement/agents/detection.py:68`) for ≤ 10 s and are discarded after
  inference. The only outbound write from detection is `blackboard.append_engagement
  (level, confidence, ts, softmax=…)` — three numbers and a 4-vector.
- **Audio is never captured.** No microphone device is ever opened.
- **No network client.** No HTTP, no socket, no cloud SDK. Data physically cannot
  leave the machine through the application.
- **No grading, attendance, scoring, or instructor reporting.** Engagement labels
  drive in-session prompts only.
- **Persistent Tier-3 is dismissable, never coercive.** The widget waits for the
  learner to click "Lanjutkan" or "Istirahat sebentar" — both supportive.
- **Bounded interruption rate.** Four governors stack: 600 s warmup, 180 s cooldown,
  fatigue penalty, 4-prompt budget cap.

See [`docs/ethics_statement.md`](docs/ethics_statement.md) for the full statement.

## 7. Reproducibility

| Aspect | Value / command |
|---|---|
| Python version | 3.12.0 |
| Pin dependencies | `pip freeze > requirements.lock.txt` |
| Deterministic seeds | The reasoning core is deterministic (pure functions). For deterministic dry-runs, seed `np.random.default_rng(seed)` in `_mock_detection_loop`. Intervention `random.choice` can be seeded externally. |
| Model checksum | `sha256sum mas_engagement/data/E3_WCE_lr1e5.pth` (Linux/macOS) <br> `Get-FileHash mas_engagement/data/E3_WCE_lr1e5.pth -Algorithm SHA256` (Windows PowerShell) |
| Tests | `python -m pytest mas_engagement/tests -q` (37 tests) |
| Frozen-release path layout | Paths derive from `_BASE_DIR` in `config.py`, which is `Path(sys.executable).resolve().parent` when `sys.frozen` else repo root. Frozen layout: `EduAgent_Release/EduAgent.exe`, `_internal/` (Python+DLLs), `mas_engagement/data/`, `profiles/`, `logs/` — all on the same level. See `docs/CHANGES_post_validation.md` § 7. |
| Recommended hardware | Mid-range laptop with a 720p webcam. Inference is CPU-bound (~ 1–2 s per 8-frame clip on a 2020-era CPU). GPU optional. |

## 8. Testing

| Area | How |
|---|---|
| **Unit + integration** | `pytest mas_engagement/tests -q` runs the belief, utility, blackboard, and orchestrator trace tests (36 tests total). |
| **Orchestrator trace replay** | `pytest mas_engagement/tests/test_orchestrator.py -v` exercises 10 canonical scenarios (steady_engaged, warmup_violation, gradual_drift, rapid_disengagement, full_escalation, low_confidence, ignored_interventions, responsive_recovery, max_budget, tier3_never_clicked). |
| **Log validation** | `python scripts/validate_logs.py logs/<session>.jsonl` — line-by-line schema + belief-sum-to-1 check; exits 0 if clean, 1 with the first offending line. |
| **Dry-run simulation** | `timeout 65 python main.py --dry-run --duration 60 --user-id ci_smoke` runs the whole pipeline without a webcam. |
| **Webcam validation** | `python -m mas_engagement.agents.detection --duration 30` posts 15 softmax records to a log. `grep -c '"softmax"' logs/session_<ts>.jsonl` should return ≥ 2. |
| **Intervention validation (Tier-1/2/3)** | `python -m mas_engagement.agents.delivery --test-tier {1,2,3}` shows the corresponding overlay. Tier-3 loads from `messages.json` and uses the persistent widget. |
| **OBS Virtual Camera** | Start OBS → Tools → Start Virtual Camera. Then run `python main.py --user-id obs_test --duration 720`. The detection agent's startup line `Camera opened on index N` will name the OBS device. |
| **Sensitivity / decision stability** | `python scripts/sensitivity_analysis.py logs/<session>.jsonl --out sensitivity.csv --n-perturbations 50` perturbs `UTILITY_MATRIX` by ±20 % and reports decision-stability rate. |

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Camera not found on indices 0-2` | No webcam, OBS Virtual Camera off, or driver issue | Start OBS Virtual Camera; or run with `--dry-run`; or raise `CAMERA_SCAN_RANGE`. |
| Detection always logs `"mock": true` | Weights missing or load failed | Place `E3_WCE_lr1e5.pth` under `mas_engagement/data/`. Verify SHA-256 against your training run. |
| CUDA error / out-of-memory | Default config is CPU-only; if a system-wide PyTorch is using CUDA, force CPU | `export CUDA_VISIBLE_DEVICES=""` before invoking. |
| Tk overlay never appears | No display server (CI), tk not installed | Use `--dry-run`; or run on a desktop. |
| No interventions ever fire in a short dry-run | Production warmup is `600 s`; durations under 10 minutes will sit entirely inside warmup | Use a session > 700 s, or temporarily lower `WARMUP_SEC` / `WARMUP_DURATION` (keep them equal). |
| Tier-3 toast doesn't auto-dismiss | Tier-3 is intentionally persistent (`TIER3_PERSISTENT = True`) | Click "Lanjutkan" or "Istirahat sebentar". |
| Frozen `EduAgent.exe` crashes with `WinError 1114` (`c10.dll`) | Built with `--onefile`; torch sibling DLLs not initialised in time | `build_release.py` uses `--onedir`. Delete `build/` `dist/` `.spec` and rebuild. |
| `VCRUNTIME140.dll missing` on the target machine | Microsoft VC++ Redistributable not installed | Install `vc_redist.x64.exe` from microsoft.com. |
| Frozen `.exe` works in `EduAgent_Release/` but crashes if moved | Launcher loads DLLs from `_internal/` beside it | Keep the folder together; right-click → Send to → Desktop (shortcut). |
| OBS Virtual Camera not picked up | OBS not started, or OBS using a non-scanned index | Start OBS; raise `CAMERA_SCAN_RANGE` to 4–6. |
| `pytest` fails on `test_messages_json_parses_with_validated_shape` | `data/messages.json` was edited and no longer matches the validated schema | Restore it from git or re-paste the contents documented in `agents/intervention.py::_load_and_validate`. |
| Profile is empty after a session | Logger path mismatch between `logger.path` and `update_profile_from_session(profile, log_path)` | They are wired together in `main.py`; ensure `logs/` is writable and `update_profile_from_session` reads the path actually written to. |

## 10. License & ethics restrictions

This is research software. **Do not deploy it in any setting where engagement labels
could affect grades, attendance, evaluations, or instructor reporting.** See
[`docs/ethics_statement.md`](docs/ethics_statement.md) §2 (function-creep prevention)
for the architectural constraints that make secondary use difficult by design — and
for the explicit policy that even those constraints are insufficient justification
for deploying the system without ethics review.

---

*Repository layout:*

```
main.py                       session entry point
mas_engagement/
  agents/                     detection · orchestrator · intervention · delivery
  reasoning/                  belief · utility · profile (pure functions)
  overlay/                    tk_overlay (toast + persistent Tier-3 widget)
  blackboard.py               thread-safe SharedState
  config.py                   all tunable constants
  data/                       messages.json, model weights (.pth — not committed)
  tests/                      belief · utility · blackboard · orchestrator + trace fixtures
scripts/
  compute_metrics.py          session-level metrics + belief plot
  sensitivity_analysis.py     UTILITY_MATRIX perturbation study
  validate_logs.py            JSONL log validator
  train_*.py / preprocess_*.py / explore_dataset.py  (offline model training)
src/                          training-time models / datasets / preprocessing
docs/                         consent · ethics · PEAS · architecture · runtime · parameters
logs/                         session JSONL output (not committed)
profiles/                     per-user profiles (not committed)
```
