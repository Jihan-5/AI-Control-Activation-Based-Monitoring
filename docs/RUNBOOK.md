# Runbook: running Research A

## Offline pilot (CPU, no downloads)

```bash
pip install -e ".[dev]"
bash scripts/run_pilots.sh          # A0, then the actor and data stages, then A1-A5
```

Or step by step:

```bash
actmon run   configs/a0_detector_gate.yaml --out results/a0_detector_gate
actmon stage actor configs/small_actor.yaml     # trains M1 (~25 min CPU) and M2 (~5 min)
actmon stage data  configs/small_actor.yaml     # screens and caches 14,700 roots x 22 passes
actmon run configs/a1_actor_audit.yaml   --out results/a1_actor_audit
actmon run configs/a2_detection.yaml     --out results/a2_detection
actmon run configs/a3_audit_utility.yaml --out results/a3_audit_utility
actmon run configs/a4_ablations.yaml     --out results/a4_ablations
actmon run configs/a5_obfuscation.yaml   --out results/a5_obfuscation
```

Every `run` writes a write-once `manifest.json` (config hash, git commit, library versions).
Rerunning a *different* config into the same directory fails. The work dir
(`results/work_small/`) caches weights, features and pool logs. Its `actor_meta.json`
stores the actor-config hash, and a mismatching config refuses to reuse it.

Plumbing check in under a minute: `actmon stage data configs/smoke.yaml`, then any
experiment with `configs/smoke.yaml experiment=<name>`.

## Declared actor (Qwen2.5-1.5B-Instruct, GPU)

Requirements: one GPU with ≥ 24 GB (bf16 weights ~3 GB per copy; two copies, M1 and M2, are held
during collection), and network access to huggingface.co or a pre-populated `HF_HOME`.

```bash
actmon stage actor configs/qwen2p5_1p5b.yaml   # LoRA adapters for M1 then M2, merged
# pin the resolved commit printed in results/work_qwen/actor/actor_meta.json
#   -> actor.qwen.revision in configs/qwen2p5_1p5b.yaml
actmon stage data  configs/qwen2p5_1p5b.yaml
for e in a1_actor_audit a2_detection a3_audit_utility a4_ablations a5_obfuscation; do
  actmon run configs/qwen2p5_1p5b.yaml experiment=$e --out results/${e}_qwen
done
```

Checks to read *before* any detector number (A1):

1. `hook_check.max_abs_diff` ≈ 0 (bf16: below 1e-2).
2. `competence_m1_id.valid` high, `competence_m2_heldout` reported.
3. `planted_mechanism.csv`: M2 violation rate above M1's on reserve/review conflicts, and higher with
   the cue than without. If not, follow decision rule 2 of the experiment plan.
4. `organism_drift.json`: M1–M2 disagreement on non-conflict and benign roots should be small.

Rough compute (one A100): adapters ~20 min each; collection ≈ 14,700 roots × ~22 forward passes
≈ 1–2 h; attack ≈ 3 targets × 2 λ × 300 steps × 7 passes × batch 8 ≈ 1 h.

## Things that silently invalidate a run

* Changing the chat template or the tokenizer revision between collection and attack (soft-prompt
  placeholder positions shift). The tokenizer fingerprint is in `actor_meta.json`.
* Changing `transformers` between collection and analysis: `ResidualRecorder` assumes the decoder
  layer returns the hidden state (or a tuple starting with it). `verify_sites` catches a change.
* Editing a split's quota after collection: delete `results/work_*/data/*_<split>.*` and recollect.
