# Frozen VLM backbone + Omni audio fine-tuning

Use the pinned caspar VLM language/vision weights and tokenizer. Initialize only
Talker and audio projector from the native upstream Omni checkpoint. Train on
sft_t2a_mini and sft_a2a_mini, excluding 128 deterministic heldout rows each.
Freeze the full VLM and external encoders; verify the original VLM tensors exactly.

Stages: T2A Talker (2e-5), A2A audio projector (1e-4), A2A both (2e-5),
one full epoch each, batch 16 × accumulation 2, sequence length 1024, seed 42.
Audio stop-token loss has weight 10. No audio augmentation or scheduled sampling.
A2A user prompts contain only audio placeholders, with no transcript leakage.

Run setup.sh from the remote project directory with OMNI_RUN_ROOT set internally.
This is a billable remote training workflow; reading these files does not start it.
Logs, models, metrics and samples belong under outputs/minimind-o/<run-id>/;
final reports belong under reports/minimind-o/<run-id>.md.
Download artifacts, verify all local/Hub hashes, then remove this run's Pod and
network volume. A watchdog stops this run's Pod on failure or after 12 hours.
Hugging Face and Runpod credentials stay on the host.

## Host capacity monitor

`monitor.py` runs only on the host and reads the existing `.env` key in memory.
It checks Secure Cloud RTX 5090 every 600 seconds for up to 24 hours, with a
$1.19/GPU-hour ceiling. A failed placement deletes its empty network volume.
Once a Pod is allocated it deploys a credential-free source bundle, starts the
12-hour-bounded pipeline, downloads and verifies results, uploads a new private
`caspar/MiniMind-O-VLM-Audio-Mini-5090-20261109` repository, verifies every Hub
Git/LFS hash, then deletes this run's Pod and volume. On a pipeline failure it
stops the Pod and preserves network checkpoints for diagnosis.

Host status: `outputs/minimind-o/20261109-vlm-audio-auto/status.json`.
Logs: `monitor.log`, `remote-latest.txt`, and, on failure, `remote-error.log`.
Final report: `reports/minimind-o/20261109-vlm-audio-auto.md`.
The monitoring process is tied to this host; host shutdown also stops monitoring.
