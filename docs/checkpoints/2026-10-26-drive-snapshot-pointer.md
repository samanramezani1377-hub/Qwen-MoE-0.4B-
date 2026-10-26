# Google Drive Snapshot Pointer — 2026-10-26

This document records the exact location of the local project snapshot created in Google Drive.

## Exact path

`/content/drive/MyDrive/qwen-moe-chatgpt-logits/status_checkpoints/2026-10-26-v1`

## Files in the snapshot

- `manifest.json` — inventory of the scripts, layer checkpoints, data files, and recorded SHA-256 hashes.
- `benchmark_results.json` — the three-round Tesla T4 generation benchmark.
- `STATUS.md` — human-readable project status.
- `SHA256SUMS.txt` — SHA-256 values recorded for the files listed in the snapshot.
- `source_snapshot/` — copies of the three important scripts that existed when the snapshot was created.

## Checkpoint inventory

- Expected layer checkpoints: 24
- Layer checkpoints recorded: 24
- Original checkpoint files were not copied over, moved, or rewritten by the snapshot operation.
- Training remained disabled; no training checkpoint was created by the benchmark.

## Important distinction

This GitHub file records the Drive path and snapshot contents. It does **not** upload or mirror the Drive files into GitHub, and the Drive path is a Colab-mounted path rather than a public download URL. The actual snapshot remains in the user's Google Drive.

## Related GitHub records

- [Project status and benchmark report](2026-10-26-benchmark-and-state.md)
- [Structured benchmark results](2026-10-26-benchmark-results.json)
