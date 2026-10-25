# Live experiment progress and FFN/MoE benchmark

## Progress logs

The unified GitHub Actions workflow streams structured JSON progress records to the job log. The training script emits events for:

- run start, corpus read, tokenizer load, and base model load;
- activation capture by batch, including processed samples and ETA;
- captured token count and MoE initialization;
- each training step, current batch MSE, steps/second, and estimated remaining time;
- evaluation batches and completion;
- benchmark batches, throughput comparison, benchmark JSON save, and checkpoint save.

The corpus collector also reports scan progress every 500 source rows. These are live job logs, not a separate dashboard.

## Fair layer benchmark

The benchmark compares the selected original Qwen FFN with the trained top-2 MoE replacement using the same captured token vectors, the same device, and the same low-precision input dtype. It measures batch sizes of one and up to 32 token vectors, with configurable warmup and repeated timed forwards. The default full experiment uses 3 warmup iterations and 12 timed repetitions per model and batch size; the smoke job uses 1 warmup and 3 repetitions.

Metrics per model/batch include median and mean milliseconds per batch, median milliseconds per token vector, and token vectors per second. The JSON also records the MoE/original throughput ratio, latency ratio, and expert routing assignment counts. MoE measurements include router and Python expert-dispatch overhead.

Output path in the full training job:

`artifacts/layer0_persian_top2_experiment.benchmark.json`

The workflow uploads this JSON beside the checkpoint and corpus manifest, then the GitHub-to-Hugging-Face sync copies it into `experiments/`.

## Scope and limitations

- This is a **single FFN layer forward-pass benchmark**, not full language-model generation.
- It does not measure time-to-first-token, autoregressive tokens/sec, full-model memory, Android performance, or reasoning quality.
- CPU results on a GitHub-hosted runner are useful for comparing the two layer implementations on the same runner, but must not be presented as phone results.
- Dataset activation capture and evaluation use the current 64-sample experiment; the evaluation split is a token-level split from those samples, not a document-independent held-out corpus.
- The prototype is still a research checkpoint, not a complete converted Qwen MoE model.

## Next milestone

Once the benchmark output is available and inspected, the next meaningful step is a controlled multi-layer conversion plan: validate the dense-to-MoE FFN replacement and quality/perplexity checks on additional layers before attempting a full-model conversion. Do not infer that target total/active parameter budgets or mobile viability have been achieved from this one-layer experiment.
