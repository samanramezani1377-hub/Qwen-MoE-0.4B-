# Training data

## Hybrid ChatGPT + Qwen distillation archive

The current text source for the single hybrid training path is:

- `chatgpt_prompt_response_archive_part2.jsonl`

Each line is one UTF-8 JSON object with these required fields:

- `sample_id`: unique stable sample identifier
- `prompt`: complete user prompt
- `chatgpt_response`: final response used for text cross-entropy

Validate the archive from the repository root:

```bash
python scripts/validate_chatgpt_archive.py
```

The Colab notebook `notebooks/02_colab_chatgpt_text_qwen_logits.ipynb` reads this archive from the checked-out repository. It generates teacher logits and training outputs under the configured Google Drive workspace. Keep checkpoints and large generated logits/training outputs in Google Drive; do not commit those generated binary artifacts to this repository.

This file is a training source, not a held-out evaluation set. Do not use it as independent evaluation data.
