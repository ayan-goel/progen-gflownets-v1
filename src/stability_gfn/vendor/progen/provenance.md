# Upstream provenance

Unmodified `configuration_progen.py`, `modeling_progen.py`, `tokenizer.json` and
`LICENSE.txt` copied from the existing checkout of
[Salesforce/progen](https://github.com/salesforce/progen), commit
`c27a419c234a0997923761e1fe7daffcebf0eaf5`. The original per-file copyright/license
headers and repository license are preserved. Our adapter lives outside this folder.

We call the model directly with `past_key_values`, avoiding the old upstream
`generate()` compatibility issue with Transformers 4.27.4. No upstream source
modification is necessary. Checkpoints are separate read-only inputs.
