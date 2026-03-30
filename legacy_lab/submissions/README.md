# Submissions

Packaged candidate artifacts and manifests go here.

This is the place for:

- frozen candidate checkpoints
- compressed / quantized artifacts
- manifest files
- exact commands used to reproduce a candidate

Workflow:

1. collect model outputs under `conker/out`
2. collect detector JSONs with `conker-detect`
3. start from:
   - [validity_manifest.template.json](/Users/asuramaya/Code/carving_machine_v3/conker/submissions/validity_manifest.template.json)
   - or generate one with:
     `python3 conker/scripts/run_validity_bundle.py template ...`
4. assemble the final bundle with:
   `python3 conker/scripts/run_validity_bundle.py bundle ...`

That keeps `conker` as the umbrella entrypoint without folding `conker-detect` and `conker-ledger` back into this repo.
