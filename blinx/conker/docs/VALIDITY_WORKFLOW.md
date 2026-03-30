# Validity Workflow

`Conker` is the umbrella repo, but it no longer tries to own every layer itself.

The current split is:

- `conker`: model code, runners, artifacts, candidate manifests
- `conker-detect`: structural, artifact-boundary, and runtime-legality audits
- `conker-ledger`: validity-bundle packaging, backlog scans, lineage, and survival reports

That means the integration point should be a workflow, not a repo merge.

## One Path

1. run the model in `conker`
- produce checkpoints, packed artifacts, and JSON result files under `conker/out`

2. run audits in `conker-detect`
- example outputs:
  - matrix audit
  - bundle audit
  - legality audit
- write those JSONs somewhere stable, usually under `conker/out/detect`

3. package the claim from `conker`
- use `conker/scripts/run_validity_bundle.py`
- that helper points at the sibling `conker-ledger` repo and assembles the final bundle

## Starter Manifest

Template:

- [validity_manifest.template.json](/Users/asuramaya/Code/carving_machine_v3/conker/submissions/validity_manifest.template.json)

Generator:

```bash
python3 conker/scripts/run_validity_bundle.py template \
  conker/submissions/example_validity \
  --bundle-id conker-example
```

That writes:

- `claim.json`
- `metrics.json`
- `provenance.json`
- `audits.json`
- `validity_manifest.json`

Fill those in, then add detector outputs to the manifest `attachments` list.

## Bundle Assembly

```bash
python3 conker/scripts/run_validity_bundle.py bundle \
  conker/submissions/example_validity/validity_manifest.json \
  conker/submissions/example_validity_bundle
```

This does not reimplement `conker-ledger`. It imports the sibling repo and calls its bundle writer directly.

## Why This Split Holds

`conker-detect` should stay credible as an auditor.
`conker-ledger` should stay clean as a packaging/report tool.
`conker` should stay the place where model claims originate.

So the unification happens at the interface:

- shared JSON evidence
- shared manifest shape
- one `conker` helper entrypoint

Not by folding all three codebases back into one monolith.
