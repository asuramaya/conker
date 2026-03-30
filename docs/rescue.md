# Rescue

This page explains what `conker` preserves on purpose and how to read the rescued material safely.

## What Was Rescued

- the old non-git lab subtree now lives under [`../legacy_lab`](../legacy_lab/README.md)
- the inherited dense branch-note store lives under [`../conker/docs`](../conker/docs/README.md)
- invalidated branch notes are indexed in [`./branches/README.md`](./branches/README.md)
- early archived notes are indexed in [`./archive/README.md`](./archive/README.md)

## Why It Was Rescued

- the invalid branches still carry real architectural lessons
- the handoff and matrices still explain how the line evolved
- the legacy lab tree is useful for forensic comparison and provenance

## How To Read It

- treat `docs/` as the curated public layer
- treat `conker/docs/` as the detailed branch-note layer
- treat `legacy_lab/` as preserved history, not current runtime truth

See also:

- [Salvage Matrix](../SALVAGE_MATRIX.md)

## What Is Not Fully Normalized

- deeper historical notes still contain lab-era absolute path links
- old matrices and handoffs still reference local artifact layouts
- these are preserved for rescue value, not polished as current public docs
