# Rescue

This page explains what `giddy-up` consolidates from the older split layout.

## What Was Rescued

- oracle logic that had been split across BLINX-local helpers
- causal bridge feature definitions that had been split across Conker-local helpers
- the boundary contract between those two legality domains

## Why It Was Rescued

- the old layout duplicated bridge code across repos
- that made ownership and legality boundaries harder to read
- moving the bridge into its own repo makes the dependency direction explicit

## What Still Exists As Mirrors

- `conker/conker/src/giddy_up`: consumer-side mirror
- `blinx/conker/src/giddy_up`: producer-side mirror

Those mirrors are intentionally thin and no longer the canonical home.

## Archive Scope

- unlike `conker` and `blinx`, `giddy-up` does not keep a large rescued handoff or matrix archive
- the main rescue job here is consolidating split ownership and keeping the mirror relationship explicit

See also:

- [Salvage Matrix](../SALVAGE_MATRIX.md)
