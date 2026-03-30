# Packed Next Wave

Goal: attack packed `bpb` directly now that the oscillatory scale law is mapped.

## Questions

1. What is the best under-cap oscillatory scale between `16x` and `18x`?
2. Does longer training on the under-cap winner still buy packed score?
3. Does training through the packed constraint help more than post-train quantization?
4. Is mixed precision by subsystem better than uniform packing?
5. Does a static gate over monotone vs oscillatory reservoir banks help the packed winner?

## Rows

- cap-fit sweep: `16.5x`, `17.0x`, `17.5x`, `87.5%`, `1500` steps
- longer-train: `16.0x`, `87.5%`, `1800` and `2200` steps
- pack-train: `16.0x`, `87.5%`, `1800` steps, train under `int6`
- subsystem quant probe: `16.0x`, `87.5%`, compare uniform vs linear/local mixed precision
- static bank gate pilot: `16.0x`, `87.5%`, `1500` and `1800` steps
