# Scripts

Queue runners, eval scripts, packaging helpers, and submission prep live here.

Good contents:

- one-cell-at-a-time experiment queues
- Parameter Golf data setup helpers
- FineWeb / token-loss and bpb evaluation wrappers
- artifact packing and manifest generation
- reproducibility scripts for submission candidates
- official-data baselines and `Conker-1` runners

Current entrypoints:

- `link_parameter_golf_data.zsh`: symlink an existing official export into `conker/data`
- `run_golf_single_bridge.py`: official-data single-expert bridge
- `run_conker_frontier_golf_queue.zsh`: baseline-vs-`Conker-1` frontier queue
- `run_conker2_golf_bridge.py`: official-data bridge for the linear-plus-correction branch
- `run_conker2_golf_queue.zsh`: `Conker-2` seed queue
- `run_conker11_golf_bridge.py`: recursive causal router bridge
- `run_conker12_golf_bridge.py`: higher-order program-tensor bridge
- `run_conker13_golf_bridge.py`: five-axis substrate-controller bridge
