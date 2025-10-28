# Machine-Unlearning  
A Python toolkit for experimenting with machine unlearning, built around the SISA (Sharded, Isolated, Sliced, Aggregated) framework.

## Features  
- Data loading / preprocessing utilities (`data_utils.py`).  
- SISA-style utilities for sharding, slicing, isolating and aggregating training portions (`sisa_utils.py`).  
- Core unlearning logic and orchestration (`mu.py`).  
- Visualization utilities to inspect model behaviour, unlearning impact, forgetting/retention dynamics (`viz_utils.py`).  
- Modular, extensible codebase suitable for experimentation with different model types, metrics and unlearning strategies.

## Structure  
- `data_utils.py`: functions and classes to load datasets, partition shards/slices, handle data removal/unlearning scenarios.  
- `sisa_utils.py`: implements the SISA unlearning workflow — shard creation, slice training, fast unlearning via slice replacement, aggregation of slice outputs.  
- `mu.py`: main orchestration of the unlearning pipeline: setup, model training, unlearning events, measurement of forgetting/retention.  
- `viz_utils.py`: helper routines to generate plots, track change in performance before vs after unlearning, visualize shard/slice structure, etc.

## Getting Started  
### Installation  
Clone this repository:  
```bash  
git clone https://github.com/Mishil449861/Machine-Unlearning.git  
cd Machine-Unlearning  
pip install -r requirements.txt
