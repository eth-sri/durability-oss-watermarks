# Towards Watermarking for Open-Source Models

This repository contains code accompanying our ICLR 2025 WMark [workshop paper](https://arxiv.org/abs/2502.10525).

## Setup

To set up the project, clone the repository and install all dependencies from `env.yaml`. Additionally, install the extra package for faster computation of the KTH watermark:

```bash
pip install additional_data/levenshtein_rust_wheel/levenshtein_rust-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
```

## Repository Structure

The project structure is as follows:

- `src/` contains the source code, specifically:
  - `main.py`: Entry point for executing experiments.
  - `config.py`: Defines the main Pydantic configuration. Specific Pydantic configurations for each model modification reside in their respective subfolders.
  - `src/watermarks/`: Contains watermark detectors and benchmark implementations. To add a watermark, refer to `watermark_detector.py`, `scheme_config.py`, and `watermark_types.py`.
  - `src/model_modifications/`: Contains subfolders for model modifications (finetuning, merging, pruning, and quantization).
- `configs/`: YAML configuration files used for experiments reported in the paper.
- `additional_data/`: Contains the wheel file for the accelerated KTH implementation.

## Running the Code

First, you need to add the root of the repository to the Pythonpath

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Run the code by passing a configuration file as an argument:

```bash
python src/main.py --config configs/llama2/kgw/benchmark_all_kgw.yaml
```

This command benchmarks the KGW-D durability evaluation across all model modifications described in the paper. It produces multiple JSON Lines files corresponding to each model modification, including additional files for finetuning checkpoints. Each line in these files contains:
- Prompt
- Model completion
- P-value
- Perplexity

## Citation

If you use this code, please cite:

```bibtex
@misc{gloaguen2025watermarkingopensourcellms,
      title={Towards Watermarking of Open-Source LLMs}, 
      author={Thibaud Gloaguen and Nikola Jovanović and Robin Staab and Martin Vechev},
      year={2025},
      eprint={2502.10525},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.10525}, 
}
