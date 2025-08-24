
# Reproducibility Package: Novel Liquid Neural Networks for Ultra-Low-Power Audio Processing: A Comprehensive Study

This package contains all code, data, and configuration files necessary to reproduce 
the results presented in our paper.

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/terragon-labs/liquid-audio-nets.git
cd liquid-audio-nets

# 2. Setup environment
docker build -t liquid-audio-nets .
docker run -it liquid-audio-nets

# 3. Run main experiments
python reproduce_paper_results.py --config configs/paper_experiments.json

# 4. Generate figures
python generate_paper_figures.py --output figures/
```

## System Requirements

- **OS**: Ubuntu 20.04+ / macOS 12+ / Windows 10+
- **Python**: 3.8+
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space
- **Optional**: CUDA-compatible GPU for acceleration

## Package Structure

```
reproducibility_package/
├── README_REPRODUCIBILITY.md     # This file
├── configs/                      # Experiment configurations
│   ├── paper_experiments.json    # Main paper experiments
│   ├── ablation_studies.json     # Ablation study configs
│   └── baseline_comparisons.json # Baseline comparison configs
├── data/                         # Datasets and preprocessing scripts
│   ├── download_datasets.py      # Automated dataset download
│   ├── preprocess_audio.py       # Audio preprocessing pipeline
│   └── synthetic_data_gen.py     # Synthetic dataset generation
├── models/                       # Model implementations
│   ├── lnn_adat.py               # ADAT algorithm implementation
│   ├── lnn_hlmn.py               # HLMN algorithm implementation
│   ├── lnn_qchd.py               # QCHD algorithm implementation
│   └── baselines.py              # Baseline model implementations
├── experiments/                  # Experiment scripts
│   ├── run_paper_experiments.py  # Main paper experiments
│   ├── power_analysis.py         # Power consumption analysis
│   ├── accuracy_evaluation.py    # Accuracy benchmarking
│   └── statistical_tests.py      # Statistical significance testing
├── results/                      # Experimental results
│   ├── raw_results/              # Raw experimental outputs
│   ├── processed_results/        # Processed and analyzed results
│   └── paper_figures/            # Generated paper figures
├── docker/                       # Docker configuration
│   ├── Dockerfile                # Main container
│   ├── requirements.txt          # Python dependencies
│   └── environment.yml           # Conda environment
├── tests/                        # Unit tests and validation
│   ├── test_algorithms.py        # Algorithm correctness tests
│   ├── test_reproducibility.py   # Reproducibility validation
│   └── benchmark_suite.py        # Performance benchmarks
└── documentation/                # Additional documentation
    ├── ALGORITHM_DETAILS.md      # Detailed algorithm descriptions
    ├── EXPERIMENTAL_PROTOCOL.md  # Complete experimental protocol
    └── TROUBLESHOOTING.md        # Common issues and solutions
```

## Reproducing Main Results

### Table 1: Accuracy Comparison
```bash
python experiments/accuracy_evaluation.py --config configs/accuracy_comparison.json
```

### Figure 2: Power Efficiency Analysis  
```bash
python experiments/power_analysis.py --generate-figures
```

### Figure 3: Ablation Studies
```bash
python experiments/ablation_studies.py --all-components
```

## Validation and Testing

Run the complete test suite:
```bash
pytest tests/ --verbose --cov=models/
```

Validate reproducibility:
```bash
python tests/test_reproducibility.py --strict
```

## Hardware-Specific Instructions

### ARM Cortex-M4 Deployment
```bash
# Cross-compile for ARM
python deployment/cross_compile.py --target cortex-m4
# Flash to device  
python deployment/flash_device.py --device /dev/ttyUSB0
```

### RISC-V Deployment
```bash
python deployment/riscv_deploy.py --config configs/riscv_config.json
```

## Expected Results

When running the complete reproduction pipeline, you should obtain results 
matching the paper within ±1% accuracy and ±5% power consumption due to 
hardware variations.

## Troubleshooting

Common issues and solutions:

1. **CUDA out of memory**: Reduce batch size in config files
2. **Missing datasets**: Run `python data/download_datasets.py`
3. **Slow execution**: Enable GPU acceleration in configs
4. **Permission errors**: Check Docker permissions

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{liquid_audio_nets_2026,
  title={Novel Liquid Neural Networks for Ultra-Low-Power Audio Processing: A Comprehensive Study},
  author={Daniel Schmidt and Terragon Research Team},
  booktitle={ICASSP 2026},
  year={2026}
}
```

## Support

For questions or issues:
- GitHub Issues: https://github.com/terragon-labs/liquid-audio-nets/issues
- Email: daniel@terragon.dev
- Documentation: https://liquid-audio-nets.readthedocs.io

## License

This code is released under the MIT License. See LICENSE file for details.
