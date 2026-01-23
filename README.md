# Align is not Enough: Multimodal Universal Jailbreak Attack

This project is a refactored implementation of the multimodal universal jailbreak attack against Multimodal Large Language Models (MLLMs). The original codebase is from [wangyouze/multimodal-universal-jailbreak-attack](https://github.com/wangyouze/multimodal-universal-jailbreak-attack.git), which has been restructured and improved for better maintainability and consistency.

## Project Overview

This project implements a white-box multimodal universal jailbreak attack that simultaneously optimizes adversarial images and adversarial text suffixes to bypass safety mechanisms in MLLMs. The attack uses VMI-FGSM (Variance-Minimizing Iterative Fast Gradient Sign Method) for both image and text optimization.

## Project Structure

```
aine/
├── LLaVA-1.5/          # LLaVA-1.5 attack implementation
│   ├── attack/        # Attack modules (text, visual, multimodal)
│   ├── utils/         # Utility modules (model loader, prompt wrapper, etc.)
│   ├── data/          # Dataset files
│   ├── results/       # Attack results and adversarial images
│   ├── main.py        # Main entry point
│   └── README.md      # LLaVA-1.5 specific documentation
│
├── MiniGPT-v2/        # MiniGPT-v2 attack implementation
│   ├── attack/        # Attack modules (text, visual, multimodal)
│   ├── utils/         # Utility modules (prompt wrapper, generator, etc.)
│   ├── data/          # Dataset files
│   ├── results/       # Attack results and adversarial images
│   ├── minigpt4/      # MiniGPT-v2 model implementation
│   ├── main.py        # Main entry point
│   └── README.md      # MiniGPT-v2 specific documentation
│
└── README.md          # This file
```

## Implementation Details

### Attack Components

Both implementations include:

1. **Visual Attack Module**: Optimizes adversarial images using VMI-FGSM
   - Iteratively perturbs images to maximize target loss
   - Supports batch processing for multiple goals

2. **Text Attack Module**: Optimizes adversarial text suffixes using VMI-FGSM
   - Generates candidate token replacements using gradient-based search
   - Uses top-k sampling and batch evaluation for efficient optimization

3. **Multimodal Attack Orchestration**: Coordinates image and text optimization
   - Alternates between image and text optimization steps
   - Evaluates attack success on test goals periodically

### Key Features

- **Unified Attack Framework**: Consistent attack logic across different MLLM architectures
- **Batch Processing**: Efficient handling of multiple goals simultaneously
- **Modular Design**: Clean separation of concerns for easy maintenance
- **Comprehensive Evaluation**: Automatic testing on harmful behavior datasets

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- HuggingFace account with access to model repositories

### Installation

Each model implementation has its own requirements. Please refer to the respective README files:

- **LLaVA-1.5**: See [LLaVA-1.5/README.md](LLaVA-1.5/README.md)
- **MiniGPT-v2**: See [MiniGPT-v2/README.md](MiniGPT-v2/README.md)

### Running the Attacks

For detailed execution instructions, please refer to the individual model README files:

- **LLaVA-1.5**: See [LLaVA-1.5/README.md](LLaVA-1.5/README.md) for usage examples
- **MiniGPT-v2**: See [MiniGPT-v2/README.md](MiniGPT-v2/README.md) for usage examples

## Dataset

The harmful behavior datasets used in this project are from the [GCG repository](https://github.com/llm-attacks/llm-attacks):
- Training goals: `harmful_behaviors.csv` (520 goals)
- Test goals: `test_harmful_behaviors.csv` (470 goals)

## Victim Models

This implementation supports attacks against:

- **LLaVA-1.5**: 7B, 13B, and 34B variants
- **MiniGPT-v2**: 7B model

For model checkpoint downloads and setup, refer to the original repository: [wangyouze/multimodal-universal-jailbreak-attack](https://github.com/wangyouze/multimodal-universal-jailbreak-attack.git)

## Refactoring Notes

This codebase has been refactored from the original implementation with the following improvements:

- **Consistent Project Structure**: Unified directory layout across both model implementations
- **Code Organization**: Better separation of attack logic, utilities, and configuration
- **Dependency Management**: Cleaned up unnecessary dependencies and created minimal `requirements.txt` files
- **Configuration Management**: Centralized configuration using `.env` files and `config.py`
- **Documentation**: Comprehensive README files for each implementation
- **Compatibility**: Fixed transformers and peft library compatibility issues

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{wangyouze2024align,
  title={Align is not Enough: Multimodal Universal Jailbreak Attack against Multimodal Large Language Models},
  author={Wang, Youze and others},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project maintains the same license as the original repository. Please refer to the original repository for license information.

## Acknowledgments

- Original implementation: [wangyouze/multimodal-universal-jailbreak-attack](https://github.com/wangyouze/multimodal-universal-jailbreak-attack.git)
- GCG dataset: [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)
