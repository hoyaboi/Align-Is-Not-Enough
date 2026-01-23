# LLaVA-1.5 Jailbreak Attack

Multimodal jailbreak attack implementation for LLaVA-1.5 model using adversarial images and adversarial suffixes.

## Project Structure

```
LLaVA-1.5/
├── attack/              # Attack implementation modules
│   ├── multimodal_step_jailbreak.py  # Main attack orchestration
│   ├── text_attack.py                 # Text suffix attack
│   └── visual_attack.py               # Image attack
├── utils/               # Utility modules
│   ├── model_loader.py  # LLaVA model loader
│   ├── prompt_wrapper.py # Prompt handling
│   ├── generator.py     # Text generation
│   └── data_utils.py    # Data loading utilities
├── data/                # Data files
│   ├── harmful_behaviors.csv
│   ├── test_harmful_behaviors.csv
│   └── clean.jpeg
├── results/             # Output directory
│   └── adv_images/     # Generated adversarial images
├── config.py            # Configuration management
├── main.py              # Main entry point
└── requirements.txt     # Dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```bash
# HuggingFace Token
HF_TOKEN=your_huggingface_token

# LLaVA Model Path (HuggingFace model name or local path)
LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf
```

## Usage

Run the attack:
```bash
python main.py \
    --model_path llava-hf/llava-1.5-7b-hf \
    --batch_size 2 \
    --iters 50 \
    --n_train_data 520
```

## Parameters

- `--model_path`: LLaVA model path (HuggingFace ID or local path, default: from .env or `llava-hf/llava-1.5-7b-hf`)
- `--batch_size`: Number of goals to sample per epoch (default: 2, adjust based on GPU memory)
- `--iters`: Total number of attack iterations (default: 50)
- `--n_train_data`: Number of training samples to use (default: 520)
- `--n_test_data`: Number of test samples (0 means use all, default: 0)
- `--load_in_8bit`: Use 8-bit quantization to save memory (optional)
- `--name`: Experiment name for result files (default: "llava_attack")

## Attack Process

1. **Image Optimization**: Optimize adversarial image using VMI-FGSM (50 iterations)
2. **Text Suffix Optimization**: Optimize adversarial suffix using VMI-FGSM (20 iterations per epoch)
3. **Evaluation**: Test on test goals every 10 epochs

## Results

- Attack results are saved to `results/{name}_results.json`
- Every 10 epochs, the attack is evaluated on all test questions (470 questions) and results are saved
- Adversarial images are saved to `results/adv_images/`

## Notes

- The default batch size is set to 2 for LLaVA-1.5-7B to fit in ~16GB GPU memory
- For larger models (13B, 34B), reduce batch size or use `--load_in_8bit` flag
- The attack uses the same harmful behavior dataset from GCG as MiniGPT-v2
