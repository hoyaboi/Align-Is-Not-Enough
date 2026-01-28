# MiniGPT-v2 Jailbreak Attack

Multimodal jailbreak attack implementation for MiniGPT-v2 model using adversarial images and adversarial suffixes.

## Project Structure

```
MiniGPT-v2/
├── attack/              # Attack implementation modules
│   ├── multimodal_step_jailbreak.py  # Main attack orchestration
│   ├── text_attack.py                 # Text suffix attack
│   └── visual_attack.py               # Image attack
├── utils/               # Utility modules
│   ├── data_utils.py    # Data loading utilities
│   ├── prompt_wrapper.py # Prompt handling
│   └── generator.py     # Text generation
├── data/                # Data files
│   ├── harmful_behaviors.csv
│   ├── test_harmful_behaviors.csv
│   └── clean.jpeg
├── results/             # Output directory
│   └── adv_images/     # Generated adversarial images
├── config.py            # Configuration management
└── main.py              # Main entry point
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

# LLaMA Model Path (HuggingFace model name or local path)
LLAMA_MODEL_PATH=meta-llama/Llama-2-7b-chat-hf

# MiniGPT-v2 Checkpoint Path
CHECKPOINT_PATH=/path/to/minigptv2_checkpoint.pth
```

## Usage

Run the attack:
```bash
python main.py \
    --cfg-path eval_configs/minigptv2_eval.yaml \
    --batch_size 6 \
    --iters 50 \
    --train_data data/harmful_behaviors.csv \
    --n_train_data 520 \
    --test_data data/test_harmful_behaviors.csv
```

## Parameters

- `--batch_size`: Number of goals to sample per epoch (default: 6)
- `--iters`: Total number of attack iterations (default: 50)
- `--train_data`: Path to training data CSV
- `--n_train_data`: Number of training samples
- `--test_data`: Path to test data CSV
- `--n_test_data`: Number of test samples (0 means use all)

## Attack Process

1. **Image Optimization**: Optimize adversarial image using VMI-FGSM (50 iterations)
2. **Text Suffix Optimization**: Optimize adversarial suffix using VMI-FGSM (20 iterations per epoch)
3. **Evaluation**: Test on test goals every 10 epochs

Results are saved to `results/minigpt_v2_results.jsonl`.
