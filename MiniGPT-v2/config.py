"""
Configuration management for the project.
Loads settings from environment variables and provides defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.absolute()))

# HuggingFace Token
HF_TOKEN = os.getenv('HF_TOKEN', '')

# Model Paths
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH', '')
CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', PROJECT_ROOT / 'checkpoints'))

# Data Paths
TRAIN_DATA_PATH = Path(os.getenv('TRAIN_DATA_PATH', PROJECT_ROOT / 'advbench' / 'harmful_behaviors.csv'))
TEST_DATA_PATH = Path(os.getenv('TEST_DATA_PATH', PROJECT_ROOT / 'advbench' / 'test_harmful_behaviors.csv'))
CLEAN_IMAGE_PATH = Path(os.getenv('CLEAN_IMAGE_PATH', PROJECT_ROOT / 'advbench' / 'clean.jpeg'))

# Results Paths
RESULTS_DIR = Path(os.getenv('RESULTS_DIR', PROJECT_ROOT / 'jailbreak_attack' / 'results'))
ADV_IMAGES_DIR = RESULTS_DIR / 'adv_images'

# Config Paths
EVAL_CONFIG_PATH = Path(os.getenv('EVAL_CONFIG_PATH', PROJECT_ROOT / 'eval_configs' / 'minigptv2_eval.yaml'))

# GPU Settings
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')

# Create necessary directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADV_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
