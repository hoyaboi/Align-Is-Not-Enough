"""
Configuration management for LLaVA-1.5 project.
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
LLAVA_MODEL_PATH = os.getenv('LLAVA_MODEL_PATH', 'llava-hf/llava-1.5-7b-hf')

# Data Paths
TRAIN_DATA_PATH = Path(os.getenv('TRAIN_DATA_PATH', PROJECT_ROOT / 'data' / 'harmful_behaviors.csv'))
TEST_DATA_PATH = Path(os.getenv('TEST_DATA_PATH', PROJECT_ROOT / 'data' / 'test_harmful_behaviors.csv'))
CLEAN_IMAGE_PATH = Path(os.getenv('CLEAN_IMAGE_PATH', PROJECT_ROOT / 'data' / 'clean.jpeg'))

# Results Paths
RESULTS_DIR = Path(os.getenv('RESULTS_DIR', PROJECT_ROOT / 'results'))
ADV_IMAGES_DIR = RESULTS_DIR / 'adv_images'

# GPU Settings
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')

# Create necessary directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADV_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
