"""
LLaVA model loader
"""
import os
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def load_llava_model(model_path, device='cuda', load_in_8bit=False):
    """
    Load LLaVA model and processor.
    
    Args:
        model_path: HuggingFace model ID or local path
        device: Device to load model on
        load_in_8bit: Whether to use 8-bit quantization
    
    Returns:
        model: LLaVA model
        processor: LLaVA processor
    """
    logger.info(f'Loading LLaVA model from: {model_path}')
    
    # Set HuggingFace token if available
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
        os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
    
    # Load processor
    processor = LlavaProcessor.from_pretrained(model_path)
    
    # Load model
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={'': device},
            quantization_config=quantization_config,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
    
    model.eval()
    logger.info('LLaVA model loaded successfully')
    
    return model, processor


def get_embedding_matrix(model):
    """
    Get embedding matrix from LLaVA model.
    LLaVA uses LlamaForCausalLM internally.
    """
    # LLaVA의 언어 모델 부분 접근
    if hasattr(model, 'language_model'):
        # LLaVA-1.5 구조
        if hasattr(model.language_model, 'model'):
            return model.language_model.model.embed_tokens
        else:
            return model.language_model.embed_tokens
    elif hasattr(model, 'get_input_embeddings'):
        # 일반적인 transformers 모델
        return model.get_input_embeddings()
    else:
        raise AttributeError("Cannot find embedding matrix in model")
