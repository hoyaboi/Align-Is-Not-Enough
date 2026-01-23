import argparse
import numpy as np

from minigpt4.common.registry import registry
from minigpt4.common.config import Config

# Optional import for BLEU score (not actively used in current code)
try:
    from nltk.translate.bleu_score import sentence_bleu
except ImportError:
    sentence_bleu = None

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *



def eval_parser():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--name", type=str, default='A2', help="evaluation name")
    parser.add_argument("--ckpt", type=str, help="path to configuration file.")
    parser.add_argument("--eval_opt", type=str, default='all', help="path to configuration file.")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="max number of generated tokens")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    return parser


def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts


def init_model(args):
    print('Initialization Model')
    cfg = Config(args)
    
    # Override llama_model if provided via command line or if default is placeholder
    current_llama_model = str(cfg.model_cfg.get('llama_model', ''))
    if hasattr(args, 'llama_model') and args.llama_model and args.llama_model.strip():
        # Always use command line argument if provided
        cfg.model_cfg.llama_model = args.llama_model
        print(f'Using llama_model from command line: {args.llama_model}')
    elif 'please set this value' in current_llama_model.lower() or current_llama_model.strip() == '':
        # If default is placeholder or empty, raise error
        raise ValueError(
            f"llama_model is not set! Current value: '{current_llama_model}'. "
            f"Please set --llama_model argument or configure LLAMA_MODEL_PATH in .env file."
        )
    
    # Override checkpoint if provided
    if hasattr(args, 'ckpt') and args.ckpt:
        cfg.model_cfg.ckpt = args.ckpt
    
    # Override lora settings if provided
    if hasattr(args, 'lora_r') and args.lora_r:
        cfg.model_cfg.lora_r = args.lora_r
    if hasattr(args, 'lora_alpha') and args.lora_alpha:
        cfg.model_cfg.lora_alpha = args.lora_alpha

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

#     import pudb; pudb.set_trace()
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print('Initialization Finished')
    return model, vis_processor

def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou
