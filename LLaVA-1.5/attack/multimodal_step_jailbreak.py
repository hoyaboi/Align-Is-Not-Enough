"""
Multimodal step jailbreak attack for LLaVA model
Main attack orchestration logic
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from copy import deepcopy
import random
from tqdm import tqdm
from torchvision.utils import save_image
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attack.visual_attack import VisualAttacker
from attack.text_attack import TextAttacker
from utils.generator import Generator
from utils.prompt_wrapper import Prompt
import config

DEFAULT_IMAGE_TOKEN = '<image>'


def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class MultimodalStepsJailbreakAttack:
    """Multimodal step jailbreak attack for LLaVA."""
    
    def __init__(self, model, processor, embedding_weight, conv_template, 
                 test_prefixes, iters, json_file_path, device, save_dir=None, test_goals=None, max_new_tokens=300):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.embedding_weight = embedding_weight
        self.conv_template = conv_template
        self.iters = iters
        self.test_prefixes = test_prefixes
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else config.ADV_IMAGES_DIR
        self.json_file_path = json_file_path
        self.test_goals = test_goals if test_goals is not None else []
        self.max_new_tokens = max_new_tokens

        # Ensure save directory exists
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _update_ids(self, goal, control, target):
        """Update token IDs for prompt construction."""
        self.conv_template.messages = []
        
        self.conv_template.append_message(self.conv_template.roles[0], 
                                         f"{DEFAULT_IMAGE_TOKEN}\n")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._user_role_slice = slice(None, len(toks) - 2)
        
        self.conv_template.update_last_message(f"{DEFAULT_IMAGE_TOKEN}{goal}")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._goal_slice = slice(self._user_role_slice.stop, 
                                max(self._user_role_slice.stop, len(toks) - 1))
        
        separator = ' ' if goal else ''
        self.conv_template.update_last_message(
            f"{DEFAULT_IMAGE_TOKEN}{goal}{separator}{control}")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._control_slice = slice(self._goal_slice.stop, len(toks) - 1)
        
        # Ensure control slice is exactly 10 tokens
        if self._control_slice.stop - self._control_slice.start < 10:
            self._control_slice = slice(
                self._control_slice.start - (10 - (self._control_slice.stop - self._control_slice.start)),
                self._control_slice.stop)
        if self._control_slice.stop - self._control_slice.start > 10:
            self._control_slice = slice(
                self._control_slice.start + ((self._control_slice.stop - self._control_slice.start) - 10),
                self._control_slice.stop)
        
        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
        
        self.conv_template.update_last_message(f"{target}")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
        self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2)
        
        self.input_ids = torch.tensor(toks, device='cpu')
        self.conv_template.messages = []
        return self.input_ids
    
    def attack(self, train_goals, enhanced_goals, image, adv_control, target_label, batch_size=20):
        """Main attack loop."""
        # Store original image for comparison
        self._original_image = image.clone()
        
        textual_attack = TextAttacker(
            self.model, self.processor, goals=train_goals, targets=target_label,
            conv_template=self.conv_template, test_prefixes=self.test_prefixes,
            n_candidates=2, device=self.device
        )
        visual_attack = VisualAttacker(
            self.model, self.processor, target_label,
            test_prefixes=self.test_prefixes,
            conv_template=self.conv_template, device=self.device
        )
        my_generator = Generator(
            model=self.model, 
            processor=self.processor,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=1.05,
            top_p=0.9,
            do_sample=True
        )
        
        enhanced_goals_list = []
        if len(enhanced_goals) > 0:
            for k, values in enhanced_goals.items():
                for v in values:
                    enhanced_goals_list.append(v)
        
        adv_control_tokens = self.tokenizer([adv_control], return_tensors="pt").input_ids[:, 1:].squeeze(0).to(self.device)
        if adv_control_tokens.shape[0] > 10:
            adv_control_tokens = adv_control_tokens[:10]
        
        for epoch in tqdm(range(1, self.iters + 1)):
            print(f'epoch={epoch}')
            print('starting to perturb the image >>>>>')
            print(f'adv_control={adv_control}')
            print('randomly selected goals>>>>')
            print(f'[image batch_size]>>> {batch_size}')
            
            batch_goals = random.sample(train_goals, batch_size)
            if len(enhanced_goals) > 0:
                batch_enhanced_goals = []
                for k, v in enhanced_goals.items():
                    if k in batch_goals:
                        for x in v:
                            batch_enhanced_goals.append(x)
                
                batch_enhanced_goals = random.sample(enhanced_goals_list, 35)
                batch_goals = train_goals + batch_enhanced_goals
            else:
                batch_goals = train_goals
            
            batch_goals_sample = random.sample(batch_goals, batch_size)
            
            img_prompts_list = []
            for i in range(batch_size):
                separator = ' '
                text_prompt = f"[INST] {DEFAULT_IMAGE_TOKEN} {batch_goals_sample[i]+'.'}{separator}{adv_control} [/INST] "
                img_prompts_list.append(text_prompt)
            
            image = visual_attack.attack_vmifgsm(
                text_prompts=img_prompts_list, img=image,
                batch_size=batch_size,
                num_iter=50, alpha=1./255
            )
            
            print('starting to perturb the text prompt>>>>>')
            
            from utils.model_loader import get_embedding_matrix
            embedding_matrix = get_embedding_matrix(self.model)
            v = torch.zeros((10, embedding_matrix.weight.shape[0])).detach().to(self.device)
            
            text_batch_size = batch_size
            print(f'[text batch_size]>>> {text_batch_size}')
            
            for t in tqdm(range(0, 20)):
                print(f'[adv_control]>>> {adv_control}')
                control_slice = []
                text_prompts_list = []
                
                for i in range(text_batch_size):
                    separator = ' '
                    text_prompt_suffix = f"[INST] {DEFAULT_IMAGE_TOKEN} {batch_goals_sample[i]+'.'}{separator}{adv_control} [/INST] "
                    text_prompts_list.append(text_prompt_suffix)
                    
                    input_ids = self._update_ids(batch_goals_sample[i]+'.', adv_control, target_label).to(self.device)
                    control_slice.append(self._control_slice)
                
                adv_control, adv_control_tokens, loss, v = textual_attack.attack_vmifgsm(
                    v, control_slice, adv_control, adv_control_tokens,
                    text_prompts_list, image, batch_goals_sample, batch_size=text_batch_size
                )
                # Update adv_control_tokens from returned value
                if isinstance(adv_control_tokens, torch.Tensor):
                    if adv_control_tokens.shape[0] > 10:
                        adv_control_tokens = adv_control_tokens[:10]
                
                # Print updated adv_control after each step (matching MiniGPT-v2 behavior)
                print(f'[Updated adv_control]>>> {adv_control}')
                print(f'[Loss]>>> {loss.item():.6f}')
            
            save_image(image.cpu().squeeze(0), str(self.save_dir / f"adv_image_epoch_{epoch}.bmp"))
            torch.save(image.cpu().detach(), str(self.save_dir / f"adv_image_epoch_{epoch}.pt"))
            
            if epoch % 10 == 0:
                print(f'\n========== Testing at Epoch {epoch} ==========')
                
                # Test on all test goals
                test_results = []
                separator = ' '
                
                print(f"Testing on {len(self.test_goals)} test questions...")
                for idx, test_goal in enumerate(tqdm(self.test_goals, desc="Testing")):
                    text_prompts = f"<s>[INST] {DEFAULT_IMAGE_TOKEN} {test_goal+'.'}{separator}{adv_control} [/INST]"
                    
                    # Create prompt with image for generation
                    test_prompts = Prompt(
                        model=self.model,
                        processor=self.processor,
                        text_prompts=[text_prompts],
                        device=self.device
                    )
                    # Store original image for generator (before embedding conversion)
                    test_prompts.img_prompts = [[image]]
                    test_prompts.update_img_prompts([[image]])
                    test_prompts.update_context_embs()
                    
                    with torch.no_grad():
                        response = my_generator.generate(test_prompts)
                    
                    jailbroken = not any([prefix in response for prefix in self.test_prefixes])
                    
                    test_results.append({
                        "epoch": epoch,
                        "test_idx": idx,
                        "test_goal": test_goal,
                        "adv_control": adv_control,
                        "response": response,
                        "jailbroken": int(jailbroken)
                    })
                    
                    if (idx + 1) % 50 == 0:
                        print(f"Processed {idx + 1}/{len(self.test_goals)} test questions")
                
                # Calculate accuracy
                jailbroken_count = sum([r["jailbroken"] for r in test_results])
                test_acc = jailbroken_count / len(test_results) if len(test_results) > 0 else 0.0
                
                print(f'\nEpoch {epoch} Test Results:')
                print(f'  Total test questions: {len(test_results)}')
                print(f'  Jailbroken: {jailbroken_count}')
                print(f'  Test Accuracy: {test_acc:.4f}')
                
                # Append results to JSON file
                epoch_summary = {
                    "epoch": epoch,
                    "adv_control": adv_control,
                    "test_acc": test_acc,
                    "jailbroken_count": jailbroken_count,
                    "total_tests": len(test_results),
                    "test_results": test_results
                }
                
                # Write mode for first test epoch, append mode for subsequent test epochs
                # Check if file exists to determine if this is the first test epoch
                file_exists = os.path.exists(self.json_file_path)
                mode = "w" if not file_exists else "a"
                with open(self.json_file_path, mode, encoding='utf-8') as f:
                    json.dump(epoch_summary, f, ensure_ascii=False)
                    f.write('\n')
            else:
                # For non-test epochs, just save basic info
                experiment_results = [{"epoch": epoch, "adv_control": adv_control}]
                
                # Write mode for first epoch, append mode for subsequent epochs
                mode = "w" if epoch == 1 else "a"
                with open(self.json_file_path, mode, encoding='utf-8') as f:
                    for result in experiment_results:
                        json.dump(result, f, ensure_ascii=False)
                        f.write('\n')
        
        return adv_control, image
