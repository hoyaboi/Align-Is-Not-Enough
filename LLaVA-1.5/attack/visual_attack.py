"""
Visual attack module for LLaVA model
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import random

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.prompt_wrapper import Prompt
from utils.model_loader import get_embedding_matrix

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


class VisualAttacker:
    """Visual attacker for LLaVA model."""
    
    def __init__(self, model, processor, targets, test_prefixes, 
                 conv_template, device='cuda', is_rtp=False):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device
        self.is_rtp = is_rtp
        
        self.targets = targets
        self.num_targets = len(targets)
        self.loss_buffer = []
        
        # Freeze model
        self.model.eval()
        self.model.requires_grad_(False)
        
        self.test_prefixes = test_prefixes
        self.conv_template = conv_template
    
    def attack_loss(self, prompts, targets, non_targeted_text=None):
        """Compute attack loss."""
        context_embs = prompts.context_embs
        
        if len(context_embs) == 1:
            context_embs = context_embs * len(targets)
        
        assert len(context_embs) == len(targets), \
            f"Unmatched batch size: {len(context_embs)} != {len(targets)}"
        
        batch_size = len(targets)
        self.tokenizer.padding_side = "right"
        
        # Get language model
        if hasattr(self.model, 'get_model'):
            language_model = self.model.get_model()
        elif hasattr(self.model, 'language_model'):
            language_model = self.model.language_model
        else:
            language_model = self.model
        
        to_regress_tokens = self.tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
            add_special_tokens=False
        ).to(self.device)
        
        embedding_matrix = get_embedding_matrix(self.model)
        to_regress_embs = embedding_matrix(to_regress_tokens.input_ids)
        
        bos = torch.ones([1, 1], dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * \
              self.tokenizer.bos_token_id
        bos_embs = embedding_matrix(bos)
        
        pad = torch.ones([1, 1], dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * \
              self.tokenizer.pad_token_id
        pad_embs = embedding_matrix(pad)
        
        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )
        
        pos_padding = torch.argmin(T, dim=1)
        
        input_embs = []
        targets_mask = []
        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []
        
        for i in range(batch_size):
            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]
            
            targets_mask.append(T[i:i+1, :target_length])
            input_embs.append(to_regress_embs[i:i+1, :target_length])
            
            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length
            
            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)
        
        max_length = max(seq_tokens_length)
        attention_mask = []
        
        for i in range(batch_size):
            context_mask = torch.ones([1, context_tokens_length[i] + 1],
                                     dtype=torch.long).to(self.device).fill_(-100)
            
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = torch.ones([1, num_to_pad],
                                     dtype=torch.long).to(self.device).fill_(-100)
            
            targets_mask[i] = torch.cat([padding_mask, context_mask, targets_mask[i]], dim=1)
            input_embs[i] = torch.cat([
                pad_embs.repeat(1, num_to_pad, 1),
                bos_embs,
                context_embs[i],
                input_embs[i]
            ], dim=1)
            attention_mask.append(torch.LongTensor(
                [[0] * num_to_pad + [1] * (1 + seq_tokens_length[i])]))
        
        targets = torch.cat(targets_mask, dim=0).to(self.device)
        inputs_embs = torch.cat(input_embs, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)
        
        outputs = language_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Get logits from lm_head (language_model is LlamaModel, not LlamaForCausalLM)
        # So we need to get last_hidden_state and apply lm_head manually
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            # Get last_hidden_state and apply lm_head
            last_hidden_state = outputs.last_hidden_state
            # Get lm_head from the full model
            if hasattr(self.model, 'lm_head'):
                lm_head = self.model.lm_head
            elif hasattr(self.model, 'get_model') and hasattr(self.model.get_model(), 'lm_head'):
                lm_head = self.model.get_model().lm_head
            else:
                raise AttributeError("Cannot find lm_head in model")
            logits = lm_head(last_hidden_state)
        
        # Calculate loss manually
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        # Flatten the tokens
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return 10 * loss
    
    def attack_vmifgsm(self, text_prompts, img, batch_size=8, num_iter=50, alpha=1./255, decay=1.0):
        """VMI-FGSM attack on images."""
        eps = 32./255
        beta = 3/2
        img = img.clone().detach().to(self.device)
        momentum = torch.zeros_like(img).detach().to(self.device)
        v = torch.zeros_like(img).detach().to(self.device)
        
        adv_img = img.clone().detach()
        
        prompt = Prompt(
            model=self.model,
            processor=self.processor,
            text_prompts=text_prompts,
            device=self.device
        )
        batch_targets = [self.targets] * batch_size
        
        for _ in range(num_iter):
            adv_img.requires_grad = True
            
            prompt.update_img_prompts([[adv_img]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()
            
            loss = -self.attack_loss(prompt, batch_targets)
            print("target_loss: %f" % (loss.item()))
            
            adv_grad = torch.autograd.grad(
                loss, adv_img, retain_graph=False, create_graph=False, allow_unused=True
            )[0]
            
            if adv_grad is None:
                # If gradient is None, it means adv_img was not used in computation
                # This can happen if image processing breaks the gradient chain
                # In this case, we'll use zero gradient (no update)
                adv_grad = torch.zeros_like(adv_img)
            
            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
            )
            grad = grad + momentum * decay
            momentum = grad
            
            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(img).detach().to(self.device)
            for _ in range(5):
                neighbor_images = adv_img.detach() + torch.randn_like(
                    img
                ).uniform_(-eps * beta, eps * beta)
                neighbor_images.requires_grad = True
                
                prompt.update_img_prompts([[neighbor_images]])
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_context_embs()
                
                loss = -self.attack_loss(prompt, batch_targets)
                
                GV_grad += torch.autograd.grad(
                    loss, neighbor_images, retain_graph=False, create_graph=False
                )[0]
            
            # obtaining the gradient variance
            v = GV_grad / 5 - adv_grad
            adv_img = adv_img.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_img - img, min=-eps, max=eps)
            adv_img = torch.clamp(img + delta, min=0, max=1).detach()
            
            self.model.zero_grad()
        
        return adv_img
