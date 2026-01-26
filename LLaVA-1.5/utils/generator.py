"""
Text generator for LLaVA model
"""
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Generator:
    """Text generator for LLaVA model."""
    
    def __init__(self, model, processor, max_new_tokens=300, num_beams=1, 
                 min_length=1, top_p=0.9, repetition_penalty=1.05, 
                 length_penalty=1, temperature=1.0, device='cuda', do_sample=True):
        self.model = model
        self.processor = processor
        self.device = device
        
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.min_length = min_length
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.do_sample = do_sample
        
        # LLaVA stop tokens
        stop_words_ids = [
            torch.tensor([835]).to(self.device),  # '###'
            torch.tensor([2277, 29937]).to(self.device)
        ]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    def generate(self, prompt):
        """
        Generate text from prompt.
        
        Args:
            prompt: Prompt object with context_embs, text_prompts, and img_prompts
        
        Returns:
            Generated text string
        """
        # For LLaVA, we need to use the processor with both image and text
        # Get the original image from img_prompts (before embedding conversion)
        if hasattr(prompt, 'img_prompts') and len(prompt.img_prompts) > 0 and len(prompt.img_prompts[0]) > 0:
            # Get the original image tensor (should be in [0, 1] range)
            img = prompt.img_prompts[0][0]
            
            # Ensure image is in correct format and range
            if isinstance(img, torch.Tensor):
                # Clamp to [0, 1] range if needed
                if img.max() > 1.0 or img.min() < 0.0:
                    img = torch.clamp(img, 0, 1)
                # Convert to PIL if needed for processor
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                if img.dim() == 4:
                    img = img[0]  # Remove batch dimension
                if img.dim() == 3:
                    img_pil = to_pil(img.cpu())
                else:
                    img_pil = img
            else:
                img_pil = img
        else:
            img_pil = None
        
        # Get text prompt
        if hasattr(prompt, 'text_prompts') and len(prompt.text_prompts) > 0:
            text = prompt.text_prompts[0]
        else:
            text = ""
        
        # Clean up prompt format for LLaVA processor
        # Remove Llama-2 style tokens if present and convert to LLaVA format
        if '<s>' in text:
            text = text.replace('<s>', '').strip()
        if '[INST]' in text:
            # Extract content between [INST] and [/INST]
            if '[/INST]' in text:
                inst_start = text.find('[INST]')
                inst_end = text.find('[/INST]')
                if inst_start != -1 and inst_end != -1:
                    text = text[inst_start + 6:inst_end].strip()  # 6 is len('[INST]')
            else:
                text = text.replace('[INST]', '').strip()
        if '[/INST]' in text:
            text = text.replace('[/INST]', '').strip()
        
        # Ensure <image> token is present if we have an image
        if img_pil is not None and '<image>' not in text:
            # If text doesn't start with <image>, prepend it
            if text.strip():
                text = f"<image>\n{text}"
            else:
                text = "<image>"
        
        # Use processor to prepare inputs
        if img_pil is not None:
            inputs = self.processor(text=text, images=img_pil, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Generate using the model
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=self.num_beams,
            do_sample=self.do_sample,  # Match MiniGPT-v2 Chat behavior (default: True)
            min_length=self.min_length,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
        )
        
        # Decode output (skip input tokens)
        input_length = inputs['input_ids'].shape[1]
        output_token = outputs[0][input_length:]
        
        output_text = self.processor.tokenizer.decode(output_token, add_special_tokens=False)
        
        # Clean up output text
        output_text = output_text.strip()
        if output_text == "" or output_text == "</s>":
            # Try decoding full sequence and extract only new tokens
            full_text = self.processor.tokenizer.decode(outputs[0], add_special_tokens=False)
            input_text = self.processor.tokenizer.decode(inputs['input_ids'][0], add_special_tokens=False)
            if full_text.startswith(input_text):
                output_text = full_text[len(input_text):].strip()
            else:
                output_text = full_text
        
        return output_text
