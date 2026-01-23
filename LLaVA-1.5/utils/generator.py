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
    
    def __init__(self, model, processor, max_new_tokens=1024, num_beams=1, 
                 min_length=1, top_p=1.0, repetition_penalty=1.0, 
                 length_penalty=1, temperature=1.0, device='cuda'):
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
            prompt: Prompt object with context_embs
        
        Returns:
            Generated text string
        """
        # Use the full LLaVA model for generation (not just language_model)
        # LLaVA's generate method handles both vision and language components
        outputs = self.model.generate(
            inputs_embeds=prompt.context_embs[0],
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=self.num_beams,
            do_sample=False,
            min_length=self.min_length,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
        )
        
        output_token = outputs[0]
        # Remove special tokens
        if len(output_token) > 0:
            if output_token[0] == 0:  # <unk>
                output_token = output_token[1:]
            if len(output_token) > 0 and output_token[0] == 1:  # <s>
                output_token = output_token[1:]
            if len(output_token) > 0 and output_token[0] == 29901:  # <s> alternative
                output_token = output_token[1:]
        
        output_text = self.processor.tokenizer.decode(output_token, add_special_tokens=False)
        return output_text
