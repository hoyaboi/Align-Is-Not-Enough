"""
Prompt wrapper for LLaVA model
Adapted from MiniGPT-v2 prompt_wrapper for LLaVA compatibility
"""
import torch

DEFAULT_IMAGE_TOKEN = '<image>'

class Prompt:
    """
    Prompt wrapper for LLaVA model.
    Handles text and image embeddings for multimodal inputs.
    """
    
    def __init__(self, model, processor, text_prompts=None, img_prompts=None, 
                 control_slice_list=None, control_embeds=None, device='cuda', 
                 max_new_tokens=1024, max_length=2000):
        self.model = model
        self.processor = processor
        self.device = device
        
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        
        self.text_prompts = text_prompts
        self.img_prompts = img_prompts
        self.control_slice_list = control_slice_list
        self.control_embeds = control_embeds
        
        self.text_embs = []
        self.img_embs = []
        self.context_embs = []
        
        self.text_embs = self.generate_text_embedding(self.text_prompts)
        self.img_embs = self.generate_img_embedding(self.img_prompts)
        self.update_context_embs()
    
    def update_context_embs(self):
        """Update context embeddings by combining text and image embeddings."""
        if len(self.text_embs) == len(self.img_embs):
            self.context_embs = self.generate_context_embedding(
                self.text_embs, self.img_embs
            )
        else:
            if len(self.text_embs) == 1 and len(self.img_embs) == 0:
                self.context_embs = self.text_embs[0]
            else:
                self.context_embs = []
    
    def update_text_prompt(self, text_prompts):
        self.text_prompts = text_prompts
        self.text_embs = self.generate_text_embedding(self.text_prompts)
        self.update_context_embs()
    
    def update_img_prompts(self, img_prompts):
        self.img_prompts = img_prompts
        self.img_embs = self.generate_img_embedding(self.img_prompts)
        self.update_context_embs()
    
    def generate_text_embedding(self, text_prompts):
        """Generate text embeddings from prompts."""
        if text_prompts is None:
            return []
        
        text_embs = []
        for i, item in enumerate(text_prompts):
            # LLaVA uses <image> token instead of <ImageHere>
            prompt_segs = item.split('<image>') if '<image>' in item else item.split('<ImageHere>')
            
            seg_tokens = []
            for seg_idx, seg in enumerate(prompt_segs):
                tokens = self.processor.tokenizer(
                    seg, return_tensors="pt", add_special_tokens=(seg_idx == 0)
                ).to(self.device).input_ids
                seg_tokens.append(tokens)
            
            embs = []
            for j, seg_t in enumerate(seg_tokens):
                if self.control_slice_list is not None and j > 0:
                    # Apply control embeddings if specified
                    start = self.control_slice_list[i].start - seg_tokens[0].shape[1] - 2
                    stop = self.control_slice_list[i].stop - seg_tokens[0].shape[1] - 2
                    tmp = self._get_embedding_matrix()(seg_t)
                    embedding = torch.cat([
                        tmp[:, :start, :], 
                        self.control_embeds, 
                        tmp[:, stop:, :]
                    ], dim=1)
                    embs.append(embedding)
                else:
                    embs.append(self._get_embedding_matrix()(seg_t))
            
            text_embs.append(embs)
        
        return text_embs
    
    def generate_img_embedding(self, img_prompts):
        """Generate image embeddings from image prompts."""
        if img_prompts is None:
            return []
        
        img_embs = []
        for items in img_prompts:
            embs = []
            for img in items:
                # For gradient computation, we need to process images as tensors directly
                # without PIL conversion which breaks the gradient chain
                if isinstance(img, torch.Tensor):
                    # Ensure image is in correct format [1, C, H, W]
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    
                    # Denormalize: convert from normalized [0,1] to [0,1] range
                    # Image is already in [0,1] range after denormalization
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)
                    img_denorm = img * std[None, :, None, None] + mean[None, :, None, None]
                    img_denorm = torch.clamp(img_denorm, 0, 1)
                    
                    # Convert to pixel_values format directly (without PIL)
                    # LLaVA-1.5 uses 336x336 image size
                    # We'll use torchvision transforms to replicate this
                    from torchvision import transforms
                    # Resize to 336x336 (keep gradient)
                    resize_transform = transforms.Resize((336, 336), antialias=True)
                    img_resized = resize_transform(img_denorm)
                    
                    # Normalize with ImageNet stats (keep gradient)
                    normalize_transform = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                    pixel_values = normalize_transform(img_resized).to(self.device)
                else:
                    # PIL image: use processor (gradient will be broken, but this is for non-tensor inputs)
                    if hasattr(self.processor, 'image_processor'):
                        image_processor = self.processor.image_processor
                        pixel_values = image_processor(img, return_tensors="pt")['pixel_values'].to(self.device)
                    else:
                        inputs = self.processor(images=img, text="dummy", return_tensors="pt").to(self.device)
                        pixel_values = inputs['pixel_values']
                
                # Remove torch.no_grad() to enable gradient computation
                # Get vision features from vision tower
                if hasattr(self.model, 'get_model'):
                    vision_tower = self.model.get_model().get_vision_tower()
                    if vision_tower is None:
                        vision_tower = self.model.get_model().vision_tower
                elif hasattr(self.model, 'vision_tower'):
                    vision_tower = self.model.vision_tower
                else:
                    raise AttributeError("Cannot find vision_tower in LLaVA model")
                
                # Get vision tower output (with gradient enabled)
                vision_output = vision_tower(pixel_values)
                if hasattr(vision_output, 'last_hidden_state'):
                    vision_features = vision_output.last_hidden_state
                elif isinstance(vision_output, torch.Tensor):
                    vision_features = vision_output
                else:
                    raise ValueError(f"Unexpected vision_tower output type: {type(vision_output)}")
                
                # Project vision features to language model dimension using multi_modal_projector
                if hasattr(self.model, 'get_model'):
                    projector = self.model.get_model().multi_modal_projector
                elif hasattr(self.model, 'multi_modal_projector'):
                    projector = self.model.multi_modal_projector
                else:
                    raise AttributeError("Cannot find multi_modal_projector in LLaVA model")
                
                # Project vision features to match text embedding dimension (with gradient enabled)
                image_features = projector(vision_features)
                embs.append(image_features)
            img_embs.append(embs)
        
        return img_embs
    
    def generate_context_embedding(self, batch_text_embs, batch_img_embs):
        """Combine text and image embeddings into context embeddings."""
        assert len(batch_text_embs) == len(batch_img_embs), \
            f"Unmatched batch size: {len(batch_text_embs)} != {len(batch_img_embs)}"
        
        batch_size = len(batch_text_embs)
        batch_context_embs = []
        
        for i in range(batch_size):
            text_embs = batch_text_embs[i]
            img_embs = batch_img_embs[i]
            
            num_text_segs = len(text_embs)
            num_img_segs = len(img_embs)
            
            if num_text_segs == 0 and num_img_segs == 0:
                mixed_embs = [torch.zeros([1, 0, 0])]
            elif num_text_segs == 0:
                mixed_embs = img_embs
            elif num_img_segs == 0:
                mixed_embs = text_embs
            else:
                # Interleave text and image embeddings
                s = t = 0
                mixed_embs = []
                while s < num_text_segs and t < num_img_segs:
                    mixed_embs.append(text_embs[s])
                    mixed_embs.append(img_embs[t])
                    s, t = s + 1, t + 1
                if s < num_text_segs:
                    mixed_embs += text_embs[s:]
                if t < num_img_segs:
                    mixed_embs += img_embs[t:]
            
            mixed_embs = torch.cat(mixed_embs, dim=1)
            
            current_max_len = mixed_embs.shape[1] + self.max_new_tokens
            if current_max_len - self.max_length > 0:
                print('Warning: The number of tokens exceeds max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - self.max_length)
            mixed_embs = mixed_embs[:, begin_idx:]
            
            batch_context_embs.append(mixed_embs.requires_grad_(True))
        
        return batch_context_embs
    
    def _get_embedding_matrix(self):
        """Get embedding matrix from LLaVA model."""
        if hasattr(self.model, 'get_model'):
            language_model = self.model.get_model()
        elif hasattr(self.model, 'language_model'):
            language_model = self.model.language_model
        else:
            language_model = self.model
        
        if hasattr(language_model, 'embed_tokens'):
            return language_model.embed_tokens
        elif hasattr(language_model, 'model') and hasattr(language_model.model, 'embed_tokens'):
            return language_model.model.embed_tokens
        else:
            return self.model.get_input_embeddings()
