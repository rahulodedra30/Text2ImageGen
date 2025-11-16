"""
Quick CPU Training Script 
For P100 GPU compatibility issues
"""

import os
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from datasets import load_dataset

# ============================================================================
# CONFIGURATION - CPU OPTIMIZED
# ============================================================================

class Config:
    device = torch.device("cpu")
    
    # Model
    model_name = "runwayml/stable-diffusion-v1-5"
    clip_model = "openai/clip-vit-large-patch14"
    
    # Training - REDUCED FOR SPEED
    batch_size = 4
    num_epochs = 2
    learning_rate = 2e-5
    gradient_accumulation_steps = 4
    
    # Diffusion
    guidance_prob = 0.15
    null_token = ""
    
    # Data - USE LESS DATA
    dataset_name = "lmms-lab/flickr30k"
    dataset_split = "test[:5%]"  # Only 5% for quick training
    img_size = 256
    min_caption_len = 5
    max_caption_len = 50
    max_train_samples = 1000  # Limit training samples
    
    # Paths
    checkpoint_dir = "checkpoints"
    samples_dir = "milestone2_samples"
    log_file = "train_log.csv"
    
    # Generation
    num_inference_steps = 30  # Reduced for speed
    guidance_scale = 7.5
    
    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

config = Config()

print("="*60)
print("QUICK CPU TRAINING - REDUCED DATASET")
print("="*60)
print(f"Device: {config.device}")
print(f"Dataset: {config.dataset_split}")
print(f"Max samples: {config.max_train_samples}")
print("="*60)

# ============================================================================
# DATASET
# ============================================================================

def clean_caption(caption):
    caption = caption.lower().strip()
    caption = re.sub(r'[^a-z0-9\s.,!?-]', '', caption)
    return ' '.join(caption.split())

class ImageCaptionDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = self.transform(pair['image']) if self.transform else pair['image']
        return {'image': image, 'caption': pair['caption']}

def load_data():
    print("\nLoading dataset...")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    print(f"Total examples: {len(dataset)}")
    
    # Create pairs
    pairs = []
    for example in tqdm(dataset, desc="Processing dataset"):
        image = example["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        for caption in example["caption"]:
            cleaned = clean_caption(caption)
            if config.min_caption_len <= len(cleaned.split()) <= config.max_caption_len:
                pairs.append({'image': image, 'caption': cleaned})
                if len(pairs) >= config.max_train_samples:
                    break
        if len(pairs) >= config.max_train_samples:
            break
    
    print(f"Using {len(pairs)} pairs for training")
    
    # Train/val split
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Create datasets
    image_transforms = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_dataset = ImageCaptionDataset(train_pairs, image_transforms)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    return train_loader, train_pairs, val_pairs

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    print("\nLoading models...")
    
    # Text Encoder
    print("  Loading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model)
    text_encoder = CLIPTextModel.from_pretrained(
        config.clip_model,
        torch_dtype=torch.float32
    ).to(config.device)
    text_encoder.eval()
    
    # VAE
    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        config.model_name,
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(config.device)
    vae.eval()
    
    # UNet
    print("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        config.model_name,
        subfolder="unet",
        torch_dtype=torch.float32
    ).to(config.device)
    unet.train()
    
    # Scheduler
    print("  Loading Scheduler...")
    scheduler = DDPMScheduler.from_pretrained(
        config.model_name,
        subfolder="scheduler"
    )
    
    print(f"  ✓ All models loaded")
    return tokenizer, text_encoder, vae, unet, scheduler

def get_text_embeddings(captions, tokenizer, text_encoder):
    tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    tokens = {k: v.to(config.device) for k, v in tokens.items()}
    
    with torch.no_grad():
        outputs = text_encoder(
            tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )
    return outputs.last_hidden_state

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(epoch, train_loader, unet, vae, text_encoder, tokenizer, scheduler, optimizer):
    unet.train()
    epoch_losses = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for step, batch in enumerate(progress_bar):
        images = batch['image'].to(config.device)
        captions = batch['caption']
        batch_size = images.size(0)
        
        # Encode images
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, 
            scheduler.config.num_train_timesteps,
            (batch_size,),
            device=config.device
        )
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        text_embeds = get_text_embeddings(captions, tokenizer, text_encoder)
        
        # Classifier-free guidance
        if torch.rand(1).item() < config.guidance_prob:
            uncond_embeds = get_text_embeddings(
                [config.null_token] * batch_size,
                tokenizer,
                text_encoder
            )
            
            model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
            timesteps_input = torch.cat([timesteps, timesteps], dim=0)
            text_embeds_input = torch.cat([uncond_embeds, text_embeds], dim=0)
        else:
            model_input = noisy_latents
            timesteps_input = timesteps
            text_embeds_input = text_embeds
        
        # Predict noise
        noise_pred = unet(
            sample=model_input,
            timestep=timesteps_input,
            encoder_hidden_states=text_embeds_input,
            return_dict=False
        )[0]
        
        # Apply guidance
        if noise_pred.shape[0] == 2 * batch_size:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        loss = loss / config.gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Track loss
        epoch_losses.append(loss.item() * config.gradient_accumulation_steps)
        progress_bar.set_postfix({"loss": f"{epoch_losses[-1]:.4f}"})
    
    return epoch_losses

def train():
    # Load data
    train_loader, train_pairs, val_pairs = load_data()
    
    # Load models
    tokenizer, text_encoder, vae, unet, scheduler = load_models()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    all_losses = []
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        epoch_losses = train_epoch(
            epoch, train_loader, unet, vae, text_encoder, 
            tokenizer, scheduler, optimizer
        )
        
        # Save losses
        for step, loss in enumerate(epoch_losses):
            all_losses.append({
                'epoch': epoch,
                'step': step,
                'loss': loss
            })
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config.checkpoint_dir, 
            f"unet_epoch_{epoch+1}.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': sum(epoch_losses) / len(epoch_losses),
        }, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Save training log
    log_df = pd.DataFrame(all_losses)
    log_df.to_csv(config.log_file, index=False)
    print(f"\n✓ Training log saved: {config.log_file}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return unet, vae, text_encoder, tokenizer, scheduler, log_df

if __name__ == "__main__":
    train()
    print("\n✓ training completed!")
    print("✓ Now run: python generate_and_evaluate.py")