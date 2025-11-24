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
import multiprocessing as mp

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from datasets import load_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    device = torch.device("cpu")
    
    num_threads = mp.cpu_count()  
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    
    # Model
    model_name = "runwayml/stable-diffusion-v1-5"
    clip_model = "openai/clip-vit-large-patch14"
    
    batch_size = 8  
    num_epochs = 6
    learning_rate = 2e-5
    gradient_accumulation_steps = 2 
    
    enable_memory_efficient_attention = True
    use_tf32 = True  
    gradient_checkpointing = True 
    
    # Diffusion
    guidance_prob = 0.15
    null_token = ""
    
    # Data loading
    dataset_name = "lmms-lab/flickr30k"
    dataset_split = "test[:25%]"
    img_size = 256
    min_caption_len = 5
    max_caption_len = 50
    max_train_samples = 5000
    num_workers = min(4, mp.cpu_count()) 
    prefetch_factor = 2
    persistent_workers = True
    
    # Paths
    checkpoint_dir = "checkpoints"
    samples_dir = "milestone2_samples"
    log_file = "train_log.csv"
    
    # Checkpoint strategy
    save_every_n_epochs = 2 
    keep_last_n_checkpoints = 3 

    # Generation
    num_inference_steps = 30
    guidance_scale = 7.5
    
    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

config = Config()

# optimizations
torch.backends.cudnn.benchmark = False 
if hasattr(torch.backends, 'mkldnn'):
    torch.backends.mkldnn.enabled = True 

print("="*60)
print("OPTIMIZED CPU TRAINING")
print("="*60)
print(f"Device: {config.device}")
print(f"CPU Threads: {config.num_threads}")
print(f"Workers: {config.num_workers}")
print(f"Batch Size: {config.batch_size}")
print(f"Dataset: {config.dataset_split}")
print(f"Max samples: {config.max_train_samples}")
print(f"Gradient Checkpointing: {config.gradient_checkpointing}")
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
        print("  Pre-processing images...")
        for pair in tqdm(self.pairs, desc="  Converting to RGB"):
            if pair['image'].mode != 'RGB':
                pair['image'] = pair['image'].convert('RGB')
    
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
    
    pairs = []
    print("Processing captions...")
    for example in tqdm(dataset, desc="Creating pairs"):
        image = example["image"]
        
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
    
    # Optimized transforms
    image_transforms = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size), 
                         interpolation=transforms.InterpolationMode.BILINEAR), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_dataset = ImageCaptionDataset(train_pairs, image_transforms)
    
    # Optimized DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False, 
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=True
    )
    
    return train_loader, train_pairs, val_pairs

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    print("\nLoading models...")
    
    print("  Loading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model)
    text_encoder = CLIPTextModel.from_pretrained(
        config.clip_model,
        torch_dtype=torch.float32
    ).to(config.device)
    text_encoder.eval()
    
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    # VAE
    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        config.model_name,
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(config.device)
    vae.eval()
    
    for param in vae.parameters():
        param.requires_grad = False
    
    # UNet
    print("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        config.model_name,
        subfolder="unet",
        torch_dtype=torch.float32
    ).to(config.device)
    
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("   Gradient checkpointing enabled")
    
    if config.enable_memory_efficient_attention:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("   Memory efficient attention enabled")
        except:
            print("   Memory efficient attention not available")
    
    unet.train()
    
    # Scheduler
    print("  Loading Scheduler...")
    scheduler = DDPMScheduler.from_pretrained(
        config.model_name,
        subfolder="scheduler"
    )
    
    print(f"   All models loaded and optimized")
    return tokenizer, text_encoder, vae, unet, scheduler

text_embedding_cache = {}

def get_text_embeddings(captions, tokenizer, text_encoder, use_cache=True):
    if use_cache:
        cache_key = tuple(captions)
        if cache_key in text_embedding_cache:
            return text_embedding_cache[cache_key]
    
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
    
    result = outputs.last_hidden_state
    
    if use_cache and len(text_embedding_cache) < 1000: 
        text_embedding_cache[cache_key] = result
    
    return result

# ============================================================================
# OPTIMIZED TRAINING
# ============================================================================

def train_epoch(epoch, train_loader, unet, vae, text_encoder, tokenizer, scheduler, optimizer):
    """Optimized training loop"""
    unet.train()
    epoch_losses = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    optimizer.zero_grad() 
    
    for step, batch in enumerate(progress_bar):
        images = batch['image'].to(config.device, non_blocking=True)
        captions = batch['caption']
        batch_size = images.size(0)
        
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
        
        # Text embeddings
        text_embeds = get_text_embeddings(captions, tokenizer, text_encoder, use_cache=False)
        
        # Classifier-free guidance
        if torch.rand(1).item() < config.guidance_prob:
            uncond_embeds = get_text_embeddings(
                [config.null_token] * batch_size,
                tokenizer,
                text_encoder,
                use_cache=True  
            )
            
            model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
            timesteps_input = torch.cat([timesteps, timesteps], dim=0)
            text_embeds_input = torch.cat([uncond_embeds, text_embeds], dim=0)
        else:
            model_input = noisy_latents
            timesteps_input = timesteps
            text_embeds_input = text_embeds
        
        # Forward pass
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
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_losses.append(loss.item() * config.gradient_accumulation_steps)
        
        avg_loss = sum(epoch_losses[-100:]) / min(len(epoch_losses), 100)
        progress_bar.set_postfix({
            "loss": f"{epoch_losses[-1]:.4f}",
            "avg_loss": f"{avg_loss:.4f}"
        })
    
    return epoch_losses

def train():
    # Load data
    train_loader, train_pairs, val_pairs = load_data()
    tokenizer, text_encoder, vae, unet, scheduler = load_models()
    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler for better convergence
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    print("\n" + "="*60)
    print("STARTING OPTIMIZED TRAINING")
    print("="*60)
    print(f"Total training steps per epoch: {len(train_loader)}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print("="*60)
    
    all_losses = []
    start_time = datetime.now()
    
    for epoch in range(config.num_epochs):
        epoch_start = datetime.now()
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
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
        
        # Calculate epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time/60:.2f} minutes")
        print(f"  Steps/sec: {len(epoch_losses)/epoch_time:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0 or epoch == config.num_epochs - 1:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f"unet_epoch_{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_lr.state_dict(),
                'loss': avg_loss,
                'config': vars(config)
            }, checkpoint_path)
            print(f"   Checkpoint saved: {checkpoint_path}")
                    
        scheduler_lr.step()
        
        if (epoch + 1) % 2 == 0:
            log_df = pd.DataFrame(all_losses)
            log_df.to_csv(config.log_file, index=False)
    
    log_df = pd.DataFrame(all_losses)
    log_df.to_csv(config.log_file, index=False)
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n Training log saved: {config.log_file}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Average time per epoch: {total_time/config.num_epochs/60:.2f} minutes")
    print("="*60)
    
    return unet, vae, text_encoder, tokenizer, scheduler, log_df

if __name__ == "__main__":
    train()
    print("\n training completed!")
    print(" Now run: python generate_and_evaluate.py")