import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    device = torch.device("cpu")
    
    model_name = "runwayml/stable-diffusion-v1-5"
    clip_model = "openai/clip-vit-large-patch14"
    checkpoint_path = "checkpoints/unet_epoch_4.pt"
    
    num_inference_steps = 30 
    guidance_scale = 7.5
    img_size = 256
    
    samples_dir = "milestone2_samples"
    log_file = "train_log.csv"
    
    def __init__(self):
        os.makedirs(self.samples_dir, exist_ok=True)

config = Config()

print("="*60)
print("IMAGE GENERATION & EVALUATION")
print("="*60)
print(f"Device: {config.device}")
print(f"Checkpoint: {config.checkpoint_path}")
print("="*60)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load trained models"""
    print("\nLoading models...")
    
    # Text Encoder
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model)
    text_encoder = CLIPTextModel.from_pretrained(config.clip_model).to(config.device)
    text_encoder.eval()
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        config.model_name,
        subfolder="vae"
    ).to(config.device)
    vae.eval()
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        config.model_name,
        subfolder="unet"
    ).to(config.device)
    
    # Load checkpoint
    if os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    else:
        print(f"    Checkpoint not found: {config.checkpoint_path}")
        print("  Using base SD 1.5 weights instead")
    
    unet.eval()
    
    # Scheduler
    scheduler = DDPMScheduler.from_pretrained(
        config.model_name,
        subfolder="scheduler"
    )
    
    print("   All models loaded")
    return tokenizer, text_encoder, vae, unet, scheduler

def get_text_embeddings(captions, tokenizer, text_encoder):
    """Get CLIP text embeddings"""
    tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    tokens = {k: v.to(config.device) for k, v in tokens.items()}
    
    with torch.no_grad():
        outputs = text_encoder(tokens['input_ids'], attention_mask=tokens['attention_mask'])
    return outputs.last_hidden_state

# ============================================================================
# IMAGE GENERATION
# ============================================================================

@torch.no_grad()
def generate_images(prompts, tokenizer, text_encoder, vae, unet, scheduler,
                   num_inference_steps=50, guidance_scale=7.5):
    """Generate images from text prompts"""
    generated_images = []
    
    for prompt in tqdm(prompts, desc="Generating images"):
        # Encode prompt
        text_embeds = get_text_embeddings([prompt], tokenizer, text_encoder)
        uncond_embeds = get_text_embeddings([""], tokenizer, text_encoder)
        
        # Random noise
        latents = torch.randn((1, 4, 32, 32), device=config.device)
        
        scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents, latents])
            timestep = torch.tensor([t, t], device=config.device)
            text_input = torch.cat([uncond_embeds, text_embeds])
            
            # Predict noise
            noise_pred = unet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_input,
                return_dict=False
            )[0]
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        generated_images.append(pil_image)
    
    return generated_images

# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================

def generate_baseline_samples(models):
    """Generate baseline samples"""
    tokenizer, text_encoder, vae, unet, scheduler = models
    
    test_prompts = [
        "a dog playing in the park",
        "a cat sitting on a windowsill",
        "people walking on a city street",
        "a child playing with toys",
        "a person riding a bicycle",
        "two friends having a conversation",
        "a man wearing a hat",
        "a woman in a red dress",
        "children playing soccer",
        "a group of people at a beach"
    ]
    
    print("\n" + "="*60)
    print("GENERATING BASELINE SAMPLES")
    print("="*60)
    
    generated_images = generate_images(
        test_prompts, tokenizer, text_encoder, vae, unet, scheduler,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale
    )
    
    # Save individual images
    for i, (img, prompt) in enumerate(zip(generated_images, test_prompts)):
        img_path = os.path.join(config.samples_dir, f"sample_{i+1}.png")
        img.save(img_path)
        print(f"   Saved: sample_{i+1}.png - '{prompt}'")
    
    # Create grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (img, prompt) in enumerate(zip(generated_images, test_prompts)):
        axes[i].imshow(img)
        axes[i].set_title(f"{i+1}. {prompt}", fontsize=10, wrap=True)
        axes[i].axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(config.samples_dir, "generated_samples_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Grid saved: {grid_path}")
    
    return generated_images, test_prompts

def test_guidance_scales(models, prompt="a dog playing in the park"):
    """Test different guidance scales"""
    tokenizer, text_encoder, vae, unet, scheduler = models
    
    print("\n" + "="*60)
    print("TESTING GUIDANCE SCALES")
    print("="*60)
    
    guidance_scales = [1.0, 3.0, 5.0, 7.5, 10.0]
    
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(20, 4))
    
    for i, scale in enumerate(guidance_scales):
        print(f"  Generating with guidance_scale={scale}...")
        img = generate_images(
            [prompt], tokenizer, text_encoder, vae, unet, scheduler,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=scale
        )[0]
        
        axes[i].imshow(img)
        axes[i].set_title(f"Scale: {scale}", fontsize=12)
        axes[i].axis('off')
        
        img.save(os.path.join(config.samples_dir, f"guidance_scale_{scale}.png"))
    
    plt.tight_layout()
    comp_path = os.path.join(config.samples_dir, "guidance_comparison.png")
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"   Saved: {comp_path}")

def test_inference_steps(models, prompt="a cat sitting on a windowsill"):
    """Test different inference steps"""
    tokenizer, text_encoder, vae, unet, scheduler = models
    
    print("\n" + "="*60)
    print("TESTING INFERENCE STEPS")
    print("="*60)
    
    inference_steps = [10, 20, 30, 50]
    
    fig, axes = plt.subplots(1, len(inference_steps), figsize=(16, 4))
    
    for i, steps in enumerate(inference_steps):
        print(f"  Generating with {steps} steps...")
        img = generate_images(
            [prompt], tokenizer, text_encoder, vae, unet, scheduler,
            num_inference_steps=steps,
            guidance_scale=config.guidance_scale
        )[0]
        
        axes[i].imshow(img)
        axes[i].set_title(f"{steps} Steps", fontsize=12)
        axes[i].axis('off')
        
        img.save(os.path.join(config.samples_dir, f"steps_{steps}.png"))
    
    plt.tight_layout()
    comp_path = os.path.join(config.samples_dir, "inference_steps_comparison.png")
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"   Saved: {comp_path}")

def plot_training_loss():
    """Plot training loss curves"""
    if not os.path.exists(config.log_file):
        print(f"\n  Training log not found: {config.log_file}")
        return
    
    print("\n" + "="*60)
    print("PLOTTING TRAINING LOSS")
    print("="*60)
    
    log_df = pd.read_csv(config.log_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss over all steps
    axes[0].plot(log_df['step'], log_df['loss'], alpha=0.6)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Loss by epoch
    for epoch in log_df['epoch'].unique():
        epoch_data = log_df[log_df['epoch'] == epoch]
        axes[1].plot(epoch_data['step'], epoch_data['loss'], 
                    label=f'Epoch {epoch}', alpha=0.7)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss by Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_path = os.path.join(config.samples_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"   Saved: {loss_path}")
    
    # Print statistics
    print("\n  Training Statistics:")
    print(f"    Total steps: {len(log_df)}")
    print(f"    Initial loss: {log_df['loss'].iloc[0]:.4f}")
    print(f"    Final loss: {log_df['loss'].iloc[-1]:.4f}")
    print(f"    Average loss: {log_df['loss'].mean():.4f}")
    print(f"    Min loss: {log_df['loss'].min():.4f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete evaluation pipeline"""
    # Load models
    models = load_models()
    
    generate_baseline_samples(models)
    test_guidance_scales(models)
    
    test_inference_steps(models)
    
    plot_training_loss()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()