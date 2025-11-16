# Generate summary report
summary = f"""
MILESTONE 2 SUMMARY REPORT
Text-to-Image Generation with Stable Diffusion
{'='*60}

1. MODEL ARCHITECTURE
   - Base Model: Stable Diffusion 1.5
   - Text Encoder: CLIP ViT-Large (768 dim)
   - Diffusion Model: UNet2DConditionModel
   - Scheduler: DDPM (Denoising Diffusion Probabilistic Model)
   - VAE: AutoencoderKL

2. DATASET
   - Source: Flickr30K (lmms-lab/flickr30k)
   - Subset: 20% test split = 6,357 images
   - Training pairs: {len(train_pairs):,} image-caption pairs
   - Validation pairs: {len(val_pairs):,} image-caption pairs
   - Caption length: 5-50 words
   - Image size: 256x256 pixels

3. TRAINING CONFIGURATION
   - Epochs: {NUM_EPOCHS}
   - Batch size: {BATCH_SIZE}
   - Learning rate: {LR}
   - Optimizer: AdamW
   - Classifier-free guidance probability: {GUIDANCE_PROB}
   - Device: CPU
   - Total training steps: {len(log_df)}

4. TRAINING RESULTS
   - Initial loss: {log_df['loss'].iloc[0]:.4f}
   - Final loss: {log_df['loss'].iloc[-1]:.4f}
   - Average loss: {log_df['loss'].mean():.4f}
   - Min loss: {log_df['loss'].min():.4f}
   - Loss reduction: {((log_df['loss'].iloc[0] - log_df['loss'].iloc[-1]) / log_df['loss'].iloc[0] * 100):.2f}%

5. GENERATION PARAMETERS
   - Inference steps: 50
   - Guidance scale: 7.5
   - Image resolution: 256x256
   - Latent size: 32x32x4

6. OBSERVATIONS

   a) Classifier-Free Guidance Impact:
      - Lower guidance (1.0-3.0): More creative but less prompt adherence
      - Medium guidance (5.0-7.5): Good balance between quality and prompt following
      - High guidance (10.0+): Strong prompt adherence but may oversaturate

   b) Inference Steps:
      - 10 steps: Fast but noisy/low quality
      - 20-30 steps: Reasonable quality-speed tradeoff
      - 50 steps: Best quality for baseline

   c) Model Performance:
      - Successfully generates images conditioned on text
      - Captures basic scene composition from captions
      - Struggles with fine details and multiple objects
      - Better performance on simple prompts vs complex scenes

   d) Challenges:
      - CPU training is extremely slow (~{len(log_df) // NUM_EPOCHS} steps/epoch)
      - Limited by small dataset subset (20% of Flickr30K)
      - Only 2 epochs may not be sufficient for convergence
      - Some generated images show artifacts or low coherence

7. NEXT STEPS (MILESTONE 3)
   - Increase training epochs (5-10)
   - Implement LoRA or fine-tuning on specific domain
   - Experiment with different text encoders
   - Add image quality metrics (FID, CLIP Score)
   - Test on more diverse prompts
   - Optimize inference speed

{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save summary
summary_path = os.path.join(SAMPLES_DIR, "milestone2_summary.txt")
with open(summary_path, 'w') as f:
    f.write(summary)

print(summary)
print(f"\nSummary saved to: {summary_path}")