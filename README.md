## Text2ImageGen: Text-to-Image Generation Using Transformer-Based Embeddings and Diffusion Models

- Text2ImageGen is a deep learning project that explores how language and vision models can work together to generate images from natural language descriptions.
- Built as part of IE 7615 – Deep Learning for AI, this project integrates a Transformer-based text encoder (CLIP) with a Diffusion-based image generator (Stable Diffusion/DDPM) to create visually coherent and semantically aligned images from text prompts.
- The work emphasizes multimodal learning, conditional generative modeling, and evaluation of model performance using both quantitative metrics (FID, Inception Score) and qualitative visual analysis.

---

Implementation of text-to-image generation using Stable Diffusion 1.5 with CLIP text encoder, trained on Flickr30K dataset with classifier-free guidance.

---

## Repository Structure

```
TEXT2IMAGEGEN/
├── logs/
│   ├── generate_and_evaluate_logs.txt  # Generation logs
│   └── training_logs.txt               # Training logs
├── notebooks/
│   └── experimentation.ipynb           # Experimental notebook
├── scripts/
│   ├── checkpoints/
│   │   └── unet_epoch_4.pt            # Trained model checkpoint
│   ├── milestone2_samples/             # 📁 GENERATED IMAGES & RESULTS
│   │   ├── sample_1.png to sample_10.png      # Baseline generations
│   │   ├── generated_samples_grid.png         # 10-image grid
│   │   ├── guidance_comparison.png            # CFG scale analysis
│   │   ├── inference_steps_comparison.png     # Steps analysis
│   │   ├── training_loss.png                  # Loss curves
│   │   └── milestone2_summary.txt             # Auto-generated summary
│   ├── generate_and_evaluate.py        # Generation & evaluation script
│   ├── train_text2img.py               # Training script (CPU optimized)
│   └── train_log.csv                   # Complete training logs (CSV)
├── .gitignore                          # Git ignore file
├── milestone2.md                       # 📄 1-PAGE SUMMARY (this file)
├── README.md                           # 📄 PROJECT README (this file)
└── requirements.txt                    # Python dependencies
```

---

## Milestone 2 Deliverables

###  1. Text Encoder Integration
- **Model**: CLIP ViT-Large-Patch14 (768-dimensional embeddings)
- **Integration**: Cross-attention layers in UNet with text conditioning
- **Status**: Successfully implemented and frozen during training

###  2. Baseline Conditional Generation
- **Generated**: 10 diverse text-to-image samples
- **Location**: `scripts/milestone2_samples/sample_*.png`
- **Prompts**: Dogs, cats, people, children, various scenes
- **Quality**: Basic scene composition achieved, recognizable subjects

###  3. Classifier-Free Guidance Tuning
- **Tested scales**: 1.0, 3.0, 5.0, 7.5, 10.0
- **Results**: Scale 7.5 optimal for current model
- **Visualization**: `scripts/milestone2_samples/guidance_comparison.png`

###  4. Noise Schedule Experimentation
- **Tested steps**: 10, 20, 30, 50
- **Scheduler**: DDPM with 1000 training timesteps
- **Finding**: 30-50 steps optimal quality-speed tradeoff
- **Visualization**: `scripts/milestone2_samples/inference_steps_comparison.png`

###  5. Training Documentation
- **Training log**: `train_log.csv` (2000 steps, 4 epochs)
- **Loss curves**: `scripts/milestone2_samples/training_loss.png`
- **Checkpoints**: `checkpoints/unet_epoch_*.pt`

---


## Technical Specifications

### Model Architecture
- **Base Model**: Stable Diffusion 1.5
- **Text Encoder**: CLIP ViT-Large-Patch14 (frozen, 768-dim output)
- **UNet**: 2D Conditional with cross-attention
- **VAE**: AutoencoderKL (frozen, 4-channel latent)
- **Scheduler**: DDPM (1000 training timesteps)

### Training Configuration
- **Dataset**: Flickr30K (5% subset, ~2000 training pairs)
- **Epochs**: 4
- **Batch Size**: 8 (effective: 16 with gradient accumulation)
- **Learning Rate**: 2e-5 with cosine annealing
- **Device**: CPU (P100 GPU compatibility issues)
- **Training Time**: ~7 hours total
- **Image Resolution**: 256×256

### Optimization Techniques
- Multi-threaded CPU processing (all cores)
- Gradient accumulation (factor: 2)
- Frozen VAE and text encoder
- Persistent data workers
- Classifier-free guidance (15% dropout probability)

---

## Results Summary

### Training Metrics
- **Initial Loss**: 0.244
- **Final Loss**: 0.223 (8.6% improvement)
- **Average Loss**: 0.184
- **Minimum Loss**: 0.017

### Generation Quality
- Current quality: Recognizable subjects, basic composition
- Issues: Low resolution, anatomical inaccuracies, artifacts
- Cause: Limited training (4 epochs, 2000 samples, 256×256)

### Classifier-Free Guidance
- Optimal scale: **7.5** for current model
- Lower (1.0-3.0): Creative but poor adherence
- Higher (10.0+): Strong adherence but oversaturated

### Inference Steps
- Optimal: **30-50 steps** for quality-speed balance
- 10 steps: Too noisy
- 50 steps: Best quality but 4× slower than 30 steps

---
# Milestone 3 

### Milestone 3 Goals

- Generate a consistent evaluation dataset  
- Compute **FID** and **Inception Score**  
- Compare real vs generated images  
- Fully analyze the notebook workflow  
- Document model performance limitations  
- Plan improvements for the final milestone  

---

## 🧪 1. Quantitative Evaluation

### **Computed Metrics (100 generated vs Flickr30k images)**

| Metric | Value |
|--------|--------|
| **FID** | ~380 |
| **Inception Score** | ~3 |

### Interpretation
- **High FID** → Generated images differ greatly from real images  
- **Low IS** → Low diversity, weak semantic meaning  
- Expected due to:  
  - Small dataset (2,000 samples)  
  - Only 4 epochs  
  - CPU-only training  
  - 256×256 resolution  

---

### 🖼 2. Qualitative Evaluation

### **Real vs Generated Image Comparison**  

This figure compares real Flickr30k images with images generated from the same captions.  
The real images show clean composition and clear subjects.  
Generated images show distortion, poor object boundaries, and noise — matching the FID/IS scores.

---

## 📈 3. Milestone 3 Results Summary

### **Model Performance**
- Recognizable structures  
- Strong color matching  
- Very poor anatomy & boundaries  
- Blurred objects  
- Weak semantic alignment  

### **CFG Behavior**
- Best scale remains **7.5**  
- Lower = poor alignment  
- Higher = oversaturation  

---

## 🔧 4. Limitations Identified

- CPU-only training = extremely slow  
- Small dataset = poor generalization  
- 256×256 resolution = low visual detail  
- Only 4 epochs = severely undertrained  
- Limited CLIP guidance  
- No LoRA or memory-efficient fine-tuning  

---

## 🚀 5. Next Steps (Before Final Milestone)

### ✔ Training Upgrades
- Expand dataset to **4k–5k samples**  
- Train for **20–30 epochs**  
- Use **LoRA fine-tuning**  

### ✔ Sampling Enhancements
- Use **Euler**, **DPM++**, or DDIM  
- Increase inference steps (50–100)  

### ✔ Evaluation Improvements
- Add **CLIP Score**  
- Add **Precision/Recall** metrics  
- Add side-by-side prompt comparison charts  

### ✔ Output Quality Improvements
- Explore **512×512** training  
- Use caption engineering templates  

---

## 🏁 6. Conclusion

Milestone 3 completes the evaluation phase of Text2ImageGen.  
Although the model still produces distorted images, the evaluation pipeline is now fully established.  
This milestone lays the groundwork for major improvements in the final stage.

---

## Dependencies

Core packages:
- `torch==2.1.0` - Deep learning framework
- `diffusers==0.25.0` - Stable Diffusion models
- `transformers==4.36.0` - CLIP text encoder
- `datasets==2.14.6` - Flickr30K dataset
- `Pillow`, `pandas`, `matplotlib` - Data processing & visualization

See `requirements.txt` for complete list.
