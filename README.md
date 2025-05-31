# AI Toolkit - Chroma LoRA Training

Streamlined AI Toolkit for Chroma LoRA training on consumer hardware.

## Quick Start

### Requirements
- Python >3.10
- NVIDIA GPU with sufficient VRAM
- Git

### Installation

```bash
git clone https://github.com/geocine/ai-toolkit.git
git checkout plain
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Key Features

- **Chroma Model Support**: Optimized for Chroma transformer architecture
- **LoRA Training**: Efficient low-rank adaptation
- **Block-specific Learning Rates**: Fine-grained control over training
- **Automatic Bucketing**: Handles varying aspect ratios
- **Resume Training**: Automatic checkpoint recovery

## Output
Training produces:
- LoRA weights (`.safetensors`)
- Training samples
- Checkpoints for resuming


