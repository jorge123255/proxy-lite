# Proxy-Lite-3B Finetuning Guide

## Model Files Overview

The model consists of several key files that are important for finetuning:

### Core Model Files
- `model-00001-of-00002.safetensors` (5 GB) - First part of model weights
- `model-00002-of-00002.safetensors` (2.51 GB) - Second part of model weights
- `model.safetensors.index.json` (65.4 kB) - Index file for model weights

### Configuration Files
- `config.json` (1.27 kB) - Main model configuration
- `generation_config.json` (126 Bytes) - Generation parameters
- `preprocessor_config.json` (578 Bytes) - Input preprocessing settings
- `tokenizer_config.json` (9.48 kB) - Tokenizer configuration

### Tokenizer Files
- `tokenizer.json` (11.4 MB) - Main tokenizer file
- `vocab.json` (2.78 MB) - Vocabulary file
- `added_tokens.json` (605 Bytes) - Additional tokens
- `special_tokens_map.json` (613 Bytes) - Special token mappings
- `merges.txt` (1.67 MB) - BPE merges file

### Templates
- `chat_template.json` (4.69 kB) - Chat formatting template

## Finetuning Setup

### 1. Model Architecture
The model is based on Qwen2.5-VL-3B-Instruct with:
- Total size: ~7.51 GB (split across two safetensors files)
- Architecture: Vision-Language Model
- Parameter count: 3.75B
- Tensor type: BF16

### 2. Required Files for Finetuning
Essential files to preserve:
```
├── config.json                    # Model architecture
├── generation_config.json         # Generation settings
├── preprocessor_config.json       # Input processing
├── tokenizer_config.json         # Tokenizer settings
├── tokenizer.json                # Tokenizer
├── vocab.json                    # Vocabulary
└── special_tokens_map.json       # Special tokens
```

### 3. Finetuning Process

#### Data Preparation
```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("convergence-ai/proxy-lite-3b")

def prepare_inputs(example):
    return {
        "input_ids": processor(
            text=example["text"],
            images=example["screenshot"],
            return_tensors="pt"
        ).input_ids
    }
```

#### Training Configuration
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./finetuned-proxy-lite",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)
```

#### LoRA Configuration
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 4. Important Considerations

1. **Memory Management**
- Use gradient checkpointing
- Implement LoRA or QLoRA for efficient training
- Consider using 8-bit or 4-bit quantization

2. **Data Format**
- Follow the chat template format in `chat_template.json`
- Ensure proper handling of vision inputs
- Maintain tool call format as specified in original model

3. **Validation**
- Test model with original WebVoyager benchmark
- Validate tool calling capabilities
- Check vision-language understanding

4. **Recommended Hardware**
- Minimum: 24GB VRAM (RTX 3090 or better)
- Recommended: 2x RTX 3090 for faster training
- SSD storage: 50GB+ for model and datasets

## Useful Commands

```bash
# Clone and setup
git clone [your-repo]
cd [your-repo]

# Install dependencies
pip install -r requirements.txt
pip install -U peft transformers accelerate bitsandbytes

# Start finetuning
python train.py \
    --base_model "convergence-ai/proxy-lite-3b" \
    --data_path "./your_dataset" \
    --output_dir "./finetuned-model" \
    --use_lora True \
    --use_8bit True
```

## Resources
- Original model: https://huggingface.co/convergence-ai/proxy-lite-3b
- Mind2Web dataset: https://github.com/OSU-NLP-Group/Mind2Web
- WebArena dataset: https://github.com/web-arena-x/webarena 