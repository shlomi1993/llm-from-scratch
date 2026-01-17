# LLM from Scratch

A comprehensive implementation of GPT-2 language models built from the ground up, featuring pre-training, fine-tuning, and interactive applications. This project is based on the excellent work in [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and provides a complete CLI interface for working with transformer-based language models.

## Features

- 🏗️ **Complete GPT-2 Implementation**: Build and understand generative transformer models from scratch
- 📥 **Download Models**: Download pre-trained GPT-2 models (124M, 355M, 774M, 1558M) in TensorFlow format and convert them to PyTorch format
- 🧩 **Flexible Configuration**: Customize model architectures and training hyperparameters
- 🏋️ **Pre-training**: Train foundation models from scratch on custom text corpora
- 🎯 **Fine-tuning**: Adapt models for specific tasks (classification, instruction-following, and coding)
- 💬 **Interactive Applications**: Chat with fine-tuned assistants, classify text in real-time, or interactively generate code with a coding assistant
- 📈 **Visualization**: Visually track training loss and performance over time

## Installation

### Prerequisites

- Python 3.10 or higher
  - See [`pyproject.toml`](pyproject.toml) for the complete dependency list.
  - Packages and libraries can be easily installed using the [`install.sh`](install.sh) script
- 8GB+ RAM (16GB+ recommended for larger models)
- CUDA GPU (optional, for faster training)

**Note:** The project was implemented and tested on Apple ecosystem (MacBook Pro 16 2022)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shlomi1993/llm-from-scratch.git
cd llm-from-scratch
```

2. Install and activate the environment:

**Option A: Automated Setup (Recommended)**
```bash
# Run the installation script
./install.sh

# Activate the environment
source activate.sh
```

**Option B: Manual Setup**
```bash
# Create virtual environment
python -m venv llm-from-scratch-venv

# Activate it
source llm-from-scratch-venv/bin/activate  # On Windows: llm-from-scratch-venv\Scripts\activate

# Install dependencies
pip install -e .
```

The `gpt2` command is now available in your environment!

## CLI Reference

The `gpt2` command provides access to all functionality:

```
gpt2 <command> [options]

Commands:
    download    Download GPT2 model files
    pretrain    Train a GPT2 foundation model from scratch
    generate    Generate text using a pre-trained GPT2 model
    finetune    Fine-tune a pre-trained GPT2 model for downstream tasks
    spam-ham    Classify text as spam or ham using a fine-tuned classification model
    chat        Chat with an instruction fine-tuned assistant model
    coder       Get coding assistance from a fine-tuned coder model
```

For each command, use `--help` to see additional arguments, options and flags. There are many options and flags, but most of them have default values!

### Command Tree

```
gpt2
├── download
├── pretrain
├── generate
│   ├── non-interactive   # pass --prompt
│   └── interactive       # don't pass --prompt
├── finetune
│   ├── classification    # classification finetuning
│   ├── instruction       # instruction finetuning
│   ├── instruction-adv   # advanced instruction finetuning (not tested!)
│   └── code-instruction  # code instruction finetuning
├── spam-ham
├── chat
└── coder
```

## Quick Start

### Download "Formal" Pre-trained Models

```bash
gpt2 download --sizes 124M --dir models --convert
```

This script downloads the selected official pre-trained models in TensorFlow format and converts them to PyTorch format.  
To download custom pre-trained or fine-tuned models, check the following section.

### Download Custom Fine-tuned Models

**Automated Download (Recommended):**

Use the provided script to download all custom models automatically:

```bash
# Download all models to default location (models/)
./download_custom_models.sh

# Download to a custom directory
./download_custom_models.sh my_models
```

**Manual Download:**

Alternatively, download models individually from Google Drive:

- **Pretrained Model:** A GPT-2 based foundation model trained from scratch on a small, custom dataset. Useful for educational purposes and experimentation, but less capable than official models above due to limited data.
  - [Download foundation.zip](https://drive.google.com/file/d/1AZP-AEEm8NJF4wGXkxEF-hxAGBAdShIt/view?usp=drive_link)
- **Classifier Model:** Fine-tuned on SMS spam-or-ham dataset using the official 124M GPT-2 model. Use this for spam detection task.
  - [Download spam_classifier.zip](https://drive.google.com/file/d/1ecl-LeMq3fgNBEDjkKzisWp4ube40Jb0/view?usp=drive_link)
- **Assistant Model:** Fine-tuned for instruction-following and chat, based on the official 355M GPT-2 model. Use this for interactive assistant or chatbot applications.
  - [Download assistant.zip](https://drive.google.com/file/d/1qiM0YHdnciGoUaJadnQzL7hDhiLHJ-QZ/view?usp=drive_link)
- **Coder Model:** Fine-tuned on Python code instruction dataset using the official 355M GPT-2 model. Use this for code generation and interactive coding assistance.
  - [Download coder.zip](https://drive.google.com/file/d/1PQqUhlMSgsgw46SKr7dnDhPOATG3017k/view?usp=drive_link)

### Pre-train a Foundation Model

```bash
gpt2 pretrain --training-set-path dataset/the-verdict.txt --saved-model-path models/pretrained/foundation.pth
```

### Generate Text

```bash
# Interactive mode
gpt2 generate --model-path models/124M/model.pth

# Single generation
gpt2 generate --model-path models/124M/model.pth --prompt "Once upon a time"
```

### Fine-tune for Classification

```bash
gpt2 finetune classification --pretrained-model-path models/124M/model.pth --tuning-set-path dataset/sms_spam_collection/SMSSpamCollection.tsv --n-epochs 5 --model-save-path classifier.pth
```

### Classify Text to Spam or Ham

```bash
gpt2 spam-ham --model-path classifier.pth --text "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
```

### Fine-tune for Instruction Following

```bash
gpt2 finetune instruction --pretrained-model-path models/355M/model.pth --tuning-set-path dataset/instruction_data/instruction-data.json --n-epochs 2 --model-save-path assistant.pth --evaluate
```

### Chat with an Assistant

```bash
gpt2 chat --model-path assistant.pth
```

### Fine-tune for Code Generation

```bash
gpt2 finetune coding --pretrained-model-path models/355M/model.pth --dataset-path dataset/python_code_instructions/ --max-samples 100 --batch_size 4 --n-epochs 1 --model-save-path coder.pth --test-output-path responses.json --evaluate
```

- **Note:** The use of `--max-samples 100` is to limit dataset size for test speed

### Interactive Coding Session

```bash
gpt2 coder --model-path coder.pth
```

## Project Structure

```
llm-from-scratch/

├── appendices/                             # Supplementary documentation adapted from the source repository
├── chapters/                               # Chapter notebooks and reference implementations from the source repository
├── dataset/                                # Training, fine-tuning, and evaluation datasets
│   ├── instruction_data/                   # Instruction-following datasets
│   ├── python_code_instructions/           # Python code generation datasets
│   ├── sms_spam_collection/                # SMS spam classification dataset
│   ├── small-text-sample.txt               # Small text sample for testing
│   └── the-verdict.txt                     # Verdict text for pretraining
├── models/                                 # Saved and checkpointed model artifacts (downloaded/trained)
├── presentation/                           # Seminar presentation slides and figures
├── src/
│   ├── cli.py                              # Primary command-line interface entry point
│   ├── dataset.py                          # Dataset definitions and abstractions
│   ├── model/
│   │   ├── activation.py                   # Activation functions (GELU, etc.)
│   │   ├── attention/                      # Attention mechanisms and modules
│   │   │   ├── base.py                     # Base attention interface
│   │   │   ├── multihead.py                # Multi-head attention implementation
│   │   │   └── advanced/                   # Examples for advanced attention variants (GQA, MLA, SWA, etc.)
│   │   ├── config.py                       # Model and training configuration
│   │   ├── feed_forward.py                 # Feed-forward network components
│   │   ├── gpt.py                          # GPT model implementation
│   │   ├── normalization.py                # Layer normalization
│   │   └── transformer.py                  # Transformer block building blocks
│   ├── scripts/
│   │   ├── chat.py                         # Interactive chat interface
│   │   ├── classify.py                     # Spam-or-ham classification flow
│   │   ├── coder.py                        # Interactive coding session
│   │   ├── download.py                     # Model download helpers
│   │   ├── finetune/
│   │   │   ├── classification.py           # Classification fine-tuning flow
│   │   │   ├── code_instruction.py         # Code instruction fine-tuning flow
│   │   │   ├── instruction.py              # Instruction fine-tuning flow
│   │   │   └── instruction_adv.py          # Advanced instruction fine-tuning (experimental)
│   │   ├── generate.py                     # Simple text generation flow
│   │   ├── interactive_session.py          # Base interactive session class
│   │   └── train.py                        # Foundation model pretraining flow
│   └── utils/
│       ├── checkpoint.py                   # Model checkpoint save/load utilities
│       ├── device.py                       # Device management (CPU/CUDA/MPS)
│       ├── logger.py                       # Logging configuration and helpers
│       ├── losses.py                       # Loss calculation utilities
│       ├── ollama.py                       # Ollama API integration for evaluation
│       ├── tokenization/                   # Tokenizer implementation and utilities
│       │   ├── bpe_openai_gpt2.py          # BPE tokenizer implementation example
│       │   ├── tokenizer.py                # Tokenizer wrapper
│       │   └── assets/                     # Tokenizer vocabulary assets
│       └── visualization.py                # Plotting and visualization helpers
├── tests/                                  # End-to-end system tests for core CLI workflows
│   ├── ref/                                # Reference for notebook original script outputs
│   ├── chapters_code.py                    # Shared test utilities from chapters
│   ├── common.py                           # Common test fixtures and helpers
│   ├── conftest.py                         # Pytest configuration
│   ├── test_instruction_finetuning.py      # Tests for chat assistant
│   ├── test_class_finetuning.py            # Tests for spam classifier
│   ├── test_code_finetuning.py             # Tests for code generation
│   ├── test_generation.py                  # Tests for text generation
│   └── test_pretraining.py                 # Tests for pretraining
├── activate.sh                             # Virtual environment activation script
├── download_custom_models.sh               # Script to download all custom fine-tuned models from Google Drive
├── install.sh                              # Automated project setup and installation
├── pyproject.toml                          # Project metadata and dependency configuration
└── README.md                               # This file

```

## Training Tips

### Memory Optimization

- **Use smaller batch sizes**: Start with `--batch-size 1` or `2` for large models
- **Choose appropriate model size**: 124M for testing, 355M for development, 774M+ for production
- **Enable gradient checkpointing**: Reduces memory at the cost of speed (if implemented)
- **Use CPU for large models**: `--device cpu` if GPU or MPS memory is insufficient

### Hyperparameter Tuning

- **Learning rate**: Start with `5e-5` for fine-tuning, `5e-4` for pre-training
- **Epochs**: 1-3 epochs for fine-tuning is usually sufficient
- **Batch size**: Balance between memory and training stability (2, 4, 8 typical)
- **Temperature**: Lower (0.7) for focused output, higher (1.2) for creative output

### Best Practices

1. **Start small**: Test with 124M model before scaling up
2. **Monitor losses**: Watch for NaN values, indicates learning rate too high
3. **Validate frequently**: Use `--eval-freq` to track progress
4. **Save checkpoints**: Always specify `--model-save-path`
5. **Use evaluation**: Enable `--evaluate` for instruction tuning to measure quality

## Dataset Format

### Classification Data (TSV/CSV)

```tsv
Label	Text
spam	Win a free iPhone now! Click here!
ham	Hi, are we still meeting for lunch?
```

### Instruction Data (JSON)

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  },
  {
    "instruction": "Translate the following to Spanish:",
    "input": "Hello, how are you?",
    "output": "Hola, ¿cómo estás?"
  }
]
```

### Coding Instruction Data (JSON)

**Note:** The coding instruction dataset is sourced from [Hugging Face: iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca). It is provided in JSON format mirrored as a `.arrow` file.

```json
[
  {
    "instruction": "Write a Python function that returns the square of a number.",
    "input": "",
    "output": "def square(x):\n    return x * x"
  },
  {
    "instruction": "Write a Python function that checks if a string is a palindrome.",
    "input": "",
    "output": "def is_palindrome(s):\n    return s == s[::-1]"
  },
]
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests with terminal outputs
pytest -s tests/ -v

# Run specific test file
pytest tests/test_pretrain_cli.py -v
pytest tests/test_generate_cli.py -v
pytest tests/test_classifier_cli.py -v
pytest tests/test_assistant_cli.py -v
pytest tests/test_coder_cli.py -v

```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch-size` to 1 or 2
- Use smaller model (124M instead of 355M/774M)
- Switch to CPU: `--device cpu`
- Close other applications

### NaN Losses

- Lower learning rate: `--lr 5e-5` or `--lr 1e-5`
- Check data format and quality
- Reduce batch size
- Ensure proper data normalization

### Slow Training

- Use GPU if available: `--device cuda` or `--device mps`
- Increase batch size (if memory allows)
- Reduce `--eval-freq` for less frequent validation
- Use smaller model for experimentation

## License

This project is based on [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.

## Acknowledgments

- Sebastian Raschka for the excellent [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) book and repository
- OpenAI for the GPT-2 architecture and pre-trained models
- The PyTorch team for the excellent deep learning framework
