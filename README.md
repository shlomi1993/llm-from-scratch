# LLM from Scratch

A comprehensive implementation of GPT-2 language models built from the ground up, featuring pre-training, fine-tuning, and interactive applications. This project is based on the excellent work in [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and provides a complete CLI interface for working with transformer-based language models.

## Features

- 🏗️ **Complete GPT-2 Implementation**: Build and understand transformer models from scratch
- 📥 **Model Management**: Download and convert pre-trained GPT-2 models (124M, 355M, 774M, 1558M)
- 🎓 **Pre-training**: Train foundation models from scratch on custom text corpora
- 🎯 **Fine-tuning**: Adapt models for specific tasks (classification, instruction-following)
- 💬 **Interactive Applications**: Chat with fine-tuned assistants or classify text in real-time
- 📊 **Visualization**: Training metrics, loss plots, and performance tracking
- 🔧 **Flexible Configuration**: Customizable model architectures and training hyperparameters

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.9.0+
- 8GB+ RAM (16GB+ recommended for larger models)
- CUDA GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-from-scratch.git
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

## Quick Start

### Download a Pre-trained Model

```bash
gpt2 download --sizes 124M --dir models --convert
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
gpt2 finetune classification \
  --pretrained-model-path models/124M/model.pth \
  --tuning-set-path datasets/sms_spam_collection/SMSSpamCollection.tsv \
  --column-names Label Text \
  --n-epochs 5 \
  --model-save-path classifier.pth
```

### Fine-tune for Instruction Following

```bash
gpt2 finetune instruction \
  --pretrained-model-path models/355M/model.pth \
  --tuning-set-path datasets/instruction_data/instruction-data.json \
  --n-epochs 2 \
  --model-save-path assistant.pth \
  --evaluate
```

### Chat with an Assistant

```bash
gpt2 chat-bot --model-path assistant.pth
```

## CLI Reference

The `gpt2` command provides access to all functionality:

```
gpt2 <command> [options]

Commands:
    download    Download GPT2 model files
    pretrain    Train a GPT2 foundation model from scratch
    generate    Generate text using a pre-trained GPT2 model
    finetune    Fine-tune a pre-trained GPT2 foundation model
    spam-ham    Classify text as spam or ham using a classification fine-tuned model
    chat-bot    Chat with an instruction fine-tuned assistant model
```

For each command, use `--help` to see additional arguments, options and flags. There are many options and flags, but most of them have default values!

### Command Tree

```
gpt2
├── download
├── pretrain
├── generate
│   ├── non-interactive  (pass --prompt)
│   └── interactive      (don't pass --prompt)
├── finetune
│   ├── classifier       (classification finetuning)
│   └── assistant        (instruction finetuning)
├── spam-ham
└── chat-bot
```


## Project Structure

```
llm-from-scratch/

├── appendices/                             # Supplementary documentation adapted from the source repository
├── chapters/                               # Chapter notebooks and reference implementations from the source repository
├── datasets/                               # Training, fine-tuning, and evaluation datasets
├── models/                                 # Saved and checkpointed model artifacts
├── presentation/                           # Seminar presentation slides and figures
├── src/
│   ├── cli.py                              # Primary command-line interface entry point
│   ├── data/
│   │   ├── datasets.py                     # Dataset definitions and abstractions
│   │   └── loaders.py                      # Data loading and preprocessing utilities
│   ├── model/
│   │   ├── activation.py                   # Activation functions
│   │   ├── attention/                      # Attention mechanisms and modules
│   │   ├── config.py                       # Model and training configuration
│   │   ├── feed_forward.py                 # Feed-forward network components
│   │   ├── gpt.py                          # GPT model implementation
│   │   ├── normalization.py                # Normalization layers
│   │   └── transformer.py                  # Transformer architecture building blocks
│   ├── scripts/
│   │   ├── chat.py                         # Interactive chat interface
│   │   ├── classify.py                     # Spam-or-ham classification flow
│   │   ├── common.py                       # Shared script utilities
│   │   ├── download.py                     # Model download helpers
│   │   ├── finetune/
│   │   │   ├── classification.py           # Model classification finetuning flow
│   │   │   ├── instruction.py              # Model Instruction fine-tuning flow
│   │   │   └── instruction_advanced.py     # Model Advanced instruction fine-tuning flow (UNTESTED)
│   │   ├── generate.py                     # Simple text generation flow
│   │   └── pretrain.py                     # Foundation model pretraining flow
│   └── utils/
│       ├── device.py                       # Device management
│       ├── logger.py                       # Logging configuration and helpers
│       ├── ollama.py                       # Ollama API integration for assistant evaluation
│       ├── tokenization/                   # Tokenizer implementation and utilities
│       └── visualization.py                # Plotting and visualization helpers
├── tests/                                  # End-to-end system tests for core CLI workflows
├── activate.sh                             # Virtual environment activation script
├── install.sh                              # Automated project setup and installation
└── pyproject.toml                          # Project metadata and dependency configuration

```

## Training Tips

### Memory Optimization

- **Use smaller batch sizes**: Start with `--batch-size 1` or `2` for large models
- **Choose appropriate model size**: 124M for testing, 355M for development, 774M+ for production
- **Enable gradient checkpointing**: Reduces memory at the cost of speed (if implemented)
- **Use CPU for large models**: `--device cpu` if GPU memory is insufficient

### Hyperparameter Tuning

- **Learning rate**: Start with `5e-5` for fine-tuning, `5e-4` for pre-training
- **Epochs**: 2-5 epochs for fine-tuning is usually sufficient
- **Batch size**: Balance between memory and training stability (4-8 typical)
- **Temperature**: Lower (0.7) for focused output, higher (1.2) for creative output

### Best Practices

1. **Start small**: Test with 124M model before scaling up
2. **Monitor losses**: Watch for NaN values (indicates learning rate too high)
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

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pretrain_cli.py -v

# Run with output
pytest tests/ -s
```

## Requirements

- Python 3.10+
- PyTorch 2.9.0+
- tiktoken (GPT-2 tokenizer)
- tqdm (progress bars)
- matplotlib (visualizations)
- ollama (optional, for evaluation)
- requests (model downloads)

See `pyproject.toml` for complete dependency list.

## Performance Benchmarks

| Model Size | Parameters | Memory (Training) | Speed (tokens/sec) |
|------------|-----------|-------------------|-------------------|
| 124M       | 124M      | ~2GB              | ~1000            |
| 355M       | 355M      | ~6GB              | ~400             |
| 774M       | 774M      | ~12GB             | ~200             |
| 1558M      | 1.5B      | ~24GB             | ~100             |

*Benchmarks on Apple M1 Max with MPS acceleration*

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

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is based on [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.

## Acknowledgments

- Sebastian Raschka for the excellent [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) book and repository
- OpenAI for the GPT-2 architecture and pre-trained models
- The PyTorch team for the excellent deep learning framework

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{llm-from-scratch,
  author = {Your Name},
  title = {LLM from Scratch: A Comprehensive GPT-2 Implementation},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/llm-from-scratch}
}
```