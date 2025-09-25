# GPT-OSS-mini: GPT Implementation from Scratch

A complete implementation of a modern GPT model incorporating state-of-the-art techniques from transformer research, built from the ground up using PyTorch.

## Overview
This project demonstrates how to build a production-ready GPT model with modern architectural improvements, complete training pipeline, and text generation capabilities. The implementation includes comprehensive explanations and working code for every component.

## Key Features

### Modern Architecture Components
- **Rotary Position Embedding (RoPE):** Superior positional encoding that enables better length extrapolation  
- **Grouped Query Attention (GQA):** Memory-efficient attention mechanism that reduces KV cache size  
- **Mixture of Experts (MoE):** Sparse routing for scalable model capacity  
- **SwiGLU Activation:** Advanced activation functions with gating mechanism  
- **RMS Normalization:** Efficient alternative to LayerNorm  

### Complete Training Pipeline
- Modern optimization with AdamW and cosine scheduling  
- Gradient clipping and accumulation  
- Comprehensive monitoring and checkpointing  
- Training visualization and metrics tracking  
- Real-time loss monitoring and best model saving  

## Technical Implementation

### Model Architecture
```python
# Example model configuration
config = {
    'vocab_size': 50304,      # Optimized vocabulary size
    'd_model': 768,           # Model dimension
    'n_layers': 12,           # Number of transformer blocks
    'n_heads': 12,            # Query heads
    'n_kv_heads': 3,          # Key-value heads (GQA)
    'd_ff': 3072,             # Feed-forward dimension
    'num_experts': 8,         # MoE experts
    'top_k': 2,               # Active experts per token
    'max_seq_len': 2048,      # Maximum sequence length
    'dropout': 0.1
}
```

### Training Features
- **Data Pipeline:** Efficient tokenization using tiktoken (GPT-4 tokenizer)  
- **Sliding Window Dataset:** Optimized data loading with configurable stride  
- **Mixed Training:** Supports both small demonstrations and large-scale training  
- **Checkpoint Management:** Automatic saving and loading of model states  
- **Training Visualization:** Real-time loss curves and learning rate scheduling  

## Quick Start

### 1. Installation
```bash
pip install torch torchvision torchaudio
pip install tiktoken==0.11.0
pip install matplotlib
pip install requests
```

### 2. Basic Usage
```python
from gpt_oss_mini import GPTOSSMini, GPTTrainer

# Create model
model = GPTOSSMini(**config)

# Train on your data
trainer = GPTTrainer(model, tokenizer, train_data)
trainer.train(train_loader, val_loader)

# Generate text
generated = model.generate(input_ids, max_new_tokens=50)
```

### 3. Demo Training
The notebook includes a complete training demonstration using Wikipedia data:
- Downloads Python programming language article  
- Creates training dataset with sliding windows  
- Trains a smaller model (256 dim, 4 layers) for demonstration  
- Shows training progress with loss curves  
- Demonstrates text generation with different sampling strategies  

## Implementation Details

### Rotary Position Embedding (RoPE)
- Encodes position by rotating query/key vectors  
- Enables better relative position understanding  
- Supports length extrapolation beyond training data  

### Grouped Query Attention (GQA)
- Reduces memory usage by sharing K,V heads across multiple Q heads  
- Maintains attention quality while improving efficiency  
- Configurable group sizes for different memory/quality trade-offs  

### Mixture of Experts (MoE)
- Routes tokens to top-k most relevant expert networks  
- Scales model capacity without increasing per-token computation  
- Includes load balancing and efficient sparse computation  

### SwiGLU Activation
- Combines Swish activation with gating mechanism  
- Used in state-of-the-art models like LLaMA and PaLM  
- Improves gradient flow and model performance  

### RMS Normalization
- More efficient than LayerNorm (25% fewer operations)  
- Better numerical stability  
- Used in modern large language models  

## Training Results
The demo training shows effective learning on Wikipedia data:
- **Training Loss:** Decreases from ~12.26 to ~5.18 (57.7% improvement)  
- **Learning Rate:** Cosine schedule with warmup for stable training  
- **Convergence:** Smooth loss curves indicating proper optimization  

## Text Generation
Multiple sampling strategies are implemented:
- **Greedy Decoding:** Deterministic, coherent output  
- **Temperature Sampling:** Balanced creativity and coherence  
- **Top-k Sampling:** Controlled randomness for diverse outputs  

## File Structure
```
gpt-oss-from-scratch.ipynb    # Complete implementation notebook
├── Environment Setup         # Dependencies and imports
├── Data Pipeline            # Wikipedia data download and processing
├── Tokenization            # tiktoken integration and utilities
├── Model Architecture      # All modern components implementation
├── Training Pipeline       # Complete training system
├── Demonstration          # Working training example
└── Text Generation        # Inference and sampling strategies
```

## Advanced Features

### Training Monitoring
- Real-time loss tracking and visualization  
- Learning rate scheduling with warmup and cosine decay  
- Automatic best model checkpointing  
- Comprehensive metrics logging  

### Memory Optimization
- Grouped Query Attention reduces KV cache size  
- RMS Normalization for computational efficiency  
- Optimized data loading with configurable batch sizes  

### Extensibility
- Modular design for easy component swapping  
- Configurable model sizes and architectures  
- Support for custom datasets and tokenizers  

## Performance Notes
The implementation prioritizes:
- **Correctness:** Faithful implementation of published techniques  
- **Clarity:** Comprehensive documentation and explanations  
- **Modularity:** Easy to modify and extend components  
- **Efficiency:** Modern optimizations and best practices  

## Next Steps

### Performance Optimizations
- KV Cache for faster inference  
- Flash Attention for memory efficiency  
- Mixed precision training (FP16/BF16)  

### Advanced Techniques
- Sliding window attention for long sequences  
- Retrieval-augmented generation (RAG)  
- Multi-query attention (MQA) for maximum efficiency  

### Scaling & Production
- Distributed training across multiple GPUs  
- Model parallelism for very large models  
- Quantization for deployment optimization  

## Requirements
- Python 3.7+  
- PyTorch 2.0+  
- tiktoken 0.11.0  
- matplotlib (for visualization)  
- requests (for data download)  

## License
This implementation is for educational purposes, demonstrating modern transformer architectures and training techniques. The code provides a complete, working example suitable for research and learning.

## Contributing
This is an educational implementation focused on clarity and completeness. The code demonstrates production-ready techniques while maintaining readability for learning purposes.
