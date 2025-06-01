# Block Muon Optimizer Experiments

This project implements a block version of the Muon optimizer and experiments with different hyperparameters to study their effects on training dynamics. The Muon optimizer uses matrix orthogonalization via Newton-Schulz iterations to provide more diverse optimization directions compared to traditional optimizers like AdamW[1]. Our implementation focuses on exploring block-wise variants of the algorithm and analyzing the relationship between block size and optimization performance.

## Prerequisites

- NVIDIA GPU with CUDA 12.1+ support
- Docker (recommended) or Python 3.8+
- Poetry for dependency management

## Build and Run

### Using Docker (Recommended)

Build the Docker image:
```bash
docker build -t block-muon-experiments .
```

Run experiments:
```bash
docker run --gpus all -e WANDB_API_KEY=<YOUR KEY> block-muon-experiments \
    --model qwen \
    --optimizer muon \
    --lr 1e-3 \
    --wd 0.1 \
    --block_size 512 \
    --dataset openwebtext-100k \
    --batch_size 8 \
    --max_steps 10000
```

### Local Installation

Install dependencies using Poetry:
```bash
poetry install
```

Run experiments:
```bash
python3 block_muon/block_muon_train.py --model qwen --optimizer muon --block_size 512
```

## Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--model` | `qwen` | Model architecture to use |
| `--optimizer` | `adamw` | Optimizer type (`adamw` or `muon`) |
| `--lr` | `1e-3` | Learning rate |
| `--wd` | `0.1` | Weight decay coefficient |
| `--block_size` | `256` | Block size for block Muon optimizer |
| `--dataset` | `openwebtext-100k` | Dataset name |
| `--split` | `train` | Dataset split to use |
| `--batch_size` | `8` | Training batch size |
| `--max_len` | `512` | Maximum sequence length |
| `--hidden_size` | `1024` | Model hidden dimension size |
| `--max_steps` | `None` | Maximum training steps (None for full dataset) |
| `--save_checkpoint_interval` | `5000` | Steps between checkpoint saves |
| `--wandb_project` | `muon-optimizer-experiments-debug` | Weights & Biases project name |
| `--upload_checkpoints` | `False` | Whether to upload checkpoints to W&B |

## Experimental Setup

**Hardware Configuration:**
- 1x NVIDIA A40 GPU (48GB VRAM)

**Model Architecture:**
- Base model: Qwen2.5-0.5B configuration
- Hidden size: 1024
- Number of layers: 12
- Attention heads: 16
- Intermediate size: 4864
- Maximum sequence length: 513
- Vocabulary size: 151,936

**Dataset:**
- [Elriggs/openwebtext-100k](https://huggingface.co/datasets/Elriggs/openwebtext-100k) - A subset of the OpenWebText dataset containing 100k samples

## Experiments

We conducted systematic experiments comparing different optimizers and block sizes:

### **Baseline Experiment**
- **AdamW Optimizer**: Standard adaptive optimizer with momentum and weight decay

### **Block Muon Experiments**
- **Block Size 1024**: Large block size for reduced communication overhead
- **Block Size 512**: Medium block size balancing performance and efficiency  
- **Block Size 256**: Smaller block size for finer-grained orthogonalization
- **Block Size 192**: Smallest block size to test granularity limits

Each experiment tracks:
- Training loss convergence
- Memory usage and computational efficiency

The experiments aim to validate the scaling properties of block Muon and identify optimal block sizes for different model scales and training scenarios.

## Further Development

To advance this research and make Muon a universal optimizer for various LLMs and a strong competitor to AdamW, we plan to implement the following enhancements:

### **Enhanced Data Tracking**
- **Gradient Norms and Update Statistics**: Monitor gradient L2 norms throughout training to detect potential instabilities and track the magnitude of parameter updates. This provides crucial insights into training dynamics and helps identify optimal learning rates.
- **SVD Entropy of Weight Matrices**: Track the singular value decomposition entropy of weight matrices to validate Muon's ability to provide more diverse optimization directions compared to AdamW. Higher SVD entropy indicates better exploration of the parameter space.
- **Intermediate Checkpoint Evaluation**: Evaluate model performance on MMLU at regular intervals during training to understand the relationship between training progress and downstream task performance.

### **Scaling Experiments**
- **Larger Model Architectures**: Extend experiments to models with 3B+ parameters to validate Muon's scalability advantages demonstrated in recent research.
- **Distributed Training Setup**: Implement distributed Muon with ZeRO-1 style optimization to achieve optimal memory efficiency and reduced communication overhead while preserving mathematical properties.

The ultimate goal is to establish Muon as a drop-in replacement for AdamW that consistently delivers superior computational efficiency while maintaining or improving model performance across diverse architectures and scales.
