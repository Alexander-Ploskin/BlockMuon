import os
import math
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import wandb


class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            self.tokens = torch.load(f"{self.dataset_name}.bin")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class BlockMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz. 
    
    This is a block-wise implementation of Muon, potentially it provides us ability to train with
    multi-GPU setup. However, we use it to investigate how size of a block affects training procedure.
    
    
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    Block-wise Muon optimizer that processes parameters in blocks for improved memory efficiency.
    
    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        block_size=64,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            block_size=block_size,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params) if muon_params else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Mark parameters for Muon vs AdamW
        for p in (muon_params or []):
            assert p.ndim == 2, f"Muon parameters must be 2D, got {p.ndim}D"
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
        
        self._block_indices_cache = {}
    

    def _get_block_indices(self, shape, block_size):
        """Generate block indices for a given shape and block size with caching."""
        cache_key = (shape, block_size)
        
        if cache_key not in self._block_indices_cache:
            rows, cols = shape
            block_indices = []
            
            for i in range(0, rows, block_size):
                for j in range(0, cols, block_size):
                    end_i = min(i + block_size, rows)
                    end_j = min(j + block_size, cols)
                    block_indices.append((i, end_i, j, end_j))
            
            self._block_indices_cache[cache_key] = block_indices
        
        return self._block_indices_cache[cache_key]

    
    def _process_muon_block(self, param, grad_block, momentum_block, group, block_slice):
        """Process a single block with Muon updates."""
        # Apply momentum
        momentum_block.mul_(group["momentum"]).add_(grad_block)
        
        if group["nesterov"]:
            g_block = grad_block.add(momentum_block, alpha=group["momentum"])
        else:
            g_block = momentum_block.clone()
        
        # Apply Newton-Schulz orthogonalization
        if g_block.numel() > 1 and min(g_block.shape) > 1:
            u_block = zeropower_via_newtonschulz5(g_block, steps=group["ns_steps"])
        else:
            # For very small blocks, just normalize
            u_block = g_block / (g_block.norm() + 1e-7)
        
        # Scale update
        adjusted_lr = self.adjust_lr_for_muon(group["lr"], param.shape)
        
        # Apply weight decay
        i1, i2, j1, j2 = block_slice
        param.data[i1:i2, j1:j2].mul_(1 - group["lr"] * group["wd"])
        param.data[i1:i2, j1:j2].add_(u_block, alpha=-adjusted_lr)

    
    def _process_adamw_block(self, param, grad_block, state, group, block_slice):
        """Process a single block with AdamW updates."""
        i1, i2, j1, j2 = block_slice
        
        # Get or initialize block states
        if "moment1_blocks" not in state:
            state["moment1_blocks"] = {}
            state["moment2_blocks"] = {}
        
        block_key = (i1, i2, j1, j2)
        if block_key not in state["moment1_blocks"]:
            state["moment1_blocks"][block_key] = torch.zeros_like(grad_block)
            state["moment2_blocks"][block_key] = torch.zeros_like(grad_block)
        
        buf1 = state["moment1_blocks"][block_key]
        buf2 = state["moment2_blocks"][block_key]
        
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        
        # Update moments
        buf1.lerp_(grad_block, 1 - beta1)
        buf2.lerp_(grad_block.square(), 1 - beta2)
        
        # Compute update
        g_block = buf1 / (eps + buf2.sqrt())
        
        # Bias correction
        step = state["step"]
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        scale = bias_correction1 / bias_correction2**0.5
        
        # Apply weight decay
        param.data[i1:i2, j1:j2].mul_(1 - group["lr"] * group["wd"])
        param.data[i1:i2, j1:j2].add_(g_block, alpha=-group["lr"] / scale)

    
    def adjust_lr_for_muon(self, lr, param_shape):
        """Adjust learning rate based on parameter shape."""
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    
    def step(self, closure=None):
        """Perform a single optimization step with block-wise processing."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            block_size = group["block_size"]

            ############################
            #      Block-wise Muon     #
            ############################

            muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]
            
            for p in muon_params:
                g = p.grad
                if g is None:
                    continue
                
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                    p_2d = p.view(g.size(0), -1)
                else:
                    p_2d = p
                
                state = self.state[p]
                
                # Initialize momentum buffer for blocks
                if "momentum_blocks" not in state:
                    state["momentum_blocks"] = {}
                
                # Get block indices
                block_indices = self._get_block_indices(g.shape, block_size)

                # Process each block
                for i1, i2, j1, j2 in block_indices:
                    grad_block = g[i1:i2, j1:j2]
                    block_key = (i1, i2, j1, j2)
                    
                    # Initialize momentum for this block if needed
                    if block_key not in state["momentum_blocks"]:
                        state["momentum_blocks"][block_key] = torch.zeros_like(grad_block)
                    
                    momentum_block = state["momentum_blocks"][block_key]
                    
                    # Process block with Muon
                    self._process_muon_block(
                        p_2d, grad_block, momentum_block, group, (i1, i2, j1, j2)
                    )

            ############################
            #    Block-wise AdamW      #
            ############################

            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            
            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                
                # Handle different parameter shapes
                if p.ndim == 1:
                    # 1D parameters: process as single block
                    self._process_adamw_block(
                        p.view(1, -1), g.view(1, -1), state, group, (0, 1, 0, g.numel())
                    )
                elif p.ndim == 2:
                    # 2D parameters: process in blocks
                    block_indices = self._get_block_indices(g.shape, block_size)
                    
                    for i1, i2, j1, j2 in block_indices:
                        grad_block = g[i1:i2, j1:j2]
                        self._process_adamw_block(
                            p, grad_block, state, group, (i1, i2, j1, j2)
                        )

        return loss


def get_model_and_dataloader(model_name, dataset_name, hidden_size, split, max_len, batch_size):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
        "openwebtext-1k": "stas/openwebtext-synthetic-testing"
    }
    train_dataset = load_dataset(name2path[dataset_name], split=split, trust_remote_code=True)
    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer, max_length=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=513,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1, block_size=64):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return BlockMuon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
            block_size=block_size
        )

    else:
        assert 0, "optimizer not supported"


def save_checkpoint_to_wandb(model, optimizer, epoch, step, loss, upload, name):
    """Save model checkpoint to W&B as an artifact."""
    checkpoint_path = f"{name}.pt"
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    if not upload:
        return
    
    # Create W&B artifact
    artifact = wandb.Artifact(
        name=f"model_checkpoint_epoch_{epoch}",
        type="model",
        description=f"Model checkpoint at epoch {epoch}, step {step}"
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_checkpoint_interval", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="muon-optimizer-experiments-debug")
    parser.add_argument("--upload_checkpoints", type=bool, default=False)
    
    args = parser.parse_args()
    max_steps = args.max_steps
    save_checkpoint_interval = args.save_checkpoint_interval
    wandb_project = args.wandb_project
    upload_checkpoints = args.upload_checkpoints
    
    logger.add(f"logs/train_{args.model}_{args.optimizer}_lr{args.lr}.log")
    wandb.init(
        project=wandb_project,
        config={
            "model": args.model,
            "optimizer": args.optimizer,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "block_size": args.block_size,
            "dataset": args.dataset,
            "hidden_size": args.hidden_size,
            "max_steps": args.max_steps,
        },
        name=f"{args.model}_{args.optimizer}_lr{args.lr}_bs{args.block_size}"
    )
    
    model, train_loader = get_model_and_dataloader(
        args.model, args.dataset, args.hidden_size, args.split, args.max_len, args.batch_size
    )
    print('len: ', len(train_loader))
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr, block_size=args.block_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    global_step = 0
    for epoch in range(epoch):
        for step, batch in enumerate(train_loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = batch.to(device)
            input_ids = batch
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/epoch": epoch,
                "train/step": step,
                "train/global_step": global_step,
            }, step=global_step)
            logger.info(
                f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr']} Training loss: {loss.item()}"
            )
            
            if global_step != 0 and global_step % save_checkpoint_interval == 0:
                print('Saving checkpoint...')
                save_checkpoint_to_wandb(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    loss=loss,
                    upload=upload_checkpoints,
                    name=f'{global_step}_{args.block_size}_{args.optimizer}.pt'
                )
                print('Saved!')
            global_step += 1
    save_checkpoint_to_wandb(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=global_step,
        loss=loss,
        upload=upload_checkpoints,
        name=f'{global_step}_{args.block_size}_{args.optimizer}.pt'
    )
    print('Saved!')
    wandb.finish()
