#!/usr/bin/env python3
"""
Deep Delta Learning (DDL) — 完整训练脚本
=========================================
Full Training Script

基于论文 "Deep Delta Learning" (arXiv:2601.00417) 的训练循环。
Training loop based on the paper "Deep Delta Learning" (arXiv:2601.00417).

特性 / Features:
  - argparse 命令行参数 / Command-line arguments
  - 日志记录 (stdout + 文件) / Logging (stdout + file)
  - 梯度裁剪 / Gradient clipping
  - 学习率调度 (cosine/linear/constant) / LR scheduling
  - 合成数据 + WikiText-2风格数据支持 / Synthetic + WikiText-2 style data
  - 检查点保存与恢复 / Checkpoint save & resume
  - DDL门控值监控 / DDL gate value monitoring

用法 / Usage:
    # 使用默认配置 / With default config
    python train.py --config configs/default.yaml

    # 命令行覆盖参数 / Override via command line
    python train.py --config configs/default.yaml --batch-size 64 --lr 1e-4

    # 使用合成数据快速测试 / Quick test with synthetic data
    python train.py --data-source synthetic --max-steps 1000

Author: Auto-generated from paper analysis
License: MIT
"""

import os
import sys
import math
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到路径 / Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DDLModel, StandardTransformer, count_params


# ============================================================
# 日志配置 / Logging Configuration
# ============================================================

def setup_logging(log_dir: str, log_interval: int = 50) -> logging.Logger:
    """
    配置日志系统 / Configure logging system

    同时输出到stdout和日志文件。
    Outputs to both stdout and log file.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("ddl_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 格式 / Format
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件handler / File handler
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# 数据加载 / Data Loading
# ============================================================

class SyntheticTokenDataset(Dataset):
    """
    合成Token数据集 / Synthetic Token Dataset

    生成带有简单模式的随机Token序列，用于验证训练流程。
    Generates random token sequences with simple patterns for
    validating the training pipeline.

    模式 / Patterns:
      - 位置相关分布: token的概率依赖于其在序列中的位置
      - 局部重复: 有概率重复前面出现的token
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        """
        Args:
            vocab_size: 词汇表大小 / Vocabulary size
            seq_len: 序列长度 / Sequence length
            num_samples: 样本数量 / Number of samples
            seed: 随机种子 / Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # 预生成数据 / Pre-generate data
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1), generator=rng)

        # 注入简单模式: 每隔k个位置重复token (使模型有东西可以学)
        # Inject simple patterns: repeat tokens every k positions
        # (so the model has something to learn)
        for i in range(num_samples):
            for t in range(4, seq_len + 1):
                if torch.rand(1, generator=rng).item() < 0.15:
                    # 15%概率: 重复4步前的token / 15% chance: repeat token from 4 steps ago
                    self.data[i, t] = self.data[i, t - 4]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # input: [0:seq_len], target: [1:seq_len+1]
        return self.data[idx, :self.seq_len], self.data[idx, 1:self.seq_len + 1]


class WikiText2StyleDataset(Dataset):
    """
    WikiText-2风格数据集 / WikiText-2 Style Dataset

    尝试加载WikiText-2；如果不可用则退回到合成数据。
    Tries to load WikiText-2; falls back to synthetic data if unavailable.

    注意: 真正的WikiText-2需要 torchtext 或手动下载。
    Note: Real WikiText-2 requires torchtext or manual download.
    这里提供一个简化版，用字符级编码模拟。
    This provides a simplified version with character-level encoding.
    """

    def __init__(self, seq_len: int, num_samples: int = 5000, split: str = "train",
                 vocab_size: int = 10000, seed: int = 42):
        super().__init__()
        self.seq_len = seq_len

        # 尝试加载真实数据 / Try loading real data
        real_data = self._try_load_wikitext2(split)

        if real_data is not None:
            self.vocab_size = real_data["vocab_size"]
            self.data = self._chunk_data(real_data["tokens"], seq_len, num_samples)
        else:
            # 退回到合成数据（带有更丰富的模式）
            # Fall back to synthetic data with richer patterns
            self.vocab_size = vocab_size
            rng = torch.Generator().manual_seed(seed + hash(split) % 1000)
            self.data = self._generate_rich_synthetic(
                vocab_size, seq_len, num_samples, rng
            )

    def _try_load_wikitext2(self, split: str) -> Optional[Dict]:
        """尝试加载WikiText-2 / Try loading WikiText-2"""
        try:
            from torchtext.datasets import WikiText2
            # 尝试用torchtext加载 / Try loading with torchtext
            iterator = WikiText2(split=split)
            tokens = []
            for line in iterator:
                tokens.extend(line.strip().split())
            if len(tokens) < 1000:
                return None
            # 简单字符编码 / Simple character encoding
            vocab = {"<pad>": 0, "<unk>": 1}
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
            token_ids = [vocab.get(t, 1) for t in tokens]
            return {"tokens": torch.tensor(token_ids), "vocab_size": len(vocab)}
        except (ImportError, Exception):
            return None

    def _chunk_data(self, tokens: torch.Tensor, seq_len: int, num_samples: int) -> torch.Tensor:
        """将长序列切分为固定长度样本 / Chunk long sequence into fixed-length samples"""
        total_needed = num_samples * (seq_len + 1)
        if len(tokens) < total_needed:
            # 循环填充 / Circular padding
            repeats = (total_needed // len(tokens)) + 1
            tokens = tokens.repeat(repeats)
        chunks = tokens[:total_needed].view(num_samples, seq_len + 1)
        return chunks

    def _generate_rich_synthetic(
        self, vocab_size: int, seq_len: int, num_samples: int,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """
        生成具有多种模式的合成数据 / Generate synthetic data with multiple patterns

        模式 / Patterns:
          1. 局部重复 (n-gram) / Local repetition
          2. 主题切换 (topic blocks) / Topic switching
          3. 位置相关偏好 / Position-dependent preferences
        """
        data = torch.randint(0, vocab_size, (num_samples, seq_len + 1), generator=rng)

        for i in range(num_samples):
            # 主题: 每个样本有一个"主题"偏移 / Topic offset per sample
            topic_offset = torch.randint(0, vocab_size // 4, (1,), generator=rng).item()

            for t in range(seq_len + 1):
                r = torch.rand(1, generator=rng).item()
                if r < 0.1:
                    # 重复2步前 / Repeat from 2 steps ago
                    if t >= 2:
                        data[i, t] = data[i, t - 2]
                elif r < 0.2:
                    # 重复5步前 / Repeat from 5 steps ago
                    if t >= 5:
                        data[i, t] = data[i, t - 5]
                elif r < 0.5:
                    # 主题范围内的token / Token within topic range
                    data[i, t] = (topic_offset + torch.randint(0, 100, (1,), generator=rng).item()) % vocab_size

        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :self.seq_len], self.data[idx, 1:self.seq_len + 1]


def create_data_loader(
    data_source: str,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    split: str = "train",
    seed: int = 42,
) -> DataLoader:
    """
    创建数据加载器 / Create data loader

    Args:
        data_source: "synthetic" 或 "wikitext2"
        vocab_size: 词汇表大小
        seq_len: 序列长度
        batch_size: 批次大小
        num_workers: 数据加载线程数
        split: 数据划分 / Data split
        seed: 随机种子

    Returns:
        DataLoader
    """
    if data_source == "wikitext2":
        dataset = WikiText2StyleDataset(
            seq_len=seq_len, num_samples=5000 if split == "train" else 500,
            split=split, vocab_size=vocab_size, seed=seed,
        )
    else:
        dataset = SyntheticTokenDataset(
            vocab_size=vocab_size, seq_len=seq_len,
            num_samples=10000 if split == "train" else 1000,
            seed=seed + (0 if split == "train" else 999),
        )

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, drop_last=True,
    )


# ============================================================
# 学习率调度 / Learning Rate Scheduling
# ============================================================

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
) -> float:
    """
    计算当前步的学习率 / Compute learning rate for current step

    支持三种调度策略 / Supports three scheduling strategies:
      1. cosine: 余弦退火 / Cosine annealing
      2. linear: 线性衰减 / Linear decay
      3. constant: 恒定 (仅预热) / Constant (warmup only)

    Args:
        step: 当前训练步 / Current training step
        warmup_steps: 预热步数 / Warmup steps
        max_steps: 最大步数 / Max steps
        base_lr: 基础学习率 / Base learning rate
        min_lr_ratio: 最小LR比例 / Min LR ratio
        schedule: 调度类型 / Schedule type

    Returns:
        当前学习率 / Current learning rate
    """
    min_lr = base_lr * min_lr_ratio

    # 预热阶段 / Warmup phase
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    # 衰减阶段 / Decay phase
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)

    if schedule == "cosine":
        # 余弦退火 / Cosine annealing
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == "linear":
        # 线性衰减 / Linear decay
        return min_lr + (base_lr - min_lr) * (1 - progress)
    else:
        # 恒定 / Constant
        return base_lr


# ============================================================
# 评估函数 / Evaluation Function
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_steps: int = 100,
) -> Dict[str, float]:
    """
    评估模型 / Evaluate model

    计算验证集上的平均损失和困惑度。
    Computes average loss and perplexity on validation set.

    Args:
        model: 待评估模型 / Model to evaluate
        data_loader: 验证数据加载器 / Validation data loader
        device: 设备 / Device
        max_steps: 最大评估步数 / Max evaluation steps

    Returns:
        dict: {'loss': ..., 'perplexity': ...}
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for step, (input_ids, target_ids) in enumerate(data_loader):
        if step >= max_steps:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += target_ids.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # 防止溢出 / Prevent overflow

    model.train()
    return {"loss": avg_loss, "perplexity": perplexity}


# ============================================================
# 检查点 / Checkpointing
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """
    保存训练检查点 / Save training checkpoint

    Args:
        model: 模型 / Model
        optimizer: 优化器 / Optimizer
        step: 当前步数 / Current step
        loss: 当前损失 / Current loss
        checkpoint_dir: 检查点目录 / Checkpoint directory
        is_best: 是否为最佳模型 / Whether this is the best model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    # 保存最新检查点 / Save latest checkpoint
    path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(state, path)

    # 按步数保存 / Save by step number
    path_step = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
    torch.save(state, path_step)

    # 保存最佳模型 / Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    加载检查点 / Load checkpoint

    Args:
        checkpoint_path: 检查点文件路径 / Checkpoint file path
        model: 模型 / Model
        optimizer: 优化器 (可选) / Optimizer (optional)
        device: 设备 / Device

    Returns:
        dict: 检查点元数据 / Checkpoint metadata
    """
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return {"step": state.get("step", 0), "loss": state.get("loss", float("inf"))}


# ============================================================
# 训练门控监控 / Gate Value Monitoring
# ============================================================

@torch.no_grad()
def log_gate_stats(model: DDLModel, logger: logging.Logger, step: int):
    """
    记录DDL门控统计信息 / Log DDL gate statistics

    监控训练过程中门控值的变化，帮助理解DDL的行为。
    Monitor gate value changes during training to understand DDL behavior.

    预期行为 / Expected behavior:
      - 训练初期: 门控值接近0.5 (初始值) / Near 0.5 initially
      - 训练中: 不同层的门控应分化 / Gates should differentiate across layers
      - 如果所有门控→0: DDL退化为标准残差 / DDL degenerates to standard residual
      - 如果所有门控→1: 过度重写，可能不稳定 / Over-rewriting, may be unstable
    """
    gate_stats = model.get_all_gate_stats()
    for i, stats in enumerate(gate_stats):
        logger.info(
            f"  Gate L{i:02d} | attn: {stats['attn_gate_magnitude']:.4f} | "
            f"ffn: {stats['ffn_gate_magnitude']:.4f}"
        )


# ============================================================
# 主训练函数 / Main Training Function
# ============================================================

def parse_args():
    """解析命令行参数 / Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="DDL Language Model Training / DDL语言模型训练"
    )

    # 配置文件 / Config file
    parser.add_argument("--config", type=str, default=None,
                        help="YAML配置文件路径 / YAML config file path")

    # 模型参数 / Model parameters
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gate-init", type=float, default=0.0)

    # 训练参数 / Training parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", "--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"])
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)

    # 数据 / Data
    parser.add_argument("--data-source", type=str, default="synthetic",
                        choices=["synthetic", "wikitext2"])
    parser.add_argument("--num-workers", type=int, default=0)

    # 日志与保存 / Logging & saving
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")

    # 恢复训练 / Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复 / Resume from checkpoint path")

    # 模型类型 / Model type
    parser.add_argument("--model-type", type=str, default="ddl",
                        choices=["ddl", "standard"],
                        help="模型类型: ddl 或 standard (对比用)")

    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict:
    """
    加载YAML配置文件 / Load YAML config file

    如果没有安装PyYAML，则返回空字典（使用命令行参数）。
    Returns empty dict if PyYAML is not installed (uses CLI args).
    """
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Warning: PyYAML not installed, using command-line args only.")
        print("Install with: pip install pyyaml")
        return {}
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults.")
        return {}


def apply_config(args, config: Dict):
    """
    将YAML配置应用到args / Apply YAML config to args

    命令行参数优先级高于YAML配置。
    Command-line arguments take priority over YAML config.
    """
    if not config:
        return args

    # 模型配置 / Model config
    model_cfg = config.get("model", {})
    if model_cfg:
        if args.d_model == 256:  # 默认值 → 使用配置
            args.d_model = model_cfg.get("d_model", args.d_model)
        if args.vocab_size == 10000:
            args.vocab_size = model_cfg.get("vocab_size", args.vocab_size)
        if args.n_layers == 6:
            args.n_layers = model_cfg.get("n_layers", args.n_layers)
        if args.n_heads == 4:
            args.n_heads = model_cfg.get("n_heads", args.n_heads)
        if args.d_ff is None:
            args.d_ff = model_cfg.get("d_ff", args.d_ff)
        if args.max_seq_len == 512:
            args.max_seq_len = model_cfg.get("max_seq_len", args.max_seq_len)
        if args.dropout == 0.1:
            args.dropout = model_cfg.get("dropout", args.dropout)
        if args.gate_init == 0.0:
            args.gate_init = model_cfg.get("gate_init", args.gate_init)

    # 训练配置 / Training config
    train_cfg = config.get("training", {})
    if train_cfg:
        if args.batch_size == 32:
            args.batch_size = train_cfg.get("batch_size", args.batch_size)
        if args.lr == 3e-4:
            args.lr = train_cfg.get("learning_rate", args.lr)
        if args.weight_decay == 0.01:
            args.weight_decay = train_cfg.get("weight_decay", args.weight_decay)
        if args.warmup_steps == 1000:
            args.warmup_steps = train_cfg.get("warmup_steps", args.warmup_steps)
        if args.max_steps == 50000:
            args.max_steps = train_cfg.get("max_steps", args.max_steps)
        if args.grad_clip == 1.0:
            args.grad_clip = train_cfg.get("grad_clip", args.grad_clip)
        if args.seed == 42:
            args.seed = train_cfg.get("seed", args.seed)
        if args.lr_scheduler == "cosine":
            args.lr_scheduler = train_cfg.get("lr_scheduler", args.lr_scheduler)
        if args.min_lr_ratio == 0.1:
            args.min_lr_ratio = train_cfg.get("min_lr_ratio", args.min_lr_ratio)
        if args.data_source == "synthetic":
            args.data_source = train_cfg.get("data_source", args.data_source)

    # 日志配置 / Logging config
    log_cfg = config.get("logging", {})
    if log_cfg:
        if args.log_interval == 50:
            args.log_interval = log_cfg.get("log_interval", args.log_interval)
        if args.eval_interval == 500:
            args.eval_interval = log_cfg.get("eval_interval", args.eval_interval)
        if args.save_interval == 2000:
            args.save_interval = log_cfg.get("save_interval", args.save_interval)
        if args.checkpoint_dir == "checkpoints":
            args.checkpoint_dir = log_cfg.get("checkpoint_dir", args.checkpoint_dir)
        if args.log_dir == "logs":
            args.log_dir = log_cfg.get("log_dir", args.log_dir)

    return args


def train():
    """
    主训练函数 / Main training function

    完整的DDL模型训练流程 / Complete DDL model training pipeline:
      1. 解析参数 / Parse arguments
      2. 设置随机种子 / Set random seeds
      3. 创建模型和数据 / Create model and data
      4. 训练循环 / Training loop
      5. 定期评估 / Periodic evaluation
      6. 保存检查点 / Save checkpoints
    """
    args = parse_args()

    # 加载YAML配置 / Load YAML config
    if args.config:
        config = load_yaml_config(args.config)
        args = apply_config(args, config)

    # 设置随机种子 / Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备 / Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志 / Logging
    logger = setup_logging(args.log_dir)
    logger.info("=" * 60)
    logger.info("Deep Delta Learning (DDL) Training")
    logger.info("=" * 60)

    # 打印配置 / Print configuration
    logger.info(f"Device: {device}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Config: vocab_size={args.vocab_size}, d_model={args.d_model}, "
                f"n_layers={args.n_layers}, n_heads={args.n_heads}")
    logger.info(f"Training: batch_size={args.batch_size}, lr={args.lr}, "
                f"max_steps={args.max_steps}, grad_clip={args.grad_clip}")
    logger.info(f"LR schedule: {args.lr_scheduler}, warmup={args.warmup_steps}")
    logger.info(f"Data source: {args.data_source}")

    # 创建模型 / Create model
    if args.model_type == "ddl":
        model = DDLModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            gate_init=args.gate_init,
        ).to(device)
    else:
        model = StandardTransformer(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        ).to(device)

    param_count = count_params(model)
    logger.info(f"Model parameters: {param_count:,}")

    # 创建数据加载器 / Create data loaders
    train_loader = create_data_loader(
        data_source=args.data_source,
        vocab_size=args.vocab_size,
        seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="train",
        seed=args.seed,
    )
    val_loader = create_data_loader(
        data_source=args.data_source,
        vocab_size=args.vocab_size,
        seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="val",
        seed=args.seed,
    )
    logger.info(f"Train samples: {len(train_loader.dataset):,}")
    logger.info(f"Val samples:   {len(val_loader.dataset):,}")

    # 优化器 / Optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # 恢复训练 / Resume training
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt_info = load_checkpoint(args.resume, model, optimizer, device)
        start_step = ckpt_info["step"]
        logger.info(f"Resumed from step {start_step}, loss={ckpt_info['loss']:.4f}")

    # 训练循环 / Training loop
    model.train()
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_tokens = 0
    start_time = time.time()

    logger.info(f"\nStarting training from step {start_step}...")
    logger.info("-" * 60)

    for step in range(start_step, args.max_steps):
        # 获取下一个batch / Get next batch
        try:
            input_ids, target_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, target_ids = next(train_iter)

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 更新学习率 / Update learning rate
        current_lr = get_lr(
            step, args.warmup_steps, args.max_steps,
            args.lr, args.min_lr_ratio, args.lr_scheduler,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # 前向传播 / Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
        )

        # 反向传播 / Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 / Gradient clipping
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
        else:
            grad_norm = torch.tensor(0.0)

        # 优化器步骤 / Optimizer step
        optimizer.step()

        # 累积统计 / Accumulate statistics
        running_loss += loss.item() * target_ids.numel()
        running_tokens += target_ids.numel()

        # 日志 / Logging
        if (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / max(running_tokens, 1)
            perplexity = math.exp(min(avg_loss, 20))
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1 - start_step) / max(elapsed, 1)

            logger.info(
                f"Step {step + 1:>6d}/{args.max_steps} | "
                f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                f"LR: {current_lr:.2e} | Grad: {grad_norm:.2f} | "
                f"Steps/s: {steps_per_sec:.1f}"
            )

            running_loss = 0.0
            running_tokens = 0

        # 评估 / Evaluation
        if (step + 1) % args.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device)
            logger.info(
                f"  [EVAL] Step {step + 1} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val PPL: {val_metrics['perplexity']:.2f}"
            )

            # DDL门控监控 / DDL gate monitoring
            if args.model_type == "ddl" and isinstance(model, DDLModel):
                logger.info("  [GATE STATS]")
                log_gate_stats(model, logger, step + 1)

            # 保存最佳模型 / Save best model
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                logger.info(f"  *** New best val loss: {best_val_loss:.4f} ***")

            # 保存检查点 / Save checkpoint
            save_checkpoint(
                model, optimizer, step + 1, val_metrics["loss"],
                args.checkpoint_dir, is_best=is_best,
            )

        # 定期保存 / Periodic saving
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, step + 1, loss.item(),
                args.checkpoint_dir,
            )

    # 最终评估 / Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete / 训练完成")
    logger.info("=" * 60)

    final_metrics = evaluate(model, val_loader, device)
    logger.info(
        f"Final Val Loss: {final_metrics['loss']:.4f} | "
        f"Final Val PPL: {final_metrics['perplexity']:.2f}"
    )

    # 最终门控统计 / Final gate statistics
    if args.model_type == "ddl" and isinstance(model, DDLModel):
        logger.info("\nFinal Gate Statistics / 最终门控统计:")
        log_gate_stats(model, logger, args.max_steps)

    # 保存最终模型 / Save final model
    save_checkpoint(
        model, optimizer, args.max_steps, final_metrics["loss"],
        args.checkpoint_dir,
    )
    logger.info(f"Final checkpoint saved to {args.checkpoint_dir}")

    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time / 60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
