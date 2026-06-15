"""
CLIP 完整训练脚本 / CLIP Full Training Script
==============================================

用法 / Usage:
    python train.py --config configs/default.yaml
    python train.py --batch_size 128 --learning_rate 1e-4 --max_steps 50000

论文参考 / Paper Reference:
  Radford et al., "Learning Transferable Visual Models From Natural
  Language Supervision", ICML 2021.

功能 / Features:
  - 从YAML配置文件加载参数 / Load params from YAML config file
  - 余弦退火学习率调度 / Cosine annealing learning rate schedule
  - 线性预热 / Linear warmup
  - 梯度裁剪 / Gradient clipping
  - 检查点保存与恢复 / Checkpoint saving and resuming
  - 训练日志 (loss, accuracy, temperature, grad norm) / Training logs
  - 支持模拟数据 (无需真实数据集即可运行) / Supports mock data (runs without real dataset)
"""

import os
import sys
import argparse
import logging
import time
import math
import json
import torch
import torch.nn as nn
from typing import Dict, Optional

# 添加当前目录到路径以便导入 / Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SimpleCLIP
from data import ImageTextDataset, SimpleBPETokenizer, build_transforms
from loss import ClipLoss
from loss.clip_loss import analyze_gradients


# ============================================================
# 日志配置 / Logging Configuration
# ============================================================

def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    配置日志系统 / Configure logging system.

    同时输出到控制台和文件 / Output to both console and file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    # 配置根日志器 / Configure root logger
    logger = logging.getLogger("CLIP_Train")
    logger.setLevel(logging.INFO)

    # 格式 / Format
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台handler / Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件handler / File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# 学习率调度 / Learning Rate Scheduling
# ============================================================

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """
    带线性预热的余弦退火学习率。
    Cosine annealing learning rate with linear warmup.

    原理 / Rationale:
      1. 预热阶段: 线性增加学习率从0到base_lr
         Warmup phase: linearly increase LR from 0 to base_lr
      2. 退火阶段: 余弦衰减从base_lr到min_lr
         Annealing phase: cosine decay from base_lr to min_lr

      预热有助于训练初期的稳定性，余弦退火有助于后期精细收敛。
      Warmup helps stability in early training; cosine annealing helps
      fine convergence in later stages.

    Args:
        step:         当前步数 / Current step.
        warmup_steps: 预热步数 / Warmup steps.
        max_steps:    最大步数 / Maximum steps.
        base_lr:      基础学习率 / Base learning rate.
        min_lr:       最小学习率 / Minimum learning rate.

    Returns:
        当前学习率 / Current learning rate.
    """
    if step < warmup_steps:
        # 线性预热 / Linear warmup
        return base_lr * step / max(warmup_steps, 1)
    else:
        # 余弦退火 / Cosine annealing
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================================================
# 检查点保存 / Checkpoint Saving
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    checkpoint_dir: str,
    experiment_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    保存训练检查点 / Save training checkpoint.

    Args:
        model:          模型 / Model.
        optimizer:      优化器 / Optimizer.
        step:           当前步数 / Current step.
        loss:           最近损失值 / Recent loss value.
        checkpoint_dir: 检查点目录 / Checkpoint directory.
        experiment_name: 实验名称 / Experiment name.
        logger:         日志器 / Logger.

    Returns:
        保存的文件路径 / Saved file path.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = f"{experiment_name}_step{step}.pt"
    filepath = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)

    if logger:
        logger.info(f"检查点已保存 / Checkpoint saved: {filepath}")

    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    """
    加载检查点 / Load checkpoint.

    Returns:
        恢复的步数 / Resumed step number.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step", 0)


# ============================================================
# 配置加载 / Configuration Loading
# ============================================================

def load_yaml_config(config_path: str) -> dict:
    """
    加载YAML配置文件 / Load YAML config file.

    使用简单解析 (不依赖PyYAML库)。
    Uses simple parsing (no PyYAML dependency required).
    """
    config = {}
    current_section = None

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        pass

    # 简单YAML解析 (无PyYAML时的后备方案)
    # Simple YAML parser (fallback when PyYAML is unavailable)
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.startswith("#"):
                continue
            # 检测section header / Detect section header
            if not stripped.startswith(" ") and stripped.endswith(":"):
                current_section = stripped[:-1].strip()
                config[current_section] = {}
                continue
            # 键值对 / Key-value pair
            if ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.strip().split("#")[0].strip()  # 去除行内注释 / Remove inline comment

                # 类型转换 / Type conversion
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.startswith("[") and val.endswith("]"):
                    val = [float(x.strip()) for x in val[1:-1].split(",")]
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            val = val.strip('"').strip("'")

                if current_section:
                    config[current_section][key] = val
                else:
                    config[key] = val

    return config


# ============================================================
# 主训练循环 / Main Training Loop
# ============================================================

def parse_args():
    """解析命令行参数 / Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CLIP 训练脚本 / CLIP Training Script"
    )

    # 配置文件 / Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML配置文件路径 / Path to YAML config file",
    )

    # 模型参数 / Model parameters
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--image_depth", type=int, default=2)
    parser.add_argument("--text_depth", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=32)

    # 训练参数 / Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--temperature_init", type=float, default=0.07)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # 数据参数 / Data parameters
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--ann_file", type=str, default="./data/annotations.json")
    parser.add_argument("--num_workers", type=int, default=0)

    # 输出参数 / Output parameters
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--experiment_name", type=str, default="clip_train")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)

    # 恢复训练 / Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="检查点文件路径 / Checkpoint file path to resume from")

    return parser.parse_args()


def main():
    """主训练函数 / Main training function."""
    args = parse_args()

    # 如果提供了配置文件，从文件加载参数
    # If config file provided, load parameters from it
    if args.config is not None and os.path.exists(args.config):
        config = load_yaml_config(args.config)

        # 从配置文件覆盖参数 / Override args from config file
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        output_cfg = config.get("output", {})

        args.image_size = model_cfg.get("image_size", args.image_size)
        args.patch_size = model_cfg.get("patch_size", args.patch_size)
        args.embed_dim = model_cfg.get("embed_dim", args.embed_dim)
        args.projection_dim = model_cfg.get("projection_dim", args.projection_dim)
        args.num_heads = model_cfg.get("num_heads", args.num_heads)
        args.image_depth = model_cfg.get("image_depth", args.image_depth)
        args.text_depth = model_cfg.get("text_depth", args.text_depth)
        args.vocab_size = model_cfg.get("vocab_size", args.vocab_size)
        args.max_seq_len = model_cfg.get("max_seq_len", args.max_seq_len)

        args.batch_size = train_cfg.get("batch_size", args.batch_size)
        args.learning_rate = train_cfg.get("learning_rate", args.learning_rate)
        args.weight_decay = train_cfg.get("weight_decay", args.weight_decay)
        args.warmup_steps = train_cfg.get("warmup_steps", args.warmup_steps)
        args.max_steps = train_cfg.get("max_steps", args.max_steps)
        args.temperature_init = train_cfg.get("temperature_init", args.temperature_init)
        args.grad_clip_norm = train_cfg.get("grad_clip_norm", args.grad_clip_norm)
        args.log_every = train_cfg.get("log_every", args.log_every)
        args.save_every = train_cfg.get("save_every", args.save_every)

        args.data_root = data_cfg.get("data_root", args.data_root)
        args.ann_file = data_cfg.get("ann_file", args.ann_file)
        args.num_workers = data_cfg.get("num_workers", args.num_workers)

        args.checkpoint_dir = output_cfg.get("checkpoint_dir", args.checkpoint_dir)
        args.log_dir = output_cfg.get("log_dir", args.log_dir)
        args.experiment_name = output_cfg.get("experiment_name", args.experiment_name)

    # ---- 设备 / Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 日志 / Logging ----
    logger = setup_logging(args.log_dir, args.experiment_name)
    logger.info("=" * 60)
    logger.info("CLIP 训练开始 / CLIP Training Started")
    logger.info("=" * 60)
    logger.info(f"设备 / Device: {device}")
    logger.info(f"参数 / Args: {vars(args)}")

    # ---- 模型 / Model ----
    model = SimpleCLIP(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        projection_dim=args.projection_dim,
        num_heads=args.num_heads,
        image_depth=args.image_depth,
        text_depth=args.text_depth,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        temperature_init=args.temperature_init,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量 / Total parameters: {total_params:,}")

    # ---- 优化器 / Optimizer ----
    # AdamW: Adam + 解耦权重衰减 / AdamW: Adam + decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    # ---- 损失函数 / Loss function ----
    criterion = ClipLoss()

    # ---- 数据 / Data ----
    # 使用简化分词器和模拟数据集
    # Use simplified tokenizer and mock dataset
    tokenizer = SimpleBPETokenizer(vocab_size=args.vocab_size)
    transform = build_transforms(image_size=args.image_size, is_train=True)

    # 如果transform不可用 (缺少torchvision)，手动创建tensor变换
    # If transform unavailable (missing torchvision), create manual tensor transform
    dataset = ImageTextDataset(
        data_root=args.data_root,
        ann_file=args.ann_file,
        transform=None,  # 模拟数据不需要真实变换 / No real transform for mock data
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        dataset_type="coco",
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"数据集大小 / Dataset size: {len(dataset)}")
    logger.info(f"批次大小 / Batch size: {args.batch_size}")
    logger.info(f"最大步数 / Max steps: {args.max_steps}")

    # ---- 恢复训练 / Resume training ----
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, model, optimizer, device=str(device))
        logger.info(f"从步 {start_step} 恢复训练 / Resumed training from step {start_step}")

    # ---- 训练循环 / Training Loop ----
    model.train()
    step = start_step
    running_loss = 0.0
    data_iter = iter(dataloader)

    logger.info("开始训练循环 / Starting training loop...")

    while step < args.max_steps:
        # 获取下一批数据 (如果迭代器耗尽则重新开始)
        # Get next batch (restart iterator if exhausted)
        try:
            images, text_ids = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, text_ids = next(data_iter)

        # 确保图像尺寸正确 / Ensure correct image dimensions
        if images.shape[-1] != args.image_size or images.shape[-2] != args.image_size:
            # 模拟数据可能尺寸不匹配，使用随机数据
            # Mock data may have wrong dimensions, use random data
            images = torch.randn(args.batch_size, 3, args.image_size, args.image_size)

        images = images.to(device)
        text_ids = text_ids.to(device)

        # 更新学习率 / Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 前向传播 / Forward pass
        loss, logits_i, logits_t = model(images, text_ids)

        # 反向传播 / Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 / Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        # 更新参数 / Update parameters
        optimizer.step()

        running_loss += loss.item()
        step += 1

        # ---- 日志记录 / Logging ----
        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            running_loss = 0.0

            # 计算批内准确率 / Compute in-batch accuracy
            with torch.no_grad():
                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device)
                i2t_acc = (logits_i.argmax(dim=1) == labels).float().mean().item()
                t2i_acc = (logits_t.argmax(dim=1) == labels).float().mean().item()

            # 温度 / Temperature
            temperature = model.get_temperature()

            # 梯度分析 / Gradient analysis
            grad_stats = analyze_gradients(model)

            logger.info(
                f"Step {step}/{args.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"I2T Acc: {i2t_acc:.3f} | "
                f"T2I Acc: {t2i_acc:.3f} | "
                f"LR: {lr:.6f} | "
                f"Temp: {temperature:.4f} | "
                f"Grad Norm: {grad_stats['total_grad_norm']:.4f}"
            )

        # ---- 保存检查点 / Save checkpoint ----
        if step % args.save_every == 0:
            save_checkpoint(
                model, optimizer, step, loss.item(),
                args.checkpoint_dir, args.experiment_name, logger,
            )

    # ---- 最终保存 / Final save ----
    final_path = save_checkpoint(
        model, optimizer, step, loss.item(),
        args.checkpoint_dir, f"{args.experiment_name}_final", logger,
    )

    logger.info("=" * 60)
    logger.info("训练完成 / Training Complete")
    logger.info(f"最终检查点 / Final checkpoint: {final_path}")
    logger.info(f"最终损失 / Final loss: {loss.item():.4f}")
    logger.info(f"最终温度 / Final temperature: {model.get_temperature():.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
