"""
CLIP 评估脚本 — 零样本分类 + 图文检索
CLIP Evaluation Script — Zero-Shot Classification + Image-Text Retrieval
=========================================================================

用法 / Usage:
    python evaluate.py --checkpoint checkpoints/clip_train_final_step1000.pt
    python evaluate.py  # 使用随机权重进行演示 (不需要预训练)
                        # Demo with random weights (no pretraining needed)

评估指标 / Evaluation Metrics:
  1. 零样本分类 / Zero-Shot Classification:
     - Top-1 准确率 / Top-1 accuracy
     - Top-5 准确率 / Top-5 accuracy
     - 使用 ImageNet 风格的标签和 prompt 模板
       Uses ImageNet-style labels and prompt templates

  2. 图文检索 / Image-Text Retrieval:
     - Recall@1, Recall@5, Recall@10 (图像→文本 / Image->Text)
     - Recall@1, Recall@5, Recall@10 (文本→图像 / Text->Image)
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

# 添加当前目录到路径以便导入 / Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SimpleCLIP
from data import SimpleBPETokenizer
from loss.clip_loss import compute_retrieval_metrics


# ============================================================
# ImageNet 风格标签 / ImageNet-Style Labels
# ============================================================

# 100个常见物体类别 (简化版) / 100 common object categories (simplified)
IMAGENET_LABELS = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
    "robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    "fire salamander", "smooth newt", "eft", "spotted salamander", "axolotl",
    "bullfrog", "tree frog", "tailed frog", "loggerhead turtle", "leatherback turtle",
    "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
    "American chameleon", "whiptail lizard", "agama", "frilled lizard", "alligator lizard",
    "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
    "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake",
    "green snake", "king snake", "garter snake", "water snake", "vine snake",
    "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba",
    "sea snake", "horned viper", "diamondback rattlesnake", "sidewinder rattlesnake", "trilobite",
    "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider",
    "black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock",
    "quail", "partridge", "African grey parrot", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
    "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
]

# Prompt模板列表 / Prompt template list
# 论文中使用多种模板取平均以提高零样本性能
# Paper uses multiple templates averaged to improve zero-shot performance
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image of a {}.",
    "a photo of the {}.",
    "this is a photo of a {}.",
]


# ============================================================
# 零样本分类评估 / Zero-Shot Classification Evaluation
# ============================================================

@torch.no_grad()
def evaluate_zero_shot(
    model: SimpleCLIP,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    class_names: List[str],
    tokenizer: SimpleBPETokenizer,
    prompt_templates: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """
    零样本分类评估 / Zero-shot classification evaluation.

    原理 / Rationale:
      1. 为每个类别生成多种prompt文本 / Generate multiple prompt texts for each class
      2. 编码所有类别的文本特征 / Encode text features for all classes
      3. 对每种模板的特征取平均 / Average features across templates
      4. 计算图像与各类别的相似度 / Compute similarity between images and classes
      5. 预测最高相似度的类别 / Predict the class with highest similarity

    Args:
        model:            CLIP模型 / CLIP model.
        test_images:      [N, 3, H, W] 测试图像 / Test images.
        test_labels:      [N] 真实标签 (类别索引) / Ground truth labels (class indices).
        class_names:      类别名称列表 / List of class names.
        tokenizer:        分词器 / Tokenizer.
        prompt_templates: prompt模板列表 / List of prompt templates.
        device:           设备 / Device.

    Returns:
        dict: {"top1_acc": float, "top5_acc": float, "num_classes": int}
    """
    model.eval()
    num_classes = len(class_names)
    max_seq_len = model.text_encoder.max_seq_len

    # 为每种模板编码所有类别 / Encode all classes for each template
    all_text_features = []

    for template in prompt_templates:
        # 构建当前模板的所有类别文本 / Build all class texts for current template
        class_texts = []
        for class_name in class_names:
            text = template.format(class_name)
            ids = tokenizer.encode(text, max_len=max_seq_len)
            class_texts.append(ids)

        # [num_classes, seq_len]
        class_ids = torch.tensor(class_texts, dtype=torch.long, device=device)

        # 编码文本 / Encode text
        text_feats = model.encode_text(class_ids)  # [num_classes, D]

        # L2归一化 / L2 normalize
        text_feats = F.normalize(text_feats, dim=-1)
        all_text_features.append(text_feats)

    # 对所有模板取平均并重新归一化
    # Average across all templates and re-normalize
    # 这是论文中的关键技巧: 多种prompt模板集成能显著提升性能
    # Key trick from the paper: ensemble over multiple prompt templates
    avg_text_features = torch.stack(all_text_features, dim=0).mean(dim=0)  # [C, D]
    avg_text_features = F.normalize(avg_text_features, dim=-1)

    # 编码测试图像 / Encode test images
    image_features = model.encode_image(test_images.to(device))  # [N, D]

    # 计算相似度 / Compute similarity
    similarity = image_features @ avg_text_features.T  # [N, C]

    # Top-1 准确率 / Top-1 accuracy
    predictions = similarity.argmax(dim=1)  # [N]
    top1_correct = (predictions == test_labels.to(device)).float().sum().item()
    top1_acc = top1_correct / len(test_labels)

    # Top-5 准确率 / Top-5 accuracy
    k = min(5, num_classes)
    _, top5_indices = similarity.topk(k, dim=1)  # [N, 5]
    top5_correct = 0
    for i in range(len(test_labels)):
        if test_labels[i].item() in top5_indices[i].tolist():
            top5_correct += 1
    top5_acc = top5_correct / len(test_labels)

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "num_classes": num_classes,
        "num_samples": len(test_labels),
    }


# ============================================================
# 图文检索评估 / Image-Text Retrieval Evaluation
# ============================================================

@torch.no_grad()
def evaluate_retrieval(
    model: SimpleCLIP,
    images: torch.Tensor,
    text_ids: torch.Tensor,
    device: torch.device,
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    图文检索评估 / Image-text retrieval evaluation.

    原理 / Rationale:
      假设第 i 张图像和第 i 条文本是配对的正样本。
      计算所有图像与所有文本之间的相似度矩阵，
      然后检查每个查询的正确匹配是否在 top-K 中。

      Assumes image i and text i are paired positive samples.
      Computes all-pairs similarity matrix, then checks if the
      correct match is in the top-K for each query.

    指标 / Metrics:
      - I2T R@K: 给定图像，检索正确文本 / Given image, retrieve correct text
      - T2I R@K: 给定文本，检索正确图像 / Given text, retrieve correct image

    Args:
        model:    CLIP模型 / CLIP model.
        images:   [N, 3, H, W] 图像 / Images.
        text_ids: [N, seq_len] 文本 / Texts.
        device:   设备 / Device.
        k_values: K值列表 / K values.

    Returns:
        dict: {"I2T_R@1": ..., "I2T_R@5": ..., "T2I_R@1": ..., ...}
    """
    model.eval()

    # 编码图像和文本 / Encode images and text
    image_features = model.encode_image(images.to(device))  # [N, D]
    text_features = model.encode_text(text_ids.to(device))  # [N, D]

    # 相似度矩阵 / Similarity matrix
    sim_matrix = image_features @ text_features.T  # [N, N]

    results = {}

    # 图像→文本检索 / Image->Text retrieval
    i2t_metrics = compute_retrieval_metrics(sim_matrix, k_values)
    for k, v in i2t_metrics.items():
        results[f"I2T_{k}"] = v

    # 文本→图像检索 / Text->Image retrieval (转置相似度矩阵)
    t2i_metrics = compute_retrieval_metrics(sim_matrix.T, k_values)
    for k, v in t2i_metrics.items():
        results[f"T2I_{k}"] = v

    return results


# ============================================================
# 主评估函数 / Main Evaluation Function
# ============================================================

def parse_args():
    """解析命令行参数 / Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CLIP 评估脚本 / CLIP Evaluation Script"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型检查点路径 / Model checkpoint path",
    )
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--image_depth", type=int, default=2)
    parser.add_argument("--text_depth", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=10)

    return parser.parse_args()


def main():
    """主评估函数 / Main evaluation function."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("CLIP 评估 / CLIP Evaluation")
    print("=" * 60)
    print(f"设备 / Device: {device}")

    # ---- 构建模型 / Build model ----
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
    ).to(device)

    # 加载检查点 (如果有) / Load checkpoint if available
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"已加载检查点 / Loaded checkpoint: {args.checkpoint}")
        print(f"训练步数 / Training step: {checkpoint.get('step', 'unknown')}")
    else:
        print("未提供检查点，使用随机权重进行演示")
        print("No checkpoint provided, demo with random weights")

    model.eval()

    # ---- 分词器 / Tokenizer ----
    tokenizer = SimpleBPETokenizer(vocab_size=args.vocab_size)

    # ---- 零样本分类评估 / Zero-shot classification evaluation ----
    print("\n" + "-" * 60)
    print("零样本分类评估 / Zero-Shot Classification Evaluation")
    print("-" * 60)

    num_classes = min(args.num_classes, len(IMAGENET_LABELS))
    class_names = IMAGENET_LABELS[:num_classes]
    num_test = args.num_test_samples

    # 生成模拟测试数据 / Generate mock test data
    test_images = torch.randn(num_test, 3, args.image_size, args.image_size, device=device)
    test_labels = torch.randint(0, num_classes, (num_test,), device=device)

    zsl_results = evaluate_zero_shot(
        model=model,
        test_images=test_images,
        test_labels=test_labels,
        class_names=class_names,
        tokenizer=tokenizer,
        prompt_templates=PROMPT_TEMPLATES,
        device=device,
    )

    print(f"  类别数 / Number of classes: {zsl_results['num_classes']}")
    print(f"  测试样本数 / Number of test samples: {zsl_results['num_samples']}")
    print(f"  Top-1 准确率 / Top-1 Accuracy: {zsl_results['top1_acc']:.4f}")
    print(f"  Top-5 准确率 / Top-5 Accuracy: {zsl_results['top5_acc']:.4f}")

    # 随机基线参考 / Random baseline reference
    random_top1 = 1.0 / num_classes
    random_top5 = min(5.0 / num_classes, 1.0)
    print(f"  随机基线 Top-1 / Random baseline Top-1: {random_top1:.4f}")
    print(f"  随机基线 Top-5 / Random baseline Top-5: {random_top5:.4f}")

    # ---- 图文检索评估 / Image-text retrieval evaluation ----
    print("\n" + "-" * 60)
    print("图文检索评估 / Image-Text Retrieval Evaluation")
    print("-" * 60)

    num_retrieval = min(num_test, 50)  # 检索评估使用较少样本 / Use fewer samples for retrieval
    retrieval_images = torch.randn(num_retrieval, 3, args.image_size, args.image_size, device=device)

    # 生成对应的文本 / Generate corresponding texts
    captions = [
        f"a photo of {class_names[i % num_classes]}"
        for i in range(num_retrieval)
    ]
    retrieval_text_ids = []
    for cap in captions:
        ids = tokenizer.encode(cap, max_len=args.max_seq_len)
        retrieval_text_ids.append(ids)
    retrieval_text_ids = torch.tensor(retrieval_text_ids, dtype=torch.long, device=device)

    retrieval_results = evaluate_retrieval(
        model=model,
        images=retrieval_images,
        text_ids=retrieval_text_ids,
        device=device,
        k_values=(1, 5, 10),
    )

    for metric_name, value in sorted(retrieval_results.items()):
        print(f"  {metric_name}: {value:.4f}")

    # 随机基线 / Random baselines
    random_r1 = 1.0 / num_retrieval
    print(f"\n  随机基线 R@1 / Random baseline R@1: {random_r1:.4f}")

    # ---- 模型统计 / Model Statistics ----
    print("\n" + "-" * 60)
    print("模型统计 / Model Statistics")
    print("-" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    img_params = sum(p.numel() for p in model.image_encoder.parameters())
    txt_params = sum(p.numel() for p in model.text_encoder.parameters())
    proj_params = total_params - img_params - txt_params

    print(f"  总参数量 / Total parameters: {total_params:,}")
    print(f"  图像编码器 / Image encoder:   {img_params:,}")
    print(f"  文本编码器 / Text encoder:    {txt_params:,}")
    print(f"  投影头 / Projection heads:    {proj_params:,}")
    print(f"  温度 / Temperature: {model.get_temperature():.4f}")

    print("\n" + "=" * 60)
    print("评估完成 / Evaluation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
