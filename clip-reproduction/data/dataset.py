"""
CLIP 图像-文本对数据集 + 简化BPE分词器
CLIP Image-Text Pair Dataset + Simplified BPE Tokenizer
========================================================

支持两种数据格式 / Supports two data formats:
  1. COCO风格: JSON标注文件 + 图片目录 / COCO-style: JSON annotation file + image directory
  2. 自定义格式: 每行一条 "image_path\\tcaption" 的TSV文件
     Custom format: TSV file with "image_path\\tcaption" per line

分词器 / Tokenizer:
  简化的字节对编码 (BPE) 分词器，用于将文本转换为token ID序列。
  Simplified Byte-Pair Encoding (BPE) tokenizer for converting text
  to token ID sequences.
"""

import os
import json
import torch
import math
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import Counter
import re


# ============================================================
# 简化BPE分词器 / Simplified BPE Tokenizer
# ============================================================

class SimpleBPETokenizer:
    """
    简化的字节对编码分词器。
    Simplified Byte-Pair Encoding tokenizer.

    原理 / Rationale:
      原始CLIP使用49408词表的BPE分词器。这里实现一个简化版本，
      核心流程相同: 字节级切分 → 迭代合并最频繁对 → 生成词汇表。

      Original CLIP uses a BPE tokenizer with vocab size 49408. Here we
      implement a simplified version with the same core pipeline:
      byte-level split -> iteratively merge most frequent pairs -> build vocabulary.

    特殊token / Special tokens:
      - SOS (id=0): 序列起始 / Start of sequence
      - EOS (id=1): 序列结束 / End of sequence
      - PAD (id=2): 填充 / Padding
      - UNK (id=3): 未知 / Unknown

    Args:
        vocab_size (int): 最大词表大小 / Maximum vocabulary size.
    """

    # 特殊token IDs / Special token IDs
    SOS_ID = 0
    EOS_ID = 1
    PAD_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_size: int = 49408):
        self.vocab_size = vocab_size

        # 初始词汇表: 特殊token + 单字节token (共 256 + 4 = 260)
        # Initial vocabulary: special tokens + single-byte tokens (256 + 4 = 260)
        self.special_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
        self.byte_tokens = [f"<byte_{i}>" for i in range(256)]

        # token -> id 映射 / token -> id mapping
        self.token2id: Dict[str, int] = {}
        for i, tok in enumerate(self.special_tokens):
            self.token2id[tok] = i
        for i, tok in enumerate(self.byte_tokens):
            self.token2id[tok] = i + len(self.special_tokens)

        # id -> token 映射 / id -> token mapping
        self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}

        # BPE合并规则: (token_a, token_b) -> merged_token
        # BPE merge rules: (token_a, token_b) -> merged_token
        self.merges: Dict[Tuple[str, str], str] = {}

        # 下一个可用ID / Next available ID
        self._next_id = len(self.token2id)

    def train(self, texts: List[str], max_merges: int = 5000) -> None:
        """
        从文本语料训练BPE词表 / Train BPE vocabulary from text corpus.

        算法 / Algorithm:
          1. 将每个字符转为字节级token / Convert each character to byte-level tokens
          2. 统计所有相邻token对的频率 / Count frequency of all adjacent token pairs
          3. 合并最高频的对，添加新token到词表 / Merge the most frequent pair, add new token to vocab
          4. 重复直到达到词表大小或合并次数上限 / Repeat until vocab size or merge limit reached

        Args:
            texts:      训练文本列表 / List of training texts.
            max_merges: 最大合并次数 / Maximum number of merges.
        """
        # 将文本转为token序列列表 / Convert texts to token sequence list
        corpus: List[List[str]] = []
        for text in texts:
            tokens = [f"<byte_{b}>" for b in text.encode("utf-8")]
            corpus.append(tokens)

        # 迭代合并 / Iterative merging
        for _ in range(max_merges):
            if self._next_id >= self.vocab_size:
                break

            # 统计所有相邻对 / Count all adjacent pairs
            pair_counts: Counter = Counter()
            for tokens in corpus:
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i + 1])] += 1

            if not pair_counts:
                break

            # 找到最频繁的对 / Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]

            # 创建合并后的新token / Create new merged token
            merged = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged
            if merged not in self.token2id:
                self.token2id[merged] = self._next_id
                self.id2token[self._next_id] = merged
                self._next_id += 1

            # 更新语料库 / Update corpus
            new_corpus = []
            for tokens in corpus:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_corpus.append(new_tokens)
            corpus = new_corpus

    def encode(self, text: str, max_len: int = 77) -> List[int]:
        """
        将文本编码为token ID序列。
        Encode text into a token ID sequence.

        格式 / Format: [SOS, token1, token2, ..., EOS, PAD, PAD, ...]

        Args:
            text:    输入文本 / Input text.
            max_len: 最大序列长度 (含SOS和EOS) / Max sequence length (incl. SOS and EOS).

        Returns:
            token IDs列表 / List of token IDs.
        """
        # 字节级token化 / Byte-level tokenization
        tokens = [f"<byte_{b}>" for b in text.encode("utf-8")]

        # 应用BPE合并规则 (按训练时的顺序)
        # Apply BPE merge rules (in the order they were trained)
        changed = True
        while changed:
            changed = False
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.merges:
                    new_tokens.append(self.merges[(tokens[i], tokens[i + 1])])
                    i += 2
                    changed = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # 添加特殊token / Add special tokens
        # 留一个位置给EOS，最多 max_len - 2 个内容token
        # Reserve one position for EOS, at most max_len - 2 content tokens
        content_limit = max_len - 2
        tokens = tokens[:content_limit]

        # 转换为IDs / Convert to IDs
        ids = [self.SOS_ID]
        for tok in tokens:
            ids.append(self.token2id.get(tok, self.UNK_ID))
        ids.append(self.EOS_ID)

        # 填充到max_len / Pad to max_len
        while len(ids) < max_len:
            ids.append(self.PAD_ID)

        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        """
        将token ID序列解码为文本。
        Decode token ID sequence back to text.

        Args:
            ids: token ID列表 / List of token IDs.

        Returns:
            解码后的文本 / Decoded text string.
        """
        # 跳过特殊token，收集字节token / Skip special tokens, collect byte tokens
        byte_values = []
        for token_id in ids:
            if token_id in (self.SOS_ID, self.EOS_ID, self.PAD_ID):
                continue
            token = self.id2token.get(token_id, "")
            if token.startswith("<byte_") and token.endswith(">"):
                try:
                    byte_val = int(token[6:-1])
                    byte_values.append(byte_val)
                except ValueError:
                    pass
            else:
                # 合并后的token，拆回字节 / Merged token, split back to bytes
                for part in self._split_merged_token(token):
                    if part.startswith("<byte_") and part.endswith(">"):
                        try:
                            byte_val = int(part[6:-1])
                            byte_values.append(byte_val)
                        except ValueError:
                            pass

        return bytes(byte_values).decode("utf-8", errors="replace")

    def _split_merged_token(self, token: str) -> List[str]:
        """
        将合并的BPE token拆回字节token。
        Split a merged BPE token back into byte tokens.
        """
        parts = []
        i = 0
        while i < len(token):
            if token[i:].startswith("<byte_"):
                end = token.index(">", i) + 1
                parts.append(token[i:end])
                i = end
            else:
                i += 1
        return parts

    @property
    def vocab_len(self) -> int:
        """当前词表大小 / Current vocabulary size."""
        return self._next_id


# ============================================================
# 图像-文本对数据集 / Image-Text Pair Dataset
# ============================================================

class ImageTextDataset(Dataset):
    """
    图像-文本对数据集，支持COCO风格和自定义格式。
    Image-text pair dataset supporting COCO-style and custom format.

    COCO风格格式 / COCO-style format:
      annotations.json:
      {
        "images": [
          {"id": 1, "file_name": "img001.jpg"},
          ...
        ],
        "annotations": [
          {"image_id": 1, "caption": "a dog playing in the park"},
          ...
        ]
      }

    自定义格式 / Custom format (TSV):
      每行 / Each line: image_path\\tcaption_text

    Args:
        data_root:       图像根目录 / Image root directory.
        ann_file:        标注文件路径 / Annotation file path.
        transform:       图像变换函数 / Image transform function.
        tokenizer:       文本分词器 / Text tokenizer.
        max_seq_len:     最大文本序列长度 / Maximum text sequence length.
        dataset_type:    "coco" 或 "custom" / "coco" or "custom".
    """

    def __init__(
        self,
        data_root: str = "./data",
        ann_file: str = "./data/annotations.json",
        transform: Optional[Callable] = None,
        tokenizer: Optional[SimpleBPETokenizer] = None,
        max_seq_len: int = 77,
        dataset_type: str = "coco",
    ):
        self.data_root = data_root
        self.transform = transform
        self.tokenizer = tokenizer or SimpleBPETokenizer()
        self.max_seq_len = max_seq_len
        self.dataset_type = dataset_type

        # 加载标注 / Load annotations
        self.pairs: List[Dict[str, Any]] = []
        self._load_annotations(ann_file)

    def _load_annotations(self, ann_file: str) -> None:
        """
        加载标注文件 / Load annotation file.

        根据dataset_type选择不同的解析方式。
        Parse differently based on dataset_type.
        """
        if self.dataset_type == "coco":
            self._load_coco(ann_file)
        elif self.dataset_type == "custom":
            self._load_custom(ann_file)
        else:
            raise ValueError(
                f"不支持的数据集类型 / Unsupported dataset type: {self.dataset_type}"
            )

    def _load_coco(self, ann_file: str) -> None:
        """
        加载COCO风格标注 / Load COCO-style annotations.
        """
        if not os.path.exists(ann_file):
            # 如果文件不存在，生成模拟数据用于测试
            # If file doesn't exist, generate mock data for testing
            self._generate_mock_data()
            return

        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 建立 image_id -> file_name 映射 / Build image_id -> file_name mapping
        id_to_file = {}
        for img_info in data.get("images", []):
            id_to_file[img_info["id"]] = img_info["file_name"]

        # 收集 (image_path, caption) 对 / Collect (image_path, caption) pairs
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in id_to_file:
                self.pairs.append({
                    "image_path": os.path.join(self.data_root, id_to_file[img_id]),
                    "caption": ann["caption"],
                })

    def _load_custom(self, ann_file: str) -> None:
        """
        加载自定义TSV格式标注 / Load custom TSV-format annotations.
        """
        if not os.path.exists(ann_file):
            self._generate_mock_data()
            return

        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    image_path, caption = parts
                    # 相对路径则拼接data_root / Join relative paths with data_root
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(self.data_root, image_path)
                    self.pairs.append({
                        "image_path": image_path,
                        "caption": caption,
                    })

    def _generate_mock_data(self, num_samples: int = 100) -> None:
        """
        生成模拟数据 (用于测试和演示)。
        Generate mock data for testing and demonstration.
        """
        captions = [
            "a photo of a cat sitting on a sofa",
            "a dog playing in the park",
            "a beautiful sunset over the ocean",
            "a red car driving on a highway",
            "a person riding a bicycle",
            "a white house with a blue door",
            "a group of people at a restaurant",
            "a mountain landscape with snow",
            "a bird flying in the sky",
            "a city skyline at night",
        ]
        for i in range(num_samples):
            self.pairs.append({
                "image_path": None,  # 使用随机tensor / Use random tensor
                "caption": captions[i % len(captions)],
            })

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本 / Get a single sample.

        Returns:
            image: [3, H, W] 图像tensor / Image tensor.
            text:  [max_seq_len] token ID tensor / Token ID tensor.
        """
        pair = self.pairs[idx]

        # 加载图像 / Load image
        if pair["image_path"] is not None and os.path.exists(pair["image_path"]):
            # 使用PIL加载真实图像 / Load real image with PIL
            try:
                from PIL import Image
                image = Image.open(pair["image_path"]).convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    # 默认变换: 转为tensor / Default transform: convert to tensor
                    import torchvision.transforms as T
                    default_transform = T.Compose([
                        T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711],
                        ),
                    ])
                    image = default_transform(image)
            except ImportError:
                # PIL不可用，使用随机图像 / PIL not available, use random image
                image = torch.randn(3, 224, 224)
        else:
            # 模拟数据: 随机图像 / Mock data: random image
            image = torch.randn(3, 224, 224)

        # 编码文本 / Encode text
        text_ids = self.tokenizer.encode(pair["caption"], max_len=self.max_seq_len)
        text = torch.tensor(text_ids, dtype=torch.long)

        return image, text


# ============================================================
# 数据变换构建 / Build Data Transforms
# ============================================================

def build_transforms(
    image_size: int = 224,
    is_train: bool = True,
    mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
) -> Callable:
    """
    构建图像数据变换管道。
    Build image data transformation pipeline.

    训练时使用数据增强 (随机裁剪、水平翻转等)，评估时只做中心裁剪。
    Use data augmentation during training (random crop, horizontal flip, etc.),
    only center crop during evaluation.

    Args:
        image_size: 目标图像尺寸 / Target image size.
        is_train:   是否为训练模式 / Whether in training mode.
        mean:       归一化均值 / Normalization mean.
        std:        归一化标准差 / Normalization std.

    Returns:
        torchvision.transforms.Compose 变换管道 / Transform pipeline.
    """
    try:
        import torchvision.transforms as T
    except ImportError:
        # torchvision不可用时返回None / Return None when torchvision unavailable
        return None

    if is_train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # 随机裁剪 / Random crop
            T.RandomHorizontalFlip(),                            # 水平翻转 / Horizontal flip
            T.ColorJitter(brightness=0.2, contrast=0.2),        # 颜色抖动 / Color jitter
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.1)),    # 略大于目标尺寸 / Slightly larger than target
            T.CenterCrop(image_size),            # 中心裁剪 / Center crop
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ============================================================
# DataLoader 构建辅助函数 / DataLoader Builder Helper
# ============================================================

def build_dataloader(
    dataset: ImageTextDataset,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    构建 DataLoader / Build DataLoader.

    Args:
        dataset:     数据集实例 / Dataset instance.
        batch_size:  批次大小 / Batch size.
        num_workers: 工作线程数 / Number of workers.
        shuffle:     是否打乱 / Whether to shuffle.
        drop_last:   是否丢弃不完整的最后批次 / Whether to drop last incomplete batch.

    Returns:
        DataLoader 实例 / DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
