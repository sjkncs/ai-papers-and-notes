"""
Atlas: 多模态终身记忆 Agent 的简化 PyTorch 复现 / Simplified PyTorch reproduction of a multimodal lifelong memory agent.

本代码演示如何为一个 Agent 构建跨文本、图像与结构化知识的记忆系统：
- 使用共享投影头将不同模态嵌入同一向量空间。
- 用图结构组织记忆节点与关系边。
- 支持基于向量相似度的跨模态检索与记忆写入。
- 附带中文注释，便于理解核心思想。

This script demonstrates how to build a multimodal memory system for an agent:
- Uses a shared projection head to embed different modalities into the same vector space.
- Organizes memory as a graph of nodes and relational edges.
- Supports cross-modal retrieval by vector similarity and memory write operations.
- Comments are in Chinese to match the weekly digest style.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class MultimodalEncoder(nn.Module):
    """
    多模态编码器：将文本、图像和结构化特征投影到同一向量空间。
    Multimodal encoder: projects text, image, and structured features into a shared vector space.
    """

    def __init__(self, dim: int):
        super().__init__()
        # 三种模态各自的简单编码器（实际中可用 CLIP、BERT 等预训练模型）
        # Simple encoders for three modalities; in practice use CLIP, BERT, etc.
        self.text_proj = nn.Linear(768, dim)
        self.image_proj = nn.Linear(512, dim)
        self.struct_proj = nn.Linear(128, dim)
        self.norm = nn.LayerNorm(dim)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.text_proj(x)))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.image_proj(x)))

    def encode_struct(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.struct_proj(x)))


class MemoryNode:
    """
    记忆节点：存储单条多模态观测。
    Memory node: stores a single multimodal observation.
    """

    def __init__(self, node_id: int, modality: str, embedding: torch.Tensor, content: str):
        self.node_id = node_id
        self.modality = modality  # 'text', 'image', 'struct'
        self.embedding = embedding
        self.content = content
        self.edges: List[Tuple[int, str]] = []  # (target_node_id, relation_type)

    def add_edge(self, target_id: int, relation: str):
        self.edges.append((target_id, relation))


class AtlasMemory:
    """
    Atlas 记忆系统：多模态向量记忆 + 情景图。
    Atlas memory system: multimodal vector memory + episodic graph.
    """

    def __init__(self, encoder: MultimodalEncoder, dim: int):
        self.encoder = encoder
        self.dim = dim
        self.nodes: Dict[int, MemoryNode] = {}
        self.next_id = 0

    def write_text(self, text_feature: torch.Tensor, content: str) -> int:
        """写入文本记忆 / Write a text memory."""
        emb = self.encoder.encode_text(text_feature)
        node = MemoryNode(self.next_id, "text", emb, content)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node.node_id

    def write_image(self, image_feature: torch.Tensor, content: str) -> int:
        """写入图像记忆 / Write an image memory."""
        emb = self.encoder.encode_image(image_feature)
        node = MemoryNode(self.next_id, "image", emb, content)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node.node_id

    def write_struct(self, struct_feature: torch.Tensor, content: str) -> int:
        """写入结构化知识记忆 / Write a structured knowledge memory."""
        emb = self.encoder.encode_struct(struct_feature)
        node = MemoryNode(self.next_id, "struct", emb, content)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node.node_id

    def add_relation(self, src_id: int, tgt_id: int, relation: str):
        """添加记忆节点之间的关系边 / Add a relational edge between memory nodes."""
        if src_id in self.nodes and tgt_id in self.nodes:
            self.nodes[src_id].add_edge(tgt_id, relation)

    def retrieve(self, query_feature: torch.Tensor, modality: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        跨模态检索：用查询向量在所有记忆中检索最相似的节点。
        Cross-modal retrieval: retrieve most similar nodes using a query vector.
        """
        if modality == "text":
            query_emb = self.encoder.encode_text(query_feature)
        elif modality == "image":
            query_emb = self.encoder.encode_image(query_feature)
        else:
            query_emb = self.encoder.encode_struct(query_feature)

        all_embeddings = torch.cat([n.embedding for n in self.nodes.values()], dim=0)
        similarities = F.cosine_similarity(query_emb, all_embeddings, dim=-1)
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices.tolist():
            node = list(self.nodes.values())[idx]
            results.append((node.node_id, similarities[idx].item(), node.content))
        return results

    def summarize_session(self, node_ids: List[int]) -> str:
        """
        简单会话摘要：按时间顺序拼接文本记忆内容。
        Simple session summary: concatenate text memory contents in chronological order.
        """
        texts = [self.nodes[i].content for i in node_ids if self.nodes[i].modality == "text"]
        return " | ".join(texts)


if __name__ == "__main__":
    dim = 256
    encoder = MultimodalEncoder(dim)
    memory = AtlasMemory(encoder, dim)

    # 模拟写入一次交互中的多模态观测
    # Simulate writing multimodal observations from one interaction.
    text_feat = torch.randn(1, 768)
    img_feat = torch.randn(1, 512)
    struct_feat = torch.randn(1, 128)

    n1 = memory.write_text(text_feat, "用户询问如何制作咖啡 / User asks how to make coffee")
    n2 = memory.write_image(img_feat, "咖啡机照片 / Photo of coffee machine")
    n3 = memory.write_struct(struct_feat, "步骤：研磨→压粉→萃取 / Steps: grind→tamp→extract")

    # 建立关系：文本→图像（相关），文本→结构化（因果）
    # Build relations: text→image (relevant), text→struct (causal).
    memory.add_relation(n1, n2, "related_to")
    memory.add_relation(n1, n3, "leads_to")

    # 用新的文本查询检索相关记忆
    # Retrieve relevant memories with a new text query.
    query = torch.randn(1, 768)
    results = memory.retrieve(query, modality="text", top_k=3)

    print("检索结果 / Retrieval results:")
    for node_id, score, content in results:
        print(f"  Node {node_id}: score={score:.4f}, content={content}")

    # 会话摘要 / Session summary.
    summary = memory.summarize_session([n1, n2, n3])
    print(f"会话摘要 / Session summary: {summary}")
