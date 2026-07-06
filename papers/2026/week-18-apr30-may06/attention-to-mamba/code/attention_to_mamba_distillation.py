"""
Attention to Mamba: 跨架构蒸馏的简化 PyTorch 复现
Attention to Mamba: Simplified PyTorch Reproduction of Cross-Architecture Distillation

本脚本复现论文 "Attention to Mamba: A Recipe for Cross-Architecture Distillation" (arXiv:2604.14191)
的核心思想：在不引入混合注意力块的前提下，将 Transformer 教师模型的知识蒸馏到纯 Mamba/SSM 学生模型。

由于原版 Mamba 依赖 CUDA 核函数（mamba-ssm），本脚本使用一个简化的结构化状态空间层（Structured State Space Layer）
作为学生模型的核心递归单元，重点展示两阶段蒸馏目标：
1. 阶段一：核技巧对齐（Kernel Trick Alignment）——将 Transformer 的注意力核与学生 SSM 的转移核建立对应；
2. 阶段二：标准蒸馏（Logits + Hidden-state Distillation）——在已对齐的表示空间上进行知识迁移。

运行环境 / Requirements:
    pip install torch numpy tqdm

使用说明 / Usage:
    python attention_to_mamba_distillation.py
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. 工具函数 / Utilities
# ---------------------------------------------------------------------------

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU 激活函数，Transformer 与部分 Mamba 变体常用。
    GELU activation commonly used in Transformers and some Mamba variants.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    构造下三角因果掩码，用于 Transformer 的自注意力。
    Construct a causal (lower-triangular) mask for Transformer self-attention.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # 将 0 填充为很大的负数，softmax 后变为 0
    mask = mask.masked_fill(mask == 0, float("-inf"))
    return mask


# ---------------------------------------------------------------------------
# 2. Transformer 教师模型 / Transformer Teacher
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    因果多头自注意力（Causal Multi-Head Self-Attention）。
    这是 Transformer 教师的核心，其注意力矩阵将被用于阶段一的核技巧对齐。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 合并 Q, K, V 投影以简化代码
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回输出与注意力权重（用于核技巧对齐）。
        Returns output and attention weights for kernel alignment.
        """
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # 分头 reshape: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力得分: (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = create_causal_mask(T, x.device)
        scores = scores + mask.view(1, 1, T, T)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out, attn_weights


class TransformerBlock(nn.Module):
    """单个 Transformer 编码块（Pre-LN）。"""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 预层归一化 + 残差连接
        h, attn = self.attn(self.ln1(x))
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, attn


class TransformerTeacher(nn.Module):
    """
    Transformer 教师模型：嵌入层 + N 个 TransformerBlock + 输出头。
    训练中冻结，仅作为蒸馏目标。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, list, list]:
        """
        返回：
        - logits: (B, T, vocab_size)
        - hidden_states: 每层输出列表
        - attention_weights: 每层注意力权重列表
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        hidden_states = []
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            hidden_states.append(x)
            attention_weights.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, hidden_states, attention_weights


# ---------------------------------------------------------------------------
# 3. 简化 Mamba/SSM 学生模型 / Simplified Mamba/SSM Student
# ---------------------------------------------------------------------------

class SimplifiedSSMLayer(nn.Module):
    """
    简化的结构化状态空间层（Selective Linear RNN）。

    原版 Mamba 使用连续时间参数化（A, B, C, delta）与离散化，并依赖硬件感知的扫描算法。
    这里为了可运行性，使用一个离散化的选择性线性递归：
        h_t = diag(A) * h_{t-1} + B_t * x_t
        y_t = C_t * h_t
    其中 B_t, C_t 是输入相关的门控，A 是可学习衰减参数。
    """

    def __init__(self, d_model: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # 输入投影：用于生成 x_in, B, C
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # 可学习的指数衰减参数 A（对数空间保证稳定性）
        self.log_A = nn.Parameter(torch.randn(d_model, state_dim) * 0.01 - 3.0)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: (B, T, D)
        输出 y: (B, T, D)
        """
        B, T, D = x.shape
        proj = self.in_proj(x)  # (B, T, 3D)
        x_in, B_gate, C_gate = proj.chunk(3, dim=-1)

        # 使用 SiLU/Swish 作为门控激活
        x_in = F.silu(x_in)
        B_gate = F.silu(B_gate)
        C_gate = F.silu(C_gate)

        # A 为 (0,1) 之间的衰减因子，保证递归稳定
        A = torch.exp(-F.softplus(-self.log_A))  # (D, state_dim)

        # 初始化隐藏状态
        h = torch.zeros(B, D, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            # h_t = A * h_{t-1} + B_t * x_t
            h = A * h + B_gate[:, t, :, None] * x_in[:, t, :, None]  # (B, D, state_dim)
            # y_t = sum(C_t * h_t, dim=state_dim)
            y_t = (C_gate[:, t, :, None] * h).sum(dim=-1)  # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, D)
        y = self.dropout(self.out_proj(y))
        return y


class SSMBlock(nn.Module):
    """SSM 学生块：Pre-LN + SSM + FFN。"""

    def __init__(self, d_model: int, state_dim: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SimplifiedSSMLayer(d_model, state_dim, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SSMStudent(nn.Module):
    """
    纯 SSM 学生模型：嵌入层 + N 个 SSMBlock + 输出头。
    结构上与教师对称，但没有使用任何注意力机制。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        state_dim: int = 16,
        dim_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SSMBlock(d_model, state_dim, dim_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        返回：
        - logits: (B, T, vocab_size)
        - hidden_states: 每层输出列表
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        hidden_states = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, hidden_states


# ---------------------------------------------------------------------------
# 4. 两阶段蒸馏目标 / Two-Stage Distillation Objectives
# ---------------------------------------------------------------------------

class AttentionToMambaDistiller(nn.Module):
    """
    将教师与学生封装在一起，实现两阶段蒸馏损失。
    """

    def __init__(self, teacher: TransformerTeacher, student: SSMStudent):
        super().__init__()
        self.teacher = teacher
        self.student = student
        # 冻结教师
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # 用于阶段一：将 SSM 隐藏状态投影到与注意力核可比的低维核空间
        self.kernel_projections = nn.ModuleList(
            [
                nn.Linear(student.d_model, teacher.d_model, bias=False)
                for _ in range(len(student.blocks))
            ]
        )

    def kernel_alignment_loss(
        self,
        teacher_attn: torch.Tensor,
        student_hidden: torch.Tensor,
        projection: nn.Linear,
    ) -> torch.Tensor:
        """
        阶段一：核技巧对齐损失。

        教师注意力矩阵 A_teacher: (B, n_heads, T, T)
        学生隐藏状态 h_student: (B, T, D)
        通过投影得到核特征：phi = projection(h_student)  -> (B, T, D)
        然后构造学生核矩阵：K_student = phi @ phi^T  -> (B, T, T)
        对教师的多个头取平均后与 K_student 做 MSE 对齐。

        这对应论文中“将二次注意力核映射到线性状态空间核”的简化实现。
        """
        B, T, D = student_hidden.shape
        # 教师注意力在头上取平均
        attn_avg = teacher_attn.mean(dim=1)  # (B, T, T)

        # 学生核特征投影
        phi = projection(student_hidden)  # (B, T, D)
        # 构造核矩阵并做温度缩放，数值更稳定
        K_student = torch.bmm(phi, phi.transpose(1, 2)) / math.sqrt(D)

        loss = F.mse_loss(K_student, attn_avg)
        return loss

    def hidden_distillation_loss(
        self,
        teacher_hidden: torch.Tensor,
        student_hidden: torch.Tensor,
        projection: nn.Linear,
    ) -> torch.Tensor:
        """隐藏状态蒸馏：将学生投影到教师表示空间后计算 MSE。"""
        student_proj = projection(student_hidden)
        return F.mse_loss(student_proj, teacher_hidden)

    def logit_distillation_loss(
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float = 2.0
    ) -> torch.Tensor:
        """Logits 蒸馏：KL 散度。"""
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
        return kl

    def forward(
        self,
        input_ids: torch.Tensor,
        stage: str = "stage1",
        alpha_kd: float = 0.5,
        temperature: float = 2.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播并计算损失。
        stage: "stage1" 仅核技巧对齐；"stage2" 核对齐 + logits + hidden-state 蒸馏。
        """
        with torch.no_grad():
            t_logits, t_hidden, t_attns = self.teacher(input_ids)

        s_logits, s_hidden = self.student(input_ids)

        total_loss = torch.tensor(0.0, device=input_ids.device)
        metrics = {}

        # 阶段一：逐层核技巧对齐
        kernel_loss = 0.0
        for i, (t_h, t_a, s_h, proj) in enumerate(
            zip(t_hidden, t_attns, s_hidden, self.kernel_projections)
        ):
            kernel_loss += self.kernel_alignment_loss(t_a, s_h, proj)
            if stage == "stage2":
                total_loss += self.hidden_distillation_loss(t_h, s_h, proj)
        kernel_loss = kernel_loss / len(t_hidden)
        metrics["kernel_loss"] = kernel_loss.item()
        total_loss += kernel_loss

        # 阶段二：logits 蒸馏
        if stage == "stage2":
            logit_loss = self.logit_distillation_loss(t_logits, s_logits, temperature)
            metrics["logit_loss"] = logit_loss.item()
            total_loss += alpha_kd * logit_loss

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics


# ---------------------------------------------------------------------------
# 5. 训练流程 / Training Loop
# ---------------------------------------------------------------------------

def create_dummy_dataloader(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> DataLoader:
    """
    构造随机文本数据用于演示（实际应替换为真实语料）。
    Creates random text data for demonstration; replace with real corpus in practice.
    """
    data = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len), device=device)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_distillation(
    teacher: TransformerTeacher,
    student: SSMStudent,
    dataloader: DataLoader,
    n_epochs_stage1: int = 3,
    n_epochs_stage2: int = 5,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
):
    """
    执行两阶段蒸馏训练。
    """
    distiller = AttentionToMambaDistiller(teacher, student).to(device)
    # 同时优化学生模型与蒸馏器中的核投影层，否则 kernel_projections 将保持随机初始化。
    # Optimize both the student and the distiller's kernel projections.
    trainable_params = list(student.parameters()) + list(distiller.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # 阶段一：核技巧对齐
    print("\n[阶段一 / Stage 1] Kernel Trick Alignment")
    student.train()
    distiller.train()
    for epoch in range(n_epochs_stage1):
        epoch_metrics = {"kernel_loss": 0.0, "total_loss": 0.0}
        pbar = tqdm(dataloader, desc=f"Stage1 Epoch {epoch+1}/{n_epochs_stage1}")
        for (batch,) in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, metrics = distiller(batch, stage="stage1")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            pbar.set_postfix(metrics)
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
        print(f"  Epoch {epoch+1} avg: {epoch_metrics}")

    # 阶段二：完整蒸馏
    print("\n[阶段二 / Stage 2] Logits + Hidden-state Distillation")
    for epoch in range(n_epochs_stage2):
        epoch_metrics = {"kernel_loss": 0.0, "logit_loss": 0.0, "total_loss": 0.0}
        pbar = tqdm(dataloader, desc=f"Stage2 Epoch {epoch+1}/{n_epochs_stage2}")
        for (batch,) in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, metrics = distiller(batch, stage="stage2", alpha_kd=0.7, temperature=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            pbar.set_postfix(metrics)
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
        print(f"  Epoch {epoch+1} avg: {epoch_metrics}")


# ---------------------------------------------------------------------------
# 6. 主函数 / Main
# ---------------------------------------------------------------------------

def main():
    # 超参数 / Hyperparameters
    VOCAB_SIZE = 10000
    D_MODEL = 256
    N_LAYERS = 4
    N_HEADS = 4
    STATE_DIM = 16
    DIM_FF = 1024
    SEQ_LEN = 64
    BATCH_SIZE = 8
    NUM_BATCHES = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 / Using device: {device}")

    # 初始化教师与学生 / Initialize teacher and student
    teacher = TransformerTeacher(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_ff=DIM_FF,
        max_len=SEQ_LEN,
    ).to(device)

    student = SSMStudent(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        state_dim=STATE_DIM,
        dim_ff=DIM_FF,
        max_len=SEQ_LEN,
    ).to(device)

    # 学生参数数量 / Student parameter count
    n_params = sum(p.numel() for p in student.parameters())
    print(f"学生模型参数量 / Student parameters: {n_params:,}")

    # 构造数据 / Create data
    dataloader = create_dummy_dataloader(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        device=device,
    )

    # 训练 / Train
    train_distillation(
        teacher=teacher,
        student=student,
        dataloader=dataloader,
        n_epochs_stage1=2,
        n_epochs_stage2=3,
        lr=1e-4,
        device=device,
    )

    # 简单评估：计算学生在随机序列上的困惑度（真实任务需用测试集）
    student.eval()
    with torch.no_grad():
        sample = next(iter(dataloader))[0]
        logits, _ = student(sample)
        # 预测下一个 token（忽略最后一个位置）
        targets = sample[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss_eval = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        ppl = torch.exp(loss_eval)
        print(f"\n学生模型困惑度 / Student perplexity: {ppl.item():.2f}")


if __name__ == "__main__":
    main()
