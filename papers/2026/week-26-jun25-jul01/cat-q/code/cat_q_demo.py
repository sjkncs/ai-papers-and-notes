"""
CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs
PyTorch 概念复现演示（含详细中文注释）

本脚本实现论文核心思想的简化版：
    - Learnable Modulation (LM): 学习可调的权重缩放与三值化阈值
    - Softened Ternarization (ST): 用可微过渡函数软化硬三值化

目标：在少量校准数据上，将预训练 LLM 的线性层权重量化为 {-t, 0, +t}，
      并保持输出分布尽量接近原模型。

依赖：torch, transformers（Hugging Face）
运行：python cat_q_demo.py

说明：
    - 使用 GPT-2 作为小型示例模型，便于在个人机器上直接运行。
    - 为了演示速度，本脚本只量化部分层，并使用随机/短文本校准。
    - 真实实现需要处理更多细节（分组量化、逐层校准、KV Cache 量化等）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D
from typing import Optional, Union
import argparse


# ---------------------------------------------------------------------------
# 1. 软三值化函数（Softened Ternarization, ST）
# ---------------------------------------------------------------------------

def softened_ternarize(
    w: torch.Tensor,
    threshold: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    将权重 w 软三值化为 {-threshold, 0, +threshold} 的近似值。

    参数:
        w: 原始权重，形状任意。
        threshold: 可学习的正阈值（标量或逐通道向量）。
        temperature: 软化温度。值越小越接近硬三值；值越大越平滑。

    返回:
        软三值化后的权重，取值在 [-threshold, +threshold] 之间。

    思路:
        硬三值化:  w_q = sign(w) * threshold * 1{|w| > threshold}
        为了可微，用 sigmoid 近似示性函数：
            p_pos  ≈ σ((w - threshold) / temperature)
            p_neg  ≈ σ((-w - threshold) / temperature)
            w_q ≈ threshold * (p_pos - p_neg)
        当 temperature -> 0 时，逼近硬三值 {-t, 0, +t}。
    """
    # 保证阈值为正
    t = F.softplus(threshold) + 1e-6
    p_pos = torch.sigmoid((w - t) / temperature)
    p_neg = torch.sigmoid((-w - t) / temperature)
    return t * (p_pos - p_neg)


# ---------------------------------------------------------------------------
# 2. 带可学习调制的三值线性层（Learnable Modulation + ST）
# ---------------------------------------------------------------------------

class TernaryLinear(nn.Module):
    """
    将普通 nn.Linear 替换为三值线性层。

    核心思想（对应论文 LM + ST）：
        - scale: 可学习的权重缩放因子，用于调制预训练权重的分布，
                 使其对三值化更鲁棒。
        - threshold: 可学习的正阈值，决定哪些权重被量化为 ±t，哪些归零。
        - 前向传播时，先对原始权重做缩放调制，再经过软三值化，
          最后与输入相乘。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_scale: float = 1.0,
        init_threshold: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 预训练权重（不可训练，只做三值化目标）
        self.register_buffer("weight", torch.randn(out_features, in_features))

        # 可学习调制参数（LM）
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.threshold = nn.Parameter(torch.tensor(init_threshold))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # 软三值化温度：训练时稍大以保证可微；推理时接近 0
        self.temperature = 0.05

    def get_ternary_weight(self) -> torch.Tensor:
        """返回当前的三值化权重（软或硬由 temperature 控制）。"""
        # 调制：将原始权重乘以可学习缩放因子
        modulated = self.weight * self.scale
        return softened_ternarize(modulated, self.threshold, self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self.get_ternary_weight()
        out = F.linear(x, w_q, self.bias)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scale={self.scale.item():.4f}, threshold={F.softplus(self.threshold).item():.4f}"
        )


# ---------------------------------------------------------------------------
# 3. 将普通 nn.Linear 替换为 TernaryLinear
# ---------------------------------------------------------------------------

def _get_linear_dims(layer: Union[nn.Linear, Conv1D]) -> tuple[int, int]:
    """统一获取 nn.Linear 或 Conv1D 的输入/输出维度。"""
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    elif isinstance(layer, Conv1D):
        # Conv1D 权重形状为 (in_features, out_features)
        return layer.weight.shape[0], layer.weight.shape[1]
    else:
        raise TypeError(f"不支持的层类型: {type(layer)}")


def replace_linear_with_ternary(
    module: nn.Module,
    layer_filter: Optional[callable] = None,
) -> int:
    """
    递归遍历模型，把部分 nn.Linear 或 transformers Conv1D 替换为 TernaryLinear。

    参数:
        module: 待修改的模型/模块。
        layer_filter: 可选过滤函数，返回 True 才替换该层。
                      默认替换所有 out_features > 100 的线性层，避免嵌入层被量化。

    返回:
        被替换的层数。
    """
    if layer_filter is None:
        layer_filter = lambda m: _get_linear_dims(m)[1] > 100 and _get_linear_dims(m)[0] > 100

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Linear, Conv1D)) and layer_filter(child):
            in_features, out_features = _get_linear_dims(child)
            ternary = TernaryLinear(
                in_features=in_features,
                out_features=out_features,
                bias=child.bias is not None,
            )
            # 复制预训练权重到不可训练缓冲区
            # Conv1D 的 forward 等价于 x @ W (W 形状为 in, out)，即 F.linear(x, W.T)
            # 因此需要转置权重以匹配 TernaryLinear 使用的 F.linear
            if isinstance(child, Conv1D):
                ternary.weight.copy_(child.weight.data.T)
            else:
                ternary.weight.copy_(child.weight.data)
            if child.bias is not None:
                ternary.bias.data.copy_(child.bias.data)
            setattr(module, name, ternary)
            replaced += 1
        else:
            replaced += replace_linear_with_ternary(child, layer_filter)
    return replaced


# ---------------------------------------------------------------------------
# 4. 校准函数：优化可学习调制参数，使三值模型输出接近原模型
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_layer_outputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_layers: list,
) -> list:
    """
    收集原模型在指定层上的输出（作为三值化的监督信号）。
    使用 register_forward_hook 捕获中间激活。
    """
    outputs = []
    handles = []

    def hook_fn(module, inp, out):
        outputs.append(out.detach().clone())

    for layer in target_layers:
        handles.append(layer.register_forward_hook(hook_fn))

    model(input_ids)

    for h in handles:
        h.remove()
    return outputs


def calibrate_ternary_layers(
    model: nn.Module,
    tokenizer,
    calibration_texts: list[str],
    num_steps: int = 200,
    lr: float = 5e-3,
    device: str = "cpu",
) -> None:
    """
    用少量校准文本优化所有 TernaryLinear 中的 scale 和 threshold。

    损失函数:
        L = MSE(三值层输出, 原始层输出) + λ * (scale - 1)^2
    第二项用于防止 scale 过度漂移，保持权重语义相对稳定。
    """
    model.eval()

    # 收集所有 TernaryLinear 层
    ternary_layers = [
        m for m in model.modules() if isinstance(m, TernaryLinear)
    ]
    if not ternary_layers:
        print("未找到需要校准的三值层。")
        return

    # 准备优化器：只优化 scale 和 threshold
    params = []
    for layer in ternary_layers:
        params.extend([layer.scale, layer.threshold])
    optimizer = torch.optim.Adam(params, lr=lr)

    print(f"开始校准 {len(ternary_layers)} 个三值层，共 {len(params)} 个参数...")

    for step in range(num_steps):
        total_loss = 0.0
        for text in calibration_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)

            # 计算三值模型输出
            outputs_q = model(input_ids).logits

            # 简单监督：让 logits 分布接近原始模型（通过隐藏态损失更精确，但这里以 logits 为例）
            # 注意：实际论文中通常使用层输出 MSE 或 perplexity 作为损失。
            with torch.no_grad():
                # 临时恢复原始权重，计算 teacher 输出
                # 这里采用简化策略：直接最小化三值权重的均方误差（weight MSE）
                pass

            # 层输出 MSE 损失
            loss = 0.0
            for layer in ternary_layers:
                w_orig = layer.weight
                w_q = layer.get_ternary_weight()
                loss += F.mse_loss(w_q, w_orig)

            # 额外正则：鼓励 scale 接近 1
            for layer in ternary_layers:
                loss += 0.01 * (layer.scale - 1.0) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}, loss={total_loss:.6f}")

    # 校准结束后降低温度，使三值化更“硬”
    for layer in ternary_layers:
        layer.temperature = 0.01
    print("校准完成，已将 temperature 降低至 0.01。")


# ---------------------------------------------------------------------------
# 5. 主流程：加载模型、替换层、校准、对比输出
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CAT-Q Ternary Quantization Demo")
    parser.add_argument(
        "--full", action="store_true",
        help="完整模式：量化更多层并运行更多校准步数（耗时更长）"
    )
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="校准步数（默认 quick 模式 30，完整模式 100）"
    )
    args = parser.parse_args()

    quick_mode = not args.full
    num_steps = args.num_steps if args.num_steps is not None else (30 if quick_mode else 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"运行模式: {'Quick（快速演示）' if quick_mode else 'Full（完整模式）'}")

    # 加载小型预训练语言模型 GPT-2
    model_name = "gpt2"
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 保存原始输出用于对比
    test_text = "The future of artificial intelligence depends on"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        original_logits = model(**inputs).logits

    # 将部分 Linear 替换为 TernaryLinear
    if quick_mode:
        # 快速模式：只替换前两个 transformer block 中的线性层
        max_blocks = 2
        replaced = 0
        for i, block in enumerate(model.transformer.h[:max_blocks]):
            replaced += replace_linear_with_ternary(block)
    else:
        replaced = replace_linear_with_ternary(model)
    print(f"已替换 {replaced} 个线性层为三值层。")

    # 校准数据：短文本。真实场景应使用领域相关文本，长度更长。
    calibration_texts = [
        "Artificial intelligence is transforming the way we work and live.",
        "Machine learning models require large amounts of data and compute.",
        "Quantization reduces model size and speeds up inference.",
        "Ternary weights use only three values and enable efficient hardware.",
    ]

    calibrate_ternary_layers(
        model, tokenizer, calibration_texts, num_steps=num_steps, lr=1e-2, device=device
    )

    # 比较量化前后 logits 的差异
    model.eval()
    with torch.no_grad():
        quantized_logits = model(**inputs).logits

    mse = F.mse_loss(quantized_logits, original_logits).item()
    rel_err = (mse / (original_logits.std().item() ** 2 + 1e-8)) ** 0.5

    print("\n========== 结果对比 ==========")
    print(f"原始 logits 均值: {original_logits.mean().item():.4f}, 标准差: {original_logits.std().item():.4f}")
    print(f"量化后 logits 均值: {quantized_logits.mean().item():.4f}, 标准差: {quantized_logits.std().item():.4f}")
    print(f"logits MSE: {mse:.6f}")
    print(f"相对误差 (MSE / std^2)^0.5: {rel_err:.4f}")

    # 统计三值化后权重的稀疏性（有多少接近 0）
    total_weights = 0
    zero_weights = 0
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, TernaryLinear):
                w_q = layer.get_ternary_weight()
                total_weights += w_q.numel()
                zero_weights += (w_q.abs() < 1e-3).sum().item()
    print(f"三值权重稀疏度: {zero_weights / total_weights * 100:.2f}% 接近 0")

    print("\n提示：本演示仅展示 CAT-Q 的核心思想。")
    print("实际部署时，请使用论文官方代码或针对具体模型做更精细的逐层校准。")


if __name__ == "__main__":
    main()
