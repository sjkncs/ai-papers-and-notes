"""
Deep Delta Learning (DDL) Models — 模型组件导入
================================================

导入所有DDL模型组件，方便外部使用。
Import all DDL model components for external use.

用法 / Usage:
    from models import DDLLayer, DDLTransformerBlock, DDLModel
    from models import StandardTransformer, count_params, estimate_flops

Author: Auto-generated from paper analysis
License: MIT
"""

# DDL核心组件 / Core DDL components
from .ddl_transformer import (
    # 门控Delta修正模块 / Gated delta correction module
    GatedDeltaGate,
    # DDL残差层 / DDL residual layer
    DDLLayer,
    # DDL Transformer Block / DDL Transformer Block
    DDLTransformerBlock,
    # 完整DDL语言模型 / Full DDL language model
    DDLModel,
    # 标准Transformer (对比基线) / Standard Transformer (comparison baseline)
    StandardTransformerBlock,
    StandardTransformer,
    # 工具函数 / Utility functions
    count_params,
    estimate_flops,
)

__all__ = [
    "GatedDeltaGate",
    "DDLLayer",
    "DDLTransformerBlock",
    "DDLModel",
    "StandardTransformerBlock",
    "StandardTransformer",
    "count_params",
    "estimate_flops",
]
