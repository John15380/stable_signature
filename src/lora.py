# src/lora.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class LoRALayer(nn.Module):
    # (这个类保持不变，但为了完整性，我们把它留在这里)
    def __init__(self, in_features, out_features, rank=32, alpha=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        
        self.lora_A = Parameter(torch.Tensor(rank, in_features))
        self.lora_B = Parameter(torch.Tensor(out_features, rank))
        
        self.scaling = self.alpha / self.rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return original_out + lora_out * self.scaling

# ==================== GEMINI 最终修复版本 ====================
class LoRAConv2d(nn.Module):
    """
    一个 LoRA 包装器，用于 nn.Conv2d。
    它使用两个卷积层（一个 down-sample，一个 up-sample）来实现 LoRA。
    """
    def __init__(self, conv_module: nn.Conv2d, rank=32, alpha=64):
        super().__init__()
        self.conv = conv_module  # 原始的、将被冻结的卷积层
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        # 冻结原始卷积层
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        # ==================== 关键修复 ====================
        # 获取原始卷积层所在的设备
        device = self.conv.weight.device
        # ===============================================

        # 获取原始层的参数
        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size
        stride = self.conv.stride
        padding = self.conv.padding

        # 定义新的可训练 LoRA 卷积层
        self.lora_down = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, (1, 1), stride=1, padding=0, bias=False)
        
        # ==================== 关键修复 ====================
        # 将新创建的LoRA层移动到与原始层相同的设备上
        self.lora_down.to(device)
        self.lora_up.to(device)
        # ===============================================

        # 初始化 LoRA 权重
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        # 计算原始输出和 LoRA 路径的输出
        original_output = self.conv(x)
        lora_output = self.lora_up(self.lora_down(x)) * self.scaling
        
        # 将两者相加
        return original_output + lora_output