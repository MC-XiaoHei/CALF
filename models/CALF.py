import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange  # 确保 einops 已安装

# 导入 transformers 和 torchvision:
from transformers import ViTModel, GPT2Model, ViTConfig, GPT2Config, GPT2Tokenizer
import math

# 尝试从 CALF 的 utils 导入 Normalize，如果失败则定义一个备用的。
try:
    from utils.tools import Normalize
except ImportError:
    import warnings

    warnings.warn("Could not import Normalize from utils.tools. Using fallback definition.")


    # 备用：定义一个与 CALF 兼容的 Normalize (假设 B, L, M -> B, M, L)
    class Normalize(nn.Module):
        def __init__(self, num_features, eps=1e-5, affine=False):
            super(Normalize, self).__init__()
            self.num_features = num_features
            self.eps = eps
            self.affine = affine
            self.mean = None
            self.std = None
            if self.affine:
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))

        def forward(self, x, mode='norm'):
            # CALF 使用 (B, L, M)
            # TimeCMA 使用 (B, M, L)
            # 这里的 x 预期是 (B, L, M)

            if mode == 'norm':
                # 沿 L 维度计算
                self.mean = torch.mean(x, dim=1, keepdim=True).detach()
                self.std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
                x = (x - self.mean) / self.std
                if self.affine:
                    x = x * self.weight.unsqueeze(0).unsqueeze(0) + self.bias.unsqueeze(0).unsqueeze(0)
            elif mode == 'denorm':
                if self.mean is None or self.std is None:
                    raise RuntimeError("Call 'norm' mode first to compute mean and std.")
                x = x * self.std + self.mean
            return x


# --- TimeCMA 辅助模块 ---

class SeriesDecomp(nn.Module):
    """
    通过一维平均池化进行时间序列分解
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            count_include_pad=False
        )

    def forward(self, x):
        """
        输入 x: (Batch, Vars, SeqLen)
        """
        trend = self.avg_pool(x)
        seasonal = x - trend
        return trend, seasonal


class GAF(nn.Module):
    """
    可微分的格拉姆角场 (GAF) 变换
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # x shape: (Batch, SeqLen)
        min_x = x.min(dim=1, keepdim=True)[0]
        max_x = x.max(dim=1, keepdim=True)[0]
        scaled_x = 2 * ((x - min_x) / (max_x - min_x + self.epsilon)) - 1
        scaled_x = torch.clamp(scaled_x, -1.0 + self.epsilon, 1.0 - self.epsilon)
        phi = torch.acos(scaled_x)  # (B, S)
        gaf = torch.cos(phi.unsqueeze(2) + phi.unsqueeze(1))
        return gaf  # (B, S, S)


class TrendModalityGenerator(nn.Module):
    """
    为“趋势”分量生成图像（热图）和 *多变量统计*
    """

    def __init__(self, n_vars, seq_len):
        super(TrendModalityGenerator, self).__init__()
        self.img_proj = nn.Conv2d(1, 3, kernel_size=1)
        self.img_resize = transforms.Resize((224, 224), antialias=True)

    def forward(self, trend):
        # trend shape: (Batch, Vars, SeqLen)

        # --- 图像生成 (热图) ---
        trend_img = self.img_resize(self.img_proj(trend.unsqueeze(1)))

        # --- 多变量统计 ---
        per_var_slope = (trend[..., -1] - trend[..., 0]).detach()
        avg_slope = per_var_slope.mean(dim=1)
        std_slope = per_var_slope.std(dim=1, unbiased=False)

        avg_magnitude = trend.mean(dim=2).detach()
        min_avg_mag = avg_magnitude.min(dim=1)[0]
        max_avg_mag = avg_magnitude.max(dim=1)[0]

        stats = {
            "avg_slope": avg_slope,
            "std_slope": std_slope,
            "min_avg_mag": min_avg_mag,
            "max_avg_mag": max_avg_mag
        }

        return trend_img, stats


class SeasonalModalityGenerator(nn.Module):
    """
    为“季节性”分量生成图像（GAF）和 *多变量统计*
    """

    def __init__(self, n_vars, seq_len):
        super(SeasonalModalityGenerator, self).__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars

        # --- 图像生成器 (GAF) ---
        self.gaf_transform = GAF()
        self.img_proj = nn.Conv2d(1, 3, kernel_size=1)
        self.img_resize = transforms.Resize((224, 224), antialias=True)

    def forward(self, seasonal):
        # seasonal shape: (Batch, Vars, SeqLen)

        # --- 图像生成 (GAF) ---
        seasonal_ts_avg = seasonal.mean(dim=1)
        seasonal_gaf = self.gaf_transform(seasonal_ts_avg)
        seasonal_img = self.img_resize(self.img_proj(seasonal_gaf.unsqueeze(1)))

        # --- 多变量统计 ---
        fft_vals = torch.fft.rfft(seasonal.detach(), dim=2)
        fft_amps = torch.abs(fft_vals)
        fft_amps[..., 0] = 0.0

        _, top_indices = torch.topk(fft_amps, 1, dim=2)
        periods = self.seq_len / top_indices.float().clamp(min=1).squeeze(-1)

        dominant_period, _ = torch.mode(torch.round(periods), dim=1)
        dominant_period = dominant_period.float()

        coherence = (torch.round(periods) == dominant_period.unsqueeze(1)).float().mean(dim=1)

        avg_amplitude = seasonal.std(dim=2, unbiased=False).detach()
        min_amp = avg_amplitude.min(dim=1)[0]
        max_amp = avg_amplitude.max(dim=1)[0]

        stats = {
            "dominant_period": dominant_period,
            "period_coherence": coherence,
            "min_amp": min_amp,
            "max_amp": max_amp
        }

        return seasonal_img, stats


class Model(nn.Module):  # 重命名 Dual -> Model
    """
    主模型：使用动态硬提示 (V4) + 分层交叉注意力 (TimeCMA 架构)
    已适配 CALF 的 __init__ 和 forward 签名
    """

    def __init__(self, configs, device):  # 适配 CALF 的 __init__
        super(Model, self).__init__()

        # 从 configs 对象中获取参数
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.n_vars = configs.enc_in  # CALF 使用 'enc_in' 作为变量数
        self.pred_len = configs.pred_len

        # 从 configs 或使用默认值
        decomp_kernel = getattr(configs, 'decomp_kernel', 25)
        n_heads = getattr(configs, 'n_heads', 4)
        dropout = getattr(configs, 'dropout', 0.7)
        # 复用 configs.d_model 作为 TimeCMA 的 'channel'
        self.channel = getattr(configs, 'd_model', 128)

        self.device = device

        # CALF 归一化器 (B, L, M)
        self.normalize_layers = Normalize(self.n_vars, affine=False)

        # --- 1. 分解层 ---
        self.decomp = SeriesDecomp(kernel_size=decomp_kernel).double()

        # --- 2. 加载冻结的 VLM 和 Tokenizer ---
        # 路径应在 configs 中定义
        vit_path = getattr(configs, 'vit_path', 'google/vit-base-patch16-224-in21k')
        gpt_path = getattr(configs, 'gpt_path', 'gpt2')

        print(f"加载 ViT 模型: {vit_path}")
        vit_config = ViTConfig.from_pretrained(vit_path)
        self.vit_model = ViTModel(vit_config)
        self.vit_d_model = vit_config.hidden_size  # 768

        print(f"加载 GPT-2 模型和 Tokenizer: {gpt_path}")
        gpt_config = GPT2Config.from_pretrained(gpt_path)
        self.gpt_model = GPT2Model(gpt_config)
        self.gpt_d_model = gpt_config.n_embd  # 768

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        assert self.vit_d_model == self.gpt_d_model, "ViT 和 GPT-2 的隐藏维度必须相同"
        self.d_model = self.vit_d_model

        print("冻结 ViT 和 GPT-2 的参数...")
        for param in self.vit_model.parameters():
            param.requires_grad = False
        for param in self.gpt_model.parameters():
            param.requires_grad = False

        # --- 3. 模态生成器 (可训练) ---
        self.trend_generator = TrendModalityGenerator(self.n_vars, self.seq_len).double()
        self.seasonal_generator = SeasonalModalityGenerator(self.n_vars, self.seq_len).double()

        # --- 4. 融合层 (可训练) ---
        # TimeCMA 以 SeqLen (L) 为 token 维度, Vars (M) 为特征维度
        # (B, L, M) -> (B, L, C)
        self.query_proj = nn.Linear(self.n_vars, self.channel).double()
        self.ts_encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.channel, nhead=n_heads, batch_first=True,
                                                            norm_first=True, dropout=dropout).to(self.device).double()
        self.ts_att = nn.TransformerEncoder(self.ts_encoder_layer2, num_layers=2).to(self.device).double()

        self.text_attn = nn.MultiheadAttention(
            embed_dim=self.channel, num_heads=n_heads, dropout=dropout, batch_first=True,
            kdim=self.d_model, vdim=self.d_model
        ).to(self.device).double()

        self.image_attn = nn.MultiheadAttention(
            embed_dim=self.channel, num_heads=n_heads, dropout=dropout, batch_first=True,
            kdim=self.d_model, vdim=self.d_model
        ).to(self.device).double()

        self.fusion_gate_proj = nn.Sequential(
            nn.Linear(self.channel * 3, self.channel * 3),
            nn.Sigmoid()
        ).double()

        # --- 5. 预测头 (适配 CALF 任务) ---

        # 适配 CALF 的特定任务输出头
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # (B, L, C) -> (B, L*C) -> (B, P*M) -> (B, P, M)
            self.out_layer = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.channel * self.seq_len, self.pred_len * self.n_vars),
                nn.Unflatten(dim=1, unflattened_size=(self.pred_len, self.n_vars))
            ).double()
        elif self.task_name == 'classification':
            self.num_class = configs.num_class
            # (B, L, C) -> (B, L*C) -> (B, NumClass)
            self.out_layer = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.channel * self.seq_len, self.num_class)
            ).double()
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # (B, L, C) -> (B, L*C) -> (B, L*M) -> (B, L, M)
            self.out_layer = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.channel * self.seq_len, self.seq_len * self.n_vars),
                nn.Unflatten(dim=1, unflattened_size=(self.seq_len, self.n_vars))
            ).double()

        # 将所有可训练层和 VLM 移动到 device
        for layer in (self.vit_model, self.gpt_model, self.trend_generator, self.seasonal_generator,
                      self.query_proj, self.ts_att, self.text_attn, self.image_attn,
                      self.fusion_gate_proj, self.out_layer, self.decomp, self.normalize_layers):
            layer.to(device=device)
            # 确保 VLM 处于 eval 模式
            if layer in [self.vit_model, self.gpt_model]:
                layer.eval()
            elif isinstance(layer, nn.Module):
                layer.train()  # 确保可训练层处于训练模式

    def pool_hidden_state(self, hidden_state, attention_mask):
        """
        对 GPT-2 的输出进行智能池化 (只平均非填充部分)
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_state).float()
        sum_hidden = torch.sum(hidden_state * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_hidden / sum_mask

    def run_timecma_core(self, x_in):
        # x_in: (B, L, M) [Batch, SeqLen, Vars]

        # 1. 适配 TimeCMA (B, V, S)
        x_timecma = x_in.transpose(1, 2)  # (B, M, L)

        # 2. 分解 (在转置后的数据上)
        trend, seasonal = self.decomp(x_timecma)

        # 3. 模态生成
        trend_img, trend_stats = self.trend_generator(trend)
        seasonal_img, seasonal_stats = self.seasonal_generator(seasonal)

        # 4. 动态提示
        batch_size = x_in.shape[0]
        device = x_in.device

        trend_prompts = []
        seasonal_prompts = []

        for i in range(batch_size):
            t_prompt = (f"Trend Analysis for {self.n_vars} variables. "
                        f"[Commonality]: The average slope is {trend_stats['avg_slope'][i]:.2f}, "
                        f"std dev {trend_stats['std_slope'][i]:.2f}. "
                        f"[Magnitude]: Range {trend_stats['min_avg_mag'][i]:.2f} "
                        f"to {trend_stats['max_avg_mag'][i]:.2f}.")
            trend_prompts.append(t_prompt)

            s_prompt = (f"Seasonal Analysis for {self.n_vars} variables. "
                        f"[Commonality]: Dominant period {seasonal_stats['dominant_period'][i]:.1f} steps. "
                        f"{seasonal_stats['period_coherence'][i] * 100:.0f}% coherence. "
                        f"[Magnitude]: Amplitude range {seasonal_stats['min_amp'][i]:.2f} "
                        f"to {seasonal_stats['max_amp'][i]:.2f}.")
            seasonal_prompts.append(s_prompt)

        # 5. 批量编码
        trend_tokens = self.tokenizer(
            trend_prompts, return_tensors='pt', padding=True, truncation=True, max_length=77
        ).to(device)

        seasonal_tokens = self.tokenizer(
            seasonal_prompts, return_tensors='pt', padding=True, truncation=True, max_length=77
        ).to(device)

        # 6. VLM 特征提取 (冻结)
        trend_img_feat = self.vit_model(trend_img).pooler_output
        seasonal_img_feat = self.vit_model(seasonal_img).pooler_output

        trend_text_hidden = self.gpt_model(**trend_tokens).last_hidden_state
        seasonal_text_hidden = self.gpt_model(**seasonal_tokens).last_hidden_state

        trend_text_feat = self.pool_hidden_state(trend_text_hidden, trend_tokens.attention_mask)
        seasonal_text_feat = self.pool_hidden_state(seasonal_text_hidden, seasonal_tokens.attention_mask)

        # 7. 分层交叉注意力融合
        # TimeCMA Query (B, L, M) -> (B, L, C)
        q_time = self.query_proj(x_in)  # (B, L, M) -> (B, L, C)
        q_time = self.ts_att(q_time)  # (B, L, C)

        kv_text = torch.stack([trend_text_feat, seasonal_text_feat], dim=1).double()  # (B, 2, D_VLM)
        kv_image = torch.stack([trend_img_feat, seasonal_img_feat], dim=1).double()  # (B, 2, D_VLM)

        fused_text, _ = self.text_attn(query=q_time, key=kv_text, value=kv_text)  # (B, L, C)
        fused_image, _ = self.image_attn(query=q_time, key=kv_image, value=kv_image)  # (B, L, C)

        # 8. 动态门控融合
        all_features_cat = torch.cat([q_time, fused_text, fused_image], dim=2)  # (B, L, C*3)
        gates = self.fusion_gate_proj(all_features_cat)

        gate_q, gate_text, gate_image = torch.chunk(gates, 3, dim=2)

        fused_all = (gate_q * q_time) + (gate_text * fused_text) + (gate_image * fused_image)  # (B, L, C)

        return fused_all

    def forecast(self, x):
        # 1. CALF 归一化 (B, L, M)
        x_norm = self.normalize_layers(x, mode='norm')

        # 2. TimeCMA 核心
        # (B, L, M) -> (B, L, C)
        fused_all = self.run_timecma_core(x_norm)

        # 3. CALF 预测头
        # (B, L, C) -> (B, P, M)
        outputs = self.out_layer(fused_all)

        # 4. CALF 反归一化
        outputs = self.normalize_layers(outputs, mode='denorm')

        # (TimeCMA 改进版只有一个输出)
        return {
            'outputs': outputs
        }

    def classification(self, x):
        # (B, L, M)

        # 1. TimeCMA 核心 (分类通常不在此阶段归一化)
        fused_all = self.run_timecma_core(x)

        # 2. CALF 预测头
        # (B, L, C) -> (B, NumClass)
        outputs = self.out_layer(fused_all)

        return {
            'outputs': outputs
        }

    def imputation(self, x, mask):
        # 1. CALF 归一化 (B, L, M)
        # (B, L, M)
        x_norm = x.clone()
        x_norm = x_norm.masked_fill(mask == 0, 0)  # 掩码
        x_norm = self.normalize_layers(x_norm, mode='norm')  # 使用掩码后的数据计算均值/标准差
        # 再次应用掩码以防万一
        x_norm = x_norm.masked_fill(mask == 0, 0)

        # 2. TimeCMA 核心
        # (B, L, M) -> (B, L, C)
        fused_all = self.run_timecma_core(x_norm)

        # 3. CALF 预测头
        # (B, L, C) -> (B, L, M)
        outputs = self.out_layer(fused_all)

        # 4. CALF 反归一化
        outputs = self.normalize_layers(outputs, mode='denorm')

        return {
            'outputs': outputs
        }

    def anomaly_detection(self, x):
        # 1. CALF 归一化 (B, L, M)
        x_norm = self.normalize_layers(x, mode='norm')

        # 2. TimeCMA 核心
        # (B, L, M) -> (B, L, C)
        fused_all = self.run_timecma_core(x_norm)

        # 3. CALF 预测头 (重构)
        # (B, L, C) -> (B, L, M)
        outputs = self.out_layer(fused_all)

        # 4. CALF 反归一化
        outputs = self.normalize_layers(outputs, mode='denorm')

        # 异常检测返回的是重构值
        return {
            'outputs': outputs
        }

    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output_dict = self.forecast(x)
        elif self.task_name == 'classification':
            output_dict = self.classification(x)
        elif self.task_name == "imputation":
            output_dict = self.imputation(x, mask)
        elif self.task_name == "anomaly_detection":
            output_dict = self.anomaly_detection(x)
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")

        # 返回字典，以适配 CALF 的训练流程
        # 注意：现在只有一个 'outputs' 键
        return output_dict