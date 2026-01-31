import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from timm.models.convnext import ConvNeXt # 用于类型提示和访问内部结构

class CrossAttention(nn.Module):
    # ... (保持不变)
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        """
        Args:
            x_q (torch.Tensor): 查询序列 (B, N_q, C)
            x_kv (torch.Tensor): 键/值序列 (B, N_kv, C)
        Returns:
            torch.Tensor: 输出序列 (B, N_q, C)
        """
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape

        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, nH, N_q, C/nH
        k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, nH, N_kv, C/nH
        v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, nH, N_kv, C/nH

        attn = (q @ k.transpose(-2, -1)) * self.scale # B, nH, N_q, N_kv
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C) # B, N_q, C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FusionBlock(nn.Module):
    """
    融合块，使用残差连接的交叉注意力来融合两个特征流。
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.norm1_diff = nn.LayerNorm(embed_dim)
        self.norm1_vessel = nn.LayerNorm(embed_dim)
        self.cross_attn_diff_vessel = CrossAttention(embed_dim, num_heads=num_heads)
        self.cross_attn_vessel_diff = CrossAttention(embed_dim, num_heads=num_heads)

        # 可学习的缩放因子，用于残差连接
        self.gamma_diff = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.gamma_vessel = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # MLP 层
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1), # 添加 Dropout 增加鲁棒性
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)  # 添加 Dropout 增加鲁棒性
        )

    def forward(self, feat_diff, feat_vessel):
        """
        Args:
            feat_diff (torch.Tensor): 扩散特征 (B, N, C)
            feat_vessel (torch.Tensor): 血管特征 (B, N, C)
        Returns:
            torch.Tensor: 融合后的特征 (B, N, C)
        """
        res_diff = feat_diff
        res_vessel = feat_vessel

        norm_diff = self.norm1_diff(feat_diff)
        norm_vessel = self.norm1_vessel(feat_vessel)

        # 交叉注意力: diff 查询 vessel 的 K,V
        attn_output_d_v = self.cross_attn_diff_vessel(norm_diff, norm_vessel)
        fused_diff = res_diff + self.gamma_diff * attn_output_d_v # 残差连接

        # 交叉注意力: vessel 查询 diff 的 K,V
        attn_output_v_d = self.cross_attn_vessel_diff(norm_vessel, norm_diff)
        fused_vessel = res_vessel + self.gamma_vessel * attn_output_v_d # 残差连接

        # 简单的平均融合两个增强后的特征，然后通过MLP
        combined_features = (fused_diff + fused_vessel) / 2.0
        # 再加一个残差连接
        fused_output = combined_features + self.mlp(self.norm2(combined_features))

        return fused_output


class EnhancedDecoder(nn.Module):
    """
    轻量化解码器版本，旨在减少参数量。
    确保动态创建的层与输入张量在同一设备上。
    """
    def __init__(self, embed_dim, out_chans, img_size, min_channels=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.img_size = img_size
        self.min_channels = min_channels # 通道数下限

        # ModuleList 用于存储动态创建的层
        self.decoder_layers = nn.ModuleList()
        # 最终卷积层在此定义为 None，将在 forward 中首次创建并赋值
        self.final_conv = None
        # self.final_activation = nn.Tanh() # or nn.Sigmoid() or None
        # self.final_activation = nn.Sigmoid() # or nn.Sigmoid() or None
        self.final_activation=None

    def forward(self, x, feature_map_h, feature_map_w):
        """
        解码器前向传播。
        输入 x: (batch_size, num_patches, embed_dim) - 来自 Encoder 的特征序列
              feature_map_h (int): 输入特征图的高度
              feature_map_w (int): 输入特征图的宽度
        输出: (batch_size, out_chans, img_size, img_size) - 重建的图像
        """
        # 获取输入张量所在的设备
        input_device = x.device

        B, N, C = x.shape
        # --- (之前的断言和 reshape 代码保持不变) ---
        if N != feature_map_h * feature_map_w:
             expected_n = feature_map_h * feature_map_w
             if N > expected_n:
                 print(f"Warning: Input N={N} > H*W={expected_n}. Assuming first token is class token and removing it.")
                 x = x[:, 1:, :]
                 B, N, C = x.shape
             assert N == expected_n, f"Corrected N={N} still doesn't match H*W={expected_n}"
        assert C == self.embed_dim, f"Input C={C}, Expected embed_dim={self.embed_dim}"
        x = x.permute(0, 2, 1) # -> (B, C, N)
        x = x.view(B, self.embed_dim, feature_map_h, feature_map_w) # -> (B, C, H', W')
        # --- (断言和 reshape 代码结束) ---


        # --- 动态构建或选择解码器层 (仅在第一次 forward 时构建) ---
        # 使用 self.training 状态来辅助判断是否需要构建（或者简单地检查列表是否为空）
        # if not self.decoder_layers and self.training: # 也可以只用 not self.decoder_layers
        if not self.decoder_layers:
            print(f"Building lightweight decoder dynamically on device: {input_device}...")
            current_h = feature_map_h
            try:
                 scale_factor = self.img_size / current_h
                 if scale_factor <= 0 or not math.isclose(math.log2(scale_factor), round(math.log2(scale_factor))):
                     raise ValueError(f"img_size ({self.img_size}) must be a power of 2 multiple of feature_map_h ({current_h})")
                 num_upsamples = int(round(math.log2(scale_factor)))
                 print(f"Calculated num_upsamples = {num_upsamples}")
            except ValueError as e:
                 print(f"Error calculating num_upsamples: {e}")
                 num_upsamples = 5 # Fallback

            current_dim = self.embed_dim

            # 用于临时存储在 CPU 上创建的层，然后再移动
            temp_layers = []
            for i in range(num_upsamples):
                next_dim = max(current_dim // 2, self.min_channels)
                if i == num_upsamples - 1:
                    pass # Keep next_dim = max(current_dim // 2, self.min_channels)

                print(f"  Layer {i}: {current_dim} -> {next_dim}")

                # 1. 在 CPU 上创建层块 (默认行为)
                block_layers_cpu = nn.Sequential(
                    nn.ConvTranspose2d(
                        current_dim, next_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(next_dim),
                    nn.GELU()
                )
                # 2. 将创建好的层块移动到目标设备
                block_layers_device = block_layers_cpu.to(input_device)
                # 3. 添加到 ModuleList (它会自动注册移动后的模块)
                self.decoder_layers.append(block_layers_device)
                current_dim = next_dim

            # --- 创建最终卷积层 ---
            print(f"  Final Conv Layer: {current_dim} -> {self.out_chans}")
            # 1. 在 CPU 上创建
            final_conv_cpu = nn.Conv2d(current_dim, self.out_chans, kernel_size=3, stride=1, padding=1)
            # 2. 移动到目标设备
            self.final_conv = final_conv_cpu.to(input_device) # 直接赋值给 self.final_conv

            # 也可以选择创建激活层并移动
            if self.final_activation is not None:
                self.final_activation = self.final_activation.to(input_device)

        # --- 通过解码器层 ---
        # 现在 decoder_layers 中的模块应该与 x 在同一设备上
        for layer_block in self.decoder_layers:
             try:
                 x = layer_block(x)
             except RuntimeError as e:
                 print(f"Error during decoder layer block execution: {e}")
                 # 可以在这里检查层和输入的设备
                 print(f"Input x device: {x.device}")
                 for name, param in layer_block.named_parameters():
                     print(f"Layer block param '{name}' device: {param.device}")
                     break # 只打印第一个参数的设备即可
                 raise e # 重新抛出异常


        # --- 通过最终卷积层 ---
        if self.final_conv:
             try:
                 reconstructed_image = self.final_conv(x)
             except RuntimeError as e:
                 print(f"Error during final conv layer execution: {e}")
                 print(f"Input x device: {x.device}")
                 for name, param in self.final_conv.named_parameters():
                     print(f"Final conv param '{name}' device: {param.device}")
                     break
                 raise e
        else:
            raise RuntimeError("Final convolution layer was not created or not assigned correctly.")

        # --- (后续的激活和尺寸调整代码保持不变) ---
        if self.final_activation:
            reconstructed_image = self.final_activation(reconstructed_image)
        if reconstructed_image.shape[-2:] != (self.img_size, self.img_size):
            print(f"Warning: Decoder output size {reconstructed_image.shape[-2:]} doesn't match target img_size {(self.img_size, self.img_size)}. Using interpolation.")
            reconstructed_image = F.interpolate(reconstructed_image, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        # --- (后续代码结束) ---

        return reconstructed_image


class FeatureDiscriminator(nn.Module):
    # ... (保持不变，但注意 embed_dim 需要匹配 ConvNeXt 输出)
    def __init__(self, embed_dim, hidden_dim=None, n_layers=3, dropout=0.1): # 增加层数和dropout
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim # 保持或增加隐藏层维度

        layers = []
        current_dim = embed_dim
        # 在序列上全局平均池化，得到 (B, C)
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten()) # (B, 1, C) -> (B, C)

        # MLP 层
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else 1 # 最后一层输出1个logit
            layers.append(nn.Linear(current_dim, out_dim))
            if i < n_layers - 1: # 非输出层
                layers.append(nn.LayerNorm(out_dim)) # 使用 LayerNorm
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                layers.append(nn.Dropout(dropout))
                current_dim = out_dim

        # Sigmoid 不在这里应用，由 BCEWithLogitsLoss 处理
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征序列 (B, N, C)
        Returns:
            torch.Tensor: 判别器输出 logits (B, 1)
        """
        if x.dim() == 3:
            # 需要将 (B, N, C) 调整为 AdaptiveAvgPool1d 需要的 (B, C, N)
            x = x.permute(0, 2, 1)
        elif x.dim() == 2:
             # 如果输入已经是 (B, C)
             pass
        else:
             raise ValueError(f"Unsupported input dimension: {x.dim()}")

        return self.model(x)

class ImageFusionModel(nn.Module):
    """
    整合 ConvNeXt 编码器、融合块和增强解码器的图像融合模型 (生成器部分)。
    """
    def __init__(self, encoder_name='convnext_base', img_size=384, in_chans=3, num_classes=0, pretrained=True):
        super().__init__()
        self.img_size = img_size
        # self.patch_size = patch_size # ConvNext 不直接用 patch_size 定义结构
        self.in_chans = in_chans

        print(f"Loading Encoder: {encoder_name}")
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=num_classes, # 不使用分类头
            in_chans=in_chans,
            features_only=False # 需要最终的特征输出，可能需要 forward_features
        )
        
        # 无网连接，加
        # 如果使用预训练权重，加载本地文件
        if pretrained:
            # 本地权重文件路径
            local_weight_path = '/home/230320040445/MI-Fusion/Ours/model_convNextWeight/pytorch_model.bin'
            
            # 加载预训练权重
            if torch.cuda.is_available():
                state_dict = torch.load(local_weight_path, map_location='cuda:0')
            else:
                state_dict = torch.load(local_weight_path, map_location='cpu')
                
            # 处理可能的键不匹配问题
            new_state_dict = {}
            for k, v in state_dict.items():
                # 移除可能的前缀（如'module.'或'model.'）
                while k.startswith('module.') or k.startswith('model.'):
                    k = k[len('module.'):] if k.startswith('module.') else k[len('model.'):]
                new_state_dict[k] = v
                
            # 加载状态字典
            self.encoder.load_state_dict(new_state_dict, strict=False)
            print(f"Successfully loaded pretrained weights from {local_weight_path}")
            
            

        # 获取 ConvNeXt 输出特征维度
        self.embed_dim = self.encoder.num_features # 通常是分类头前的特征维度
        # 或者，如果模型结构不同，可能需要检查 self.encoder_diff.num_features (如果存在)
        # 或者直接看 forward_features 的输出通道数
        print(f"Detected Encoder embed_dim: {self.embed_dim}")


        # 确保两个编码器参数独立 (如果需要的话，否则可以共享)
        # self.encoder_vessel = timm.create_model(...) # 重新加载以确保独立

        self.fusion_block = FusionBlock(embed_dim=self.embed_dim)
        self.decoder = EnhancedDecoder( # 使用增强解码器
            embed_dim=self.embed_dim,
            out_chans=in_chans,
            img_size=img_size,
            # feature_map_h/w 在 forward 中确定
        )
        # self.decoder_fusion = EnhancedDecoder(...) # 如果需要独立的解码器


    def encode(self, encoder, x):
        """
        使用 ConvNeXt 提取特征并塑形。
        输出: (B, N, C) 特征序列, H_feat, W_feat
        """
        # ConvNeXt 通常没有简单的 forward_features 返回期望的序列
        # 我们需要执行到最后一个 stage 但在 head (分类器) 之前
        # timm ConvNeXt 模型通常可以通过 forward_features 实现
        features = encoder.forward_features(x) # 输出通常是 (B, C, H', W')
        B, C, H_feat, W_feat = features.shape

        # 展平为序列 (B, N, C) 以适配 FusionBlock
        features_seq = features.flatten(2).permute(0, 2, 1) # (B, C, H'*W') -> (B, H'*W', C)

        return features_seq, H_feat, W_feat

    def forward(self, I_diff, I_vessel):
        """
        Args:
            I_diff (torch.Tensor): 扩散图像 (B, C, H, W)
            I_vessel (torch.Tensor): 血管图像 (B, C, H, W)
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                pRE_diff, pRE_vessel, pRE_fusion, feat_diff, feat_vessel, fused_features
        """
        # 1. Encode
        feat_diff, H_diff, W_diff = self.encode(self.encoder, I_diff)       # (B, N, C), H', W'
        feat_vessel, H_ves, W_ves = self.encode(self.encoder, I_vessel) # (B, N, C), H', W'

        # 确保特征图大小一致 (理论上输入相同尺寸图像和相同encoder应该一致)
        assert H_diff == H_ves and W_diff == W_ves, "Feature map sizes from encoders mismatch!"
        H_feat, W_feat = H_diff, W_diff

        # 2. Fusion
        fused_features = self.fusion_block(feat_diff, feat_vessel) # (B, N, C)

        # 3. Decode
        # 将特征图 H, W 传递给解码器
        pRE_diff = self.decoder(feat_diff, H_feat, W_feat)       # (B, C, H, W)
        pRE_vessel = self.decoder(feat_vessel, H_feat, W_feat) # (B, C, H, W)
        pRE_fusion = self.decoder(fused_features, H_feat, W_feat)# (B, C, H, W)

        return pRE_diff, pRE_vessel, pRE_fusion, feat_diff, feat_vessel, fused_features

class FusionAdversarialModel(nn.Module):
    """
    包含生成器和判别器的完整模型。
    """
    def __init__(self, encoder_name='convnext_tiny', img_size=384, in_chans=3, pretrained=True):
        super().__init__()

        # 创建生成器以获取 embed_dim
        # 注意：这里创建了一次生成器，然后下面的 generator 属性又创建了一次
        # 稍作修改，先创建，再获取 embed_dim
        temp_generator = ImageFusionModel(
            encoder_name=encoder_name,
            img_size=img_size,
            in_chans=in_chans,
            pretrained=pretrained
        )
        self.generator = temp_generator
        embed_dim = self.generator.embed_dim # 从已创建的生成器获取 embed_dim

        print(f"Initializing Discriminator with embed_dim: {embed_dim}")
        self.discriminator = FeatureDiscriminator(
            embed_dim=embed_dim,
            n_layers=3 # 可以调整判别器深度
        )

    def forward(self, I_diff, I_vessel):
        # 主要用于推理或生成阶段调用生成器
        return self.generator(I_diff, I_vessel)

# --- 主程序入口 ---
if __name__ == '__main__':
    IMG_SIZE = 512
    # PATCH_SIZE = 16 # 对于 ConvNeXt 不再直接相关
    IN_CHANS = 1 # 改为单通道
    BATCH_SIZE = 2 # 减少 Batch Size 以防 OOM，因为模型变大了
    # 选择一个 ConvNeXt 模型
    ENCODER_NAME = 'convnext_tiny'
    # 注意: ConvNeXt Base 的 embed_dim (head输入维度) 通常是 1024
    # 如果使用 convnext_tiny (384), convnext_small (768), 需要相应修改 embed_dim
    # 这里我们基于 encoder_name 创建模型后动态获取，所以不需要手动指定 embed_dim

    print("Creating FusionAdversarialModel...")
    model = FusionAdversarialModel(
        encoder_name=ENCODER_NAME,
        img_size=IMG_SIZE,
        in_chans=IN_CHANS,
        pretrained=True # 使用预训练权重
    )
    print("Model Created.")

    # 创建虚拟输入数据
    dummy_diff = torch.randn(BATCH_SIZE, IN_CHANS, IMG_SIZE, IMG_SIZE)
    dummy_vessel = torch.randn(BATCH_SIZE, IN_CHANS, IMG_SIZE, IMG_SIZE)
    print(f"Input shape: {dummy_diff.shape}")

    # --- 测试生成器 ---
    print("\nTesting Generator...")
    model.generator.eval()
    with torch.no_grad():
        try:
            pRE_diff, pRE_vessel, pRE_fusion, feat_diff, feat_vessel, feat_fusion = model.generator(dummy_diff, dummy_vessel)

            print("Generator Output Shapes:")
            print(f"  pRE_diff: {pRE_diff.shape}")
            print(f"  pRE_vessel: {pRE_vessel.shape}")
            print(f"  pRE_fusion: {pRE_fusion.shape}")
            print(f"  feat_diff: {feat_diff.shape}")
            print(f"  feat_vessel: {feat_vessel.shape}")
            print(f"  feat_fusion: {feat_fusion.shape}")

            # --- 测试判别器 ---
            print("\nTesting Discriminator...")
            model.discriminator.eval()
            # 使用生成器输出的特征作为判别器输入
            disc_output_diff = model.discriminator(feat_diff)
            disc_output_vessel = model.discriminator(feat_vessel)
            disc_output_fusion = model.discriminator(feat_fusion) # 判别融合后的特征

            print("\nDiscriminator Output Shapes (Logits):")
            print(f"  Disc(feat_diff): {disc_output_diff.shape}")
            print(f"  Disc(feat_vessel): {disc_output_vessel.shape}")
            print(f"  Disc(feat_fusion): {disc_output_fusion.shape}")

        except Exception as e:
            print(f"\nError during model testing: {e}")
            import traceback
            traceback.print_exc()


    # --- 计算参数量 ---
    print("\nCalculating Parameter Counts...")
    try:
        num_gen_params = sum(p.numel() for p in model.generator.parameters() if p.requires_grad)
        num_dec_params = sum(p.numel() for p in model.generator.decoder.parameters() if p.requires_grad) # 单独计算解码器参数
        num_enc_params = sum(p.numel() for p in model.generator.encoder.parameters() if p.requires_grad) # 单个编码器参数 (假设两个一样)
        num_fus_params = sum(p.numel() for p in model.generator.fusion_block.parameters() if p.requires_grad) # 融合块参数
        num_disc_params = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)

        print(f"\nNumber of trainable parameters:")
        print(f"  Encoder (x2): {(num_enc_params / 1e6)*2:.2f} M")
        print(f"  Fusion Block: {num_fus_params / 1e6:.2f} M")
        print(f"  Decoder: {num_dec_params / 1e6:.2f} M")
        print(f"  -----------------------------")
        print(f"  Total Generator: {num_gen_params / 1e6:.2f} M")
        print(f"  Discriminator: {num_disc_params / 1e6:.2f} M")
        print(f"  -----------------------------")
        print(f"  Total Model: {(num_gen_params + num_disc_params) / 1e6:.2f} M")

    except Exception as e:
        print(f"\nError calculating parameters: {e}")