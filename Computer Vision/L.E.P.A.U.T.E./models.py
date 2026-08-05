import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, Any

from globals import logger, mps_safe
from pipeline_and_config import LepauteConfig 
from geometry import se3_exp_map

class MonocularSE3Warping(nn.Module):
    def __init__(self, config: LepauteConfig):
        super().__init__()
        self.config = config

    def forward(self, img: torch.Tensor, xi: torch.Tensor, scale_prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = img.shape
        device = img.device
        
        vy, vx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        sp_expanded = scale_prior.view(B, 1, 1).expand(B, H, W)
        Z_proxy = torch.ones(B, H, W, device=device, dtype=torch.float32) * sp_expanded
        
        X = (vx.expand(B, H, W) - self.config.cx) * Z_proxy / self.config.fx
        Y = (vy.expand(B, H, W) - self.config.cy) * Z_proxy / self.config.fy

        P = torch.stack((X, Y, Z_proxy, torch.ones_like(Z_proxy)), dim=3).view(B, -1, 4)
        T = se3_exp_map(xi)
        P_t = torch.bmm(T, P.transpose(1, 2)).transpose(1, 2)

        X_t, Y_t, Z_t = P_t[:, :, 0], P_t[:, :, 1], P_t[:, :, 2]
        Z_t_safe = torch.clamp(Z_t, min=1e-3)
        u_t = self.config.fx * X_t / Z_t_safe + self.config.cx
        v_t = self.config.fy * Y_t / Z_t_safe + self.config.cy

        u_norm = (u_t / (W - 1)) * 2.0 - 1.0
        v_norm = (v_t / (H - 1)) * 2.0 - 1.0
        
        # FIXED: dim=2 caused a corrupted (B, H, 2, W) shape when passed to view. 
        # Using dim=-1 yields the correct (B, H, W, 2) shape natively for grid_sample.
        grid = torch.stack((u_norm, v_norm), dim=-1)

        warped_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        valid_mask = ((u_t >= 0) & (u_t <= W - 1) & (v_t >= 0) & (v_t <= H - 1) & (Z_t > 0.1)).view(B, 1, H, W).float()
        return warped_img, valid_mask

class SE3CrossAttentionBlock(nn.Module):
    """
    Cross-Attention block to correlate reference and current frame features.
    Utilizes Pre-LayerNorm architecture for training stability.
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_k = nn.LayerNorm(dim)
        self.norm1_v = nn.LayerNorm(dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Features from reference image (B, N, C)
            key: Features from current image (B, N, C)
            value: Features from current image (B, N, C)
        Returns:
            Attended features (B, N, C)
        """
        q = self.norm1_q(query)
        k = self.norm1_k(key)
        v = self.norm1_v(value)
        
        attn_output, _ = self.cross_attn(query=q, key=k, value=v)
        x = query + attn_output
        
        x = x + self.mlp(self.norm2(x))
        return x

class SE3ResidualRefiner(nn.Module):
    """
    Refines the initial SE(3) pose estimate and dynamically resolves monocular scale ambiguity.
    Predicts the residual pose delta_xi, a scale correction factor delta_s, and their joint uncertainties.
    """
    def __init__(self, config: LepauteConfig, feature_dim: int = 256, max_resolution: int = 64):
        super().__init__()
        self.config = config
        
        # 1. Pre-trained Backbone (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, # 1/4
            resnet.layer1, # 1/4, 64 channels
            resnet.layer2, # 1/8, 128 channels
            resnet.layer3  # 1/16, 256 channels
        )
        
        # 2. Lightweight Monocular Depth Geometry Head
        self.depth_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, feature_dim, max_resolution, max_resolution) * 0.02)
        
        # 4. Pose Conditioning Embedding
        self.pose_emb = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, feature_dim)
        )
        
        # 5. Cross Attention Block
        self.cross_attn = SE3CrossAttentionBlock(dim=feature_dim, num_heads=8, dropout=0.1)
        
        # 6. Joint Pose & Scale Regression Head
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim * 2 + 1, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 14)
        )
        
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # Removed internal sub-module torch.compile hooks. Partial module compilation can lead 
        # to fragmented graph optimization and inconsistent fallback states. Standard graph 
        # compilation is now orchestrated exclusively at the top-level outer loop.
        
        self.to(self.config.device)

    def load_compiled_state_dict(self, state_dict: Dict[str, Any]):
        current_state = self.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            clean_k = k.replace("_orig_mod.", "")
            if clean_k in current_state:
                if current_state[clean_k].shape == v.shape:
                    new_state_dict[clean_k] = v
                else:
                    logger.warning(
                        f"[SE3ResidualRefiner] Shape mismatch on layer '{clean_k}'. "
                        f"Dropping checkpoint weight and using fresh initialization."
                    )
            else:
                new_state_dict[clean_k] = v
                
        return self.load_state_dict(new_state_dict, strict=False)

    def forward(self, img_ref: torch.Tensor, img_cur: torch.Tensor, xi_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            img_ref: Reference image tensor (B, 3, H, W)
            img_cur: Current image tensor (B, 3, H, W)
            xi_init: Initial relative pose estimate in se(3) tangent space (B, 6)
        Returns:
            delta_xi: Residual se(3) tangent vector (B, 6)
            delta_scale: Multiplicative scale correction factor (B, 1)
            uncertainty_pose: Log variance of the pose prediction (B, 6)
            uncertainty_scale: Log variance of the scale prediction (B, 1)
        """
        is_unbatched = img_ref.dim() == 3
        if is_unbatched:
            img_ref = img_ref.unsqueeze(0)
            img_cur = img_cur.unsqueeze(0)
            xi_init = xi_init.unsqueeze(0)
            
        B = img_ref.shape[0]
        
        # 1. Feature Extraction
        f_ref = self.backbone(img_ref) # (B, 256, H/16, W/16)
        f_cur = self.backbone(img_cur)
        
        _, C, H_f, W_f = f_ref.shape
        
        # 2. Monocular Depth Estimation Guidance
        depth_map = self.depth_head(f_cur) # (B, 1, H_f, W_f)
        
        # 3. Add Spatial Positional Embeddings
        pos = F.interpolate(self.pos_embed, size=(H_f, W_f), mode='bilinear', align_corners=False)
        f_ref = f_ref + pos
        f_cur = f_cur + pos
        
        # 4. Inject Pose Conditioning
        p_emb = self.pose_emb(xi_init).view(B, C, 1, 1).expand(-1, -1, H_f, W_f)
        f_cur = f_cur + p_emb
        
        # 5. Spatial Sequence Attention
        f_ref_flat = f_ref.view(B, C, -1).permute(0, 2, 1)
        f_cur_flat = f_cur.view(B, C, -1).permute(0, 2, 1)
        f_attn_flat = self.cross_attn(query=f_ref_flat, key=f_cur_flat, value=f_cur_flat)
        f_attn = f_attn_flat.permute(0, 2, 1).view(B, C, H_f, W_f)
        
        # 6. Combined Geometric Regression
        combined_feat = torch.cat([f_ref, f_attn, depth_map], dim=1) # (B, 513, H_f, W_f)
        out = self.head(combined_feat)
        
        delta_xi = out[:, 0:6]
        delta_scale = out[:, 6:7]
        uncertainty_pose = out[:, 7:13]
        uncertainty_scale = out[:, 13:14]
        
        if is_unbatched:
            delta_xi = delta_xi.squeeze(0)
            delta_scale = delta_scale.squeeze(0)
            uncertainty_pose = uncertainty_pose.squeeze(0)
            uncertainty_scale = uncertainty_scale.squeeze(0)
            
        return delta_xi, delta_scale, uncertainty_pose, uncertainty_scale
        
    def export_onnx(self, output_path: str = "lepaute_refiner.onnx"):
        self.eval()
        device = next(self.parameters()).device
        dummy_a = torch.randn(1, 3, self.config.orig_h, self.config.orig_w, device=device)
        dummy_b = torch.randn(1, 3, self.config.orig_h, self.config.orig_w, device=device)
        dummy_xi = torch.randn(1, 6, device=device)
        
        torch.onnx.export(
            self, 
            (dummy_a, dummy_b, dummy_xi),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['img_a', 'img_b', 'xi_prior'],
            output_names=['delta_xi', 'delta_scale', 'uncertainty_pose', 'uncertainty_scale'],
            dynamic_axes={'img_a': {0: 'batch_size'}, 'img_b': {0: 'batch_size'}, 'xi_prior': {0: 'batch_size'}}
        )
        logger.info(f"[SE3ResidualRefiner] Successfully exported static graph to {output_path}")

    def to_quantized_cpu(self) -> nn.Module:
        self.to("cpu")
        self.eval()
        quantized_model = torch.ao.quantization.quantize_dynamic(
            self, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        logger.info("[SE3ResidualRefiner] Successfully converted internal Linear layers to 8-bit precision.")
        return quantized_model