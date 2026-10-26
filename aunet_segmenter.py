"""
Module AUNet (Attention on U-Net) Phân Đoạn Ngập Lụt Đô Thị Chuyên Sâu
Dựa trên bài báo khoa học:
"Urban Waterlogging Monitoring and Recognition in Low-Light Scenarios Using Surveillance Videos and Deep Learning" (Water 2025, MDPI)

Kiến trúc gồm:
1. DisLAModule: Tách thành phần chiếu sáng L và cấu trúc bề mặt nước bất biến A = X - L.
2. BrightnessAttention: Swin Transformer Block (W-MSA) trích xuất ngữ cảnh chùm sáng đèn xe/đèn đường.
3. Semantic U-Net: Phân đoạn xác suất từng pixel (Pixel-level probability map) thay vì bounding box.
4. Lớp bao bọc AUNetFloodSegmenter: Tương thích 100% API predict() của Ultralytics YOLO.
"""

import math
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4, window_size=7, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        x_perm = x.permute(0, 2, 3, 1)
        num_h_win = Hp // self.window_size
        num_w_win = Wp // self.window_size
        x_win = x_perm.view(B, num_h_win, self.window_size, num_w_win, self.window_size, C)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        shortcut = x_win
        attn_out = self.attn(self.norm1(x_win))
        x_win = shortcut + attn_out
        x_win = x_win + self.mlp(self.norm2(x_win))

        x_perm = x_win.view(B, num_h_win, num_w_win, self.window_size, self.window_size, C)
        x_perm = x_perm.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        if pad_h > 0 or pad_w > 0:
            x_perm = x_perm[:, :H, :W, :]

        return x_perm.permute(0, 3, 1, 2)


class BrightnessAttention(nn.Module):
    def __init__(self, in_channels=3, mid_channels=128):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.swin1 = SwinTransformerBlock(dim=mid_channels, num_heads=4, window_size=7)
        self.swin2 = SwinTransformerBlock(dim=mid_channels, num_heads=4, window_size=7)
        self.conv_out = nn.Conv2d(mid_channels, 16, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        f = self.maxpool(x)
        f = self.gelu(self.conv1x1(f))
        f = self.swin1(f)
        f = self.swin2(f)
        f = self.sigmoid(self.conv_out(f))
        f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
        return f


class DisLAModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv_lf = ConvBlock(in_channels, 32, kernel_size=7, padding=3)
        self.conv_mid = ConvBlock(32, 64, kernel_size=7, padding=3)
        self.conv_hf = ConvBlock(64, 128, kernel_size=7, padding=3)

        self.brightness_attn = BrightnessAttention(in_channels=in_channels, mid_channels=128)
        self.attn_proj = nn.Conv2d(16, 32, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.global_relu = nn.ReLU(inplace=True)

        self.proj_d = nn.Conv2d(32 + 32 + 128 + 32, 128, kernel_size=1)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)

        self.up_l = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, padding=1),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=1)
        )

        self.up_a = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, padding=1),
            ConvBlock(64, 32, kernel_size=3, padding=1)
        )
        self.conv_a_out = ConvBlock(32, 64, kernel_size=3, padding=1)

    def forward(self, x):
        lf = self.conv_lf(x)
        mid = self.conv_mid(lf)
        hf = self.conv_hf(mid)

        g_feat = self.global_relu(self.global_conv(self.global_pool(hf)))
        g_feat = g_feat.expand(-1, -1, hf.shape[2], hf.shape[3])

        b_attn = self.attn_proj(self.brightness_attn(x))

        fused = torch.cat([lf, g_feat, hf, b_attn], dim=1)
        d = self.proj_d(fused)
        d_res = self.res2(self.res1(d))

        l_disla = self.up_l(d_res)
        a_feat = self.up_a(d_res)
        a_disla = a_feat - l_disla + lf
        a_out = self.conv_a_out(a_disla)

        return a_out, l_disla


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, padding=1),
            ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=67, num_classes=2):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 128)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(128, 256)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(256, 512)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256, 128)

        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_conv(d1)


class AUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.disla = DisLAModule(in_channels=in_channels)
        self.unet = UNet(in_channels=64 + in_channels, num_classes=num_classes)

    def forward(self, x):
        a_feat, _ = self.disla(x)
        fused_input = torch.cat([a_feat, x], dim=1)
        logits = self.unet(fused_input)
        return logits


class AUNetMasks:
    def __init__(self, data_tensor):
        self.data = data_tensor


class AUNetPredictResult:
    def __init__(self, mask_tensor, orig_shape):
        self.masks = AUNetMasks(mask_tensor)
        self.orig_shape = orig_shape


class AUNetFloodSegmenter:
    """
    Lớp bao bọc suy luận AUNet SOTA với giao diện chuẩn .predict()
    tương thích hoàn toàn 100% với Ultralytics YOLO trong hệ thống đo ngập lụt.
    """
    def __init__(self, weights_path=None, device=None, img_size=(256, 384)):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.img_size = img_size  # (H, W) = (256, 384)
        self.model_name = Path(weights_path).name if weights_path else "aunet_flood_sota.pth"
        self.model = AUNet(in_channels=3, num_classes=2)

        if weights_path and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt)
            print(f"   [+] AUNet SOTA: Đã nạp thành công trọng số từ {Path(weights_path).name}!")
        self.model.to(self.device)
        self.model.eval()

    def to(self, device):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.model.to(self.device)
        return self

    def predict(self, frame_bgr, conf=0.25, imgsz=800, retina_masks=True, device=None, verbose=False, **kwargs):
        """
        Giao diện chuẩn tương thích 100% với Ultralytics YOLO predict API.
        Nhận vào frame_bgr (H, W, 3), trả về [AUNetPredictResult] với .masks.data shape (1, H, W).
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # Tiền xử lý RGB + Resize + Normalize
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        target_h, target_w = self.img_size
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_float = resized.astype(np.float32) / 255.0

        # Normalize ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normed = (img_float - mean) / std

        tensor = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            autocast_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(autocast_device, enabled=(self.device.type == 'cuda')):
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
                water_prob = probs[:, 1:2, :, :].float()  # shape (1, 1, 256, 384)

                # Resize về kích thước gốc (orig_h, orig_w)
                water_prob_orig = F.interpolate(water_prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                mask_data = water_prob_orig.squeeze(1)  # shape (1, orig_h, orig_w)

                # Áp dụng ngưỡng conf
                conf_val = float(conf) if conf is not None else 0.25
                mask_data = torch.where(mask_data >= conf_val, mask_data, torch.zeros_like(mask_data))

        return [AUNetPredictResult(mask_data, (orig_h, orig_w))]
