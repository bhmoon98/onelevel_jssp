import torch.nn as nn
import torch
import random

class UTransformer(nn.Module):
    def __init__(self, args, input_dim: int = 100 * 20, patch_size: List[int] = [5, 5, 5], d_model: int = 512,
                 nhead: int = 8, num_encoder_layers: List[int] = [5, 5, 5],
                 num_decoder_layers: List[int] = [5, 5, 5], dim_feedforward: int = 2048, dropout: float = 0.1):
        super(UTransformer, self).__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

       # Level�� ����ŷ�� ���� ũ�� ����
        self.level2_rows_to_remove = 70
        self.level2_cols_to_remove = 5
        self.level3_rows_to_remove = 15

        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                             dim_feedforward=dim_feedforward, dropout=dropout,
                                                             activation=args.activation), num_encoder_layers[i],
                                  encoder_norm) for i in range(len(patch_size))])

        self.decoder = nn.ModuleList([
            nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                             dim_feedforward=dim_feedforward, dropout=dropout,
                                                             activation=args.activation), num_decoder_layers[i],
                                  decoder_norm) for i in range(len(patch_size))])

        self.bottle_neck = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model // patch_size[i]), nn.LayerNorm(d_model // patch_size[i]),
                          nn.GELU(),
                          nn.Linear(d_model // patch_size[i], d_model), nn.LayerNorm(d_model))
            for i in range(len(patch_size))])

        self.pre_conv = nn.Conv2d(input_dim, d_model, kernel_size=1, bias=False)

        self.final_layer1 = nn.Conv2d(d_model, input_dim, kernel_size=1, bias=False)

    # ���� 2 ����ŷ ���� �Լ�
    def mask_level2(self, x):
        # �������� 70�� ���� ����
        rows_to_remove = random.sample(range(x.size(-2)), self.level2_rows_to_remove)
        x[:, rows_to_remove, :] = 0

        # ���� 5�� ���� ����
        x[:, :, :self.level2_cols_to_remove] = 0
        return x

    # ���� 3 ����ŷ ���� �Լ�
    def mask_level3(self, x):
        # �������� 15�� ���� ����
        rows_to_remove = random.sample(range(x.size(-2)), self.level3_rows_to_remove)
        x[:, rows_to_remove, :] = 0
        return x

    def forwardDOWN(self, x, encoder_block, level_mask_fn):
        # ���� ����ŷ �Լ� ȣ��
        x = level_mask_fn(x)
        x = encoder_block(x)
        return x

    def forward(self, x):
        x = self.pre_conv(x)
        B, C, H, W = x.size()
        x = x.view(B * H * W, C).unsqueeze(0)

        # ������ ����ŷ �Լ� ����
        x = self.forwardDOWN(x, self.encoder[1], self.mask_level2)
        x = self.forwardDOWN(x, self.encoder[2], self.mask_level3)

        return self.final_layer1(x)
