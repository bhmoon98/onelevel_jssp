import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
#보현씨가 보내준거
from torch.autograd import Variable

import os
import sys
import copy
import math
import numpy as np
from typing import List

class LearnedPositionEncoding1(nn.Embedding): # LearnedPositionEncoding1은 global position embedding
    def __init__(self,d_model, dropout = 0.1, input_size = 7):
        super().__init__(d_model, input_size*input_size)
        self.input_size = input_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.view(self.d_model, self.input_size, self.input_size, ).unsqueeze(0)
        x = x + weight
        return self.dropout(x)


class LearnedPositionEncoding2(nn.Embedding): # LearnedPositionEncoding2은 local position embedding
    def __init__(self,d_model, dropout = 0.1, input_size = 7):
        super().__init__(input_size*input_size, d_model)
        self.input_size = input_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.view(self.input_size * self.input_size, self.d_model).unsqueeze(1)
        x = x + weight
        return self.dropout(x)


class UTransformer(nn.Module):
    def __init__(self, args, input_dim: int = 100*20, patch_size: List[int] = [5,5,5], d_model: int = 512, 
                nhead: int = 8, num_encoder_layers: List[int] = [5,5,5],
                num_decoder_layers: List[int] = [5,5,5], dim_feedforward: int = 2048, dropout: float = 0.1, sparse_embedding_dim: List[int]=[100*20, 25*20, 5*20]):
        super(UTransformer, self).__init__()

        # zero embedding이 원래 다음 level encoder애 들어가는 header latent vector의 역할인데 여기서는 뭐로 해야할지?
        # self.zeros = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=False)
        self.d_model = d_model
        self.patch_size = patch_size

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # 만약 들어오는 vector가 conjunctive, disjuncitve한 의미를 가지고 있으면 필요한지 의문
        # 만약 graph similarity만 나타내는 vector라면 필요할 수 있음
        # self.global_position_embedding = LearnedPositionEncoding1(d_model = d_model, dropout = dropout, input_size = np.prod(patch_size))
        # self.position_embedding = nn.ModuleList([
        #     LearnedPositionEncoding2(d_model = d_model, dropout = dropout, input_size = patch_size[i]) \
        #         for i in range(len(patch_size))])

        # sparse embedding이 들어온다고 가정
        # self.sparse_embedding = nn.ModuleList([nn.Embedding(patch_size[i] * patch_size[i], d_model) for i in range(len(patch_size))])
        # self.sparse_embedding = nn.ModuleList([nn.Embedding(sparse_embedding_dim[i], d_model) for i in range(len(patch_size))])
        # node2vec한 후에 들어오는 vector를 변환, 아니면 query_embedding 부분을 sparse embedding 부분으로 사용
        self.query_embedding = nn.ModuleList([nn.Embedding(patch_size[i] * patch_size[i], d_model) for i in range(len(patch_size))])

        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, 
                        dim_feedforward = dim_feedforward, dropout = dropout, activation = args.activation)
                        , num_encoder_layers[i], encoder_norm) for i in range(len(patch_size))])

        self.decoder = nn.ModuleList([
            nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model = d_model, nhead = nhead, 
                        dim_feedforward = dim_feedforward, dropout = dropout, activation = args.activation)
                        , num_decoder_layers[i], decoder_norm) for i in range(len(patch_size))])

        self.bottle_neck = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model//patch_size[i]), nn.LayerNorm(d_model//patch_size[i]), nn.GELU(),
            nn.Linear(d_model//patch_size[i], d_model), nn.LayerNorm(d_model)) 
            for i in range(len(patch_size))])

        
        self.pre_conv = nn.Conv2d(input_dim, d_model, kernel_size=1, bias=False)

        # 원래 recon error랑 비교하기 위해 input_dim으로 바꿔주는 부분
        self.final_layer1 = nn.Conv2d(d_model, input_dim, kernel_size=1, bias=False)
        
        # 최종 MLP 부분
        self.final_layer2 = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(d_model),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(d_model, d_model // 2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(d_model // 2),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(d_model // 2, d_model // 4, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(d_model // 4),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(d_model // 4, d_model // 8, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(d_model // 8),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(d_model // 8, 1, kernel_size=1, bias=False))
            
    def calculate_size(self, level):
        S = self.patch_size[level]
        P = 1
        for i in range(level+1, len(self.patch_size), 1):
            P *= self.patch_size[i]
        return P, S

    def forwardDOWN(self, x, encoder_block, position_embedding, sparse_embedding, level):
        # B는 batch size, P는 patch division multiplier, S는 현재 patch_size, C는 channel size
        _, BPSPS, C = x.size()
        P, S = self.calculate_size(level)
        B = BPSPS // (P*S*P*S)
        x = x.view(B, P, S, P, S, C).permute(2,4,0,1,3,5).contiguous().view(S*S, B*P*P, C) #(SS, BPP, C)
        pad = self.zeros.expand(-1, B*P*P, -1)

        # 이 식에 sparse embedding을 cat하면 가능할 것 같다.
        x = encoder_block(src = torch.cat((pad.detach(), position_embedding(x), sparse_embedding), dim=0))

        latent_patch = x[0,:,:].unsqueeze(0).contiguous() #(1, BPP, C)
        latent_pixel = x[1:,:,:].contiguous() #(SS, BPP, C)
        #print(x.size())

        return latent_patch, latent_pixel


    def forwardUP(self, latent_patch, latent_pixel, decoder_block, query, level):
        SS, BPP, C = latent_pixel.size()
        #1, BPP, C = latent_patch.size()
        P, S = self.calculate_size(level)
        B = BPP // (P*P)
        latent = torch.cat((latent_patch, latent_pixel),dim=0)
        out = decoder_block(memory = latent, tgt = query.weight.unsqueeze(1).expand(-1, BPP, -1) ) #(SS, BPP, C)
        out = out.view(S, S, B, P, P, C).permute(2,3,0,4,1,5).contiguous().view(1, B*P*S*P*S, C) #(1, BSPSP, C)
        return out
        
    
    def forward(self, x):
        x = self.pre_conv(x)
        B, C, H, W = x.size()  #(B, C, H, W)
        # x = self.global_position_embedding(x)
        x = x.permute(0,2,3,1).contiguous().view(B*H*W,C).unsqueeze(0) #(1, BHW, C)
        latent_list = []
        for i in range(len(self.encoder)):
            #sparse embedding 추가해서 forwardDOWN 수정해야함
            x, l = self.forwardDOWN(x=x, encoder_block=self.encoder[i], position_embedding=self.position_embedding[i], sparse_embedding=self.sparse_embedding, level=i)
            latent_list.append(self.bottle_neck[i](l))
        for i in range(len(self.encoder)-1, -1, -1):
            x = self.forwardUP(latent_patch=x, latent_pixel=latent_list[i], decoder_block=self.decoder[i], query=self.query_embedding[i], level=i)
        x = x.squeeze(0).view(B, H, W, C).permute(0,3,1,2).contiguous()
        #out = self.final_layer1(x)
        return self.final_layer1(x), self.final_layer2(x.detach())