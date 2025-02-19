from functools import partial
from typing import Any, Callable, List, Union

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1_score
from torchvision.models.vision_transformer import EncoderBlock
from typing_extensions import OrderedDict

from tubevit.positional_encoding import get_3d_sincos_pos_embed


class Encoder(nn.Module):
    """
    Transformer Model Encoder for sequence to sequence translation.
    Code from torch.
    Move pos_embedding to TubeViT
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: Tensor):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        return self.ln(self.layers(self.dropout(x)))


class SparseTubesTokenizer(nn.Module):
    def __init__(self, hidden_dim, kernel_sizes, strides, offsets):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.offsets = offsets

        self.conv_proj_weight = nn.Parameter(
            torch.empty((self.hidden_dim, 3, *self.kernel_sizes[0])).normal_(), requires_grad=True
        )

        self.register_parameter("conv_proj_weight", self.conv_proj_weight)

        self.conv_proj_bias = nn.Parameter(torch.zeros(len(self.kernel_sizes), self.hidden_dim), requires_grad=True)
        self.register_parameter("conv_proj_bias", self.conv_proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, t, h, w = x.shape  # CTHW
        tubes = []
        for i in range(len(self.kernel_sizes)):
            if i == 0:
                weight = self.conv_proj_weight
            else:
                weight = F.interpolate(self.conv_proj_weight, self.kernel_sizes[i], mode="trilinear")

            tube = F.conv3d(
                x[:, :, self.offsets[i][0] :, self.offsets[i][1] :, self.offsets[i][2] :],
                weight,
                bias=self.conv_proj_bias[i],
                stride=self.strides[i],
            )

            tube = tube.reshape((n, self.hidden_dim, -1))

            tubes.append(tube)

        x = torch.cat(tubes, dim=-1)
        x = x.permute(0, 2, 1).contiguous()
        return x


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf

    code from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """

        # (N, T, H) -> (N, T) -> (N, T, 1)
        att_w = nn.functional.softmax(self.W(x).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x

class TubeViT_v2_abl_fea(nn.Module):
    def __init__(
        self,
        video_shape: Union[List[int], np.ndarray],  # CTHW
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
    ):
        super(TubeViT_v2_abl_fea, self).__init__()
        self.video_shape = np.array(video_shape)  # CTHW
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.kernel_sizes = (
            (8, 8, 8),
            (16, 4, 4),
            (4, 12, 12),
        )

        self.strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
        )

        self.offsets = (
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        self.sparse_tubes_tokenizer = SparseTubesTokenizer(
            self.hidden_dim, self.kernel_sizes, self.strides, self.offsets
        )

        self.pos_embedding = self._generate_position_embedding()
        self.pos_embedding = torch.nn.Parameter(self.pos_embedding, requires_grad=False)
        self.register_parameter("pos_embedding", self.pos_embedding)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)
        self.register_parameter("class_token", self.class_token)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.attention_pooling = SelfAttentionPooling(self.hidden_dim)

        input_size=2048
        reduced_size=128
        hidden_size=32
        self.hidden_size = hidden_size
        self.ann = nn.Linear(input_size, reduced_size)
        self.ann2 = nn.Linear(self.hidden_dim, reduced_size)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, x, fea, fea_len):
        

        fea = self.ann(fea)  # (B, seq, 2048) -> (B, seq, 128)

        outputs, _ = self.rnn(fea, self._get_initial_state(fea.size(0), fea.device))    ### (B, seq_len, hidden_dim)  (1, B, hidden_dim)
        q = self.q(outputs)  # frame quality (B, seq_len, 1)
        # print(q.shape)
        score = torch.zeros(fea.shape[0], device=fea.device)  # (B, 1)
        for i in range(fea.shape[0]):  #
            qi = q[i, :int(fea_len[i].item()+x.shape[1])]  ### (seq_len_i, 1)
            score[i] = torch.mean(qi)

        return score
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        output = np.ceil((self.video_shape[[1, 2, 3]] - offset - kernel_size + 1) / stride).astype(int)
        return output

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        position_embedding = [torch.zeros(1, self.hidden_dim)]

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                offset=self.offsets[i],
            )
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous()
        return position_embedding

class TubeViT_v2_abl_resfea(TubeViT_v2_abl_fea):
    def __init__(
        self,
        video_shape: Union[List[int], np.ndarray],  # CTHW
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
    ):
        super(TubeViT_v2_abl_resfea, self).__init__(
            video_shape,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            num_classes
        )

    def forward(self, x, fea, fea_len):
        
        fea1 = fea[:, :-1]
        fea2 = fea[:, 1:]
        fea = fea1 - fea2
        fea = self.ann(fea)  # (B, seq, 2048) -> (B, seq, 128)

        outputs, _ = self.rnn(fea, self._get_initial_state(fea.size(0), fea.device))    ### (B, seq_len, hidden_dim)  (1, B, hidden_dim)
        q = self.q(outputs)  # frame quality (B, seq_len, 1)
        # print(q.shape)
        score = torch.zeros(fea.shape[0], device=fea.device)  # (B, 1)
        for i in range(fea.shape[0]):  #
            qi = q[i, :int(fea_len[i].item()+x.shape[1])]  ### (seq_len_i, 1)
            score[i] = torch.mean(qi)

        return score
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        output = np.ceil((self.video_shape[[1, 2, 3]] - offset - kernel_size + 1) / stride).astype(int)
        return output

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        position_embedding = [torch.zeros(1, self.hidden_dim)]

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                offset=self.offsets[i],
            )
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous()
        return position_embedding

class TubeViT_v2_abl_resfeav2(TubeViT_v2_abl_fea):
    def __init__(
        self,
        video_shape: Union[List[int], np.ndarray],  # CTHW
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
    ):
        super(TubeViT_v2_abl_resfeav2, self).__init__(
            video_shape,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            num_classes
        )

    def forward(self, x, fea, fea_len):
        
        fea1 = fea[:, :-1]
        fea2 = fea[:, 1:]
        fea = torch.cat([fea[:,::5], fea1 - fea2], dim=1)

        fea = self.ann(fea)  # (B, seq, 2048) -> (B, seq, 128)

        outputs, _ = self.rnn(fea, self._get_initial_state(fea.size(0), fea.device))    ### (B, seq_len, hidden_dim)  (1, B, hidden_dim)
        q = self.q(outputs)  # frame quality (B, seq_len, 1)
        # print(q.shape)
        score = torch.zeros(fea.shape[0], device=fea.device)  # (B, 1)
        for i in range(fea.shape[0]):  #
            qi = q[i, :int(fea_len[i].item()+x.shape[1])]  ### (seq_len_i, 1)
            score[i] = torch.mean(qi)

        return score
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        output = np.ceil((self.video_shape[[1, 2, 3]] - offset - kernel_size + 1) / stride).astype(int)
        return output

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        position_embedding = [torch.zeros(1, self.hidden_dim)]

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                offset=self.offsets[i],
            )
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous()
        return position_embedding

from .model import Tube_fps
class Tube_fps_abl_noqm(Tube_fps):
    def __init__(
        self,
        video_shape,  # CTHW
        num_layers= 1,
        num_heads= 16,
        hidden_dim = 768,
        context_dim = None,

        spa_fea_dim=2048, # mean 1024+ std 1024
        gru_hidden_size=32,
    ):
        super().__init__(video_shape,
        num_layers,
        num_heads,
        hidden_dim,
        context_dim,
        spa_fea_dim, 
        gru_hidden_size)
        
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, spa_feas, fea_len, *args, **kwargs):

        # x = self.tubes_embedding(x) # torch.Size([4, 343, 768])
        # print(x.shape)
        n, c, t, _, _ = x.shape

        # # Expand the class token to the full batch
        # batch_class_token = self.class_token.expand(n, -1, -1)
        # x = torch.cat([batch_class_token, x], dim=1) # torch.Size([4, 344, 768])

        # x = x + self.pos_embedding # 

        # # with torch.no_grad():

        # if self.context_dim is not None:
        #     fps = torch.tensor(fea_len.unsqueeze(1), dtype=x.dtype)
        #     fps = self.fps_embedding(fps.unsqueeze(1))

        # x = rearrange(x,'b t c -> b c t')
        # for block in self.blocks:
        #     if self.context_dim is not None:
        #         x = block(x, fps) 
        #     else: 
        #         x = block(x) 
        # x = rearrange(x,'b c t -> b t c')

        spa_fea = self.spa_proj_in(spa_feas)  # (B, seq, 2048) -> (B, seq, 768)
        # x = spa_fea + self.fea_fuse(self.fuse_norm(spa_fea), x) # 

        # x = torch.cat([x, spa_fea], dim=1)

        # outputs, _ = self.rnn(x, self._get_initial_state(x.size(0), x.device))    ### (B, seq_len, hidden_dim)  (1, B, hidden_dim)
        
        q = self.q_head(spa_fea)  # frame quality (B, seq_len, 1)
        
        score = torch.zeros(x.shape[0], device=x.device)  # (B, 1)
        for i in range(x.shape[0]):  #
            qi = q[i, :int(fea_len[i].item()+t+1)]  
            score[i] = torch.mean(qi)
        return score

from torchvision.models.video import r3d_18

from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat
def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        # print('q', q.shape)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # print('k', k.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print('qkv', q.shape, k.shape)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # print('sim', sim.shape)
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        # print('attn', attn.shape)
        out = einsum('b i j, b j d -> b i d', attn, v)
        # print(out.shape)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # print(out.shape)
        # print(self.to_out(out).shape)
        return self.to_out(out)

from torchvision.models.vision_transformer import MLPBlock
class crossEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_1_y = norm_layer(512)
        # self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.attention = CrossAttention(hidden_dim, 512, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input):
        # torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # y = input[1]
        # input = input[0]
        input, flow = input
        x = self.ln_1(input)
        flow = self.ln_1_y(flow)
        # x, _ = self.self_attention(query=flow, key=x, value=x, need_weights=False)
        x = self.attention(x, flow)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return (x + y, flow)
class TemporalEncoder(nn.Module):
    """
    Transformer Model Encoder for sequence to sequence translation.
    Code from torch.
    Move pos_embedding to TubeViT
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = crossEncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: Tensor, y):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        z, flow= self.layers((self.dropout(x),self.dropout(y)))
        return self.ln(z)
class TubeViT_v2_flowv4(nn.Module):
    def __init__(
        self,
        video_shape: Union[List[int], np.ndarray],  # CTHW
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
    ):
        super(TubeViT_v2_flowv4, self).__init__()

        self.video_shape = np.array(video_shape)  # CTHW
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.kernel_sizes = (
            (8, 8, 8),
            (16, 4, 4),
            (4, 12, 12),
            (1, 16, 16),
        )

        self.strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
        )

        self.offsets = (
            (0, 0, 0),
            (4, 8, 8),
            (0, 16, 16),
            (0, 0, 0),
        )
        self.sparse_tubes_tokenizer = SparseTubesTokenizer(
            self.hidden_dim, self.kernel_sizes, self.strides, self.offsets
        )

        self.pos_embedding = self._generate_position_embedding()
        self.pos_embedding = torch.nn.Parameter(self.pos_embedding, requires_grad=False)
        self.register_parameter("pos_embedding", self.pos_embedding)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)
        self.register_parameter("class_token", self.class_token)

        self.encoder = TemporalEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        input_size=2048
        reduced_size=128
        hidden_size=32
        self.hidden_size = hidden_size
        self.ann = nn.Linear(input_size, reduced_size)
        self.ann2 = nn.Linear(self.hidden_dim, reduced_size)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)

        r3d = r3d_18()
        flow_stem = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.res3d18 = nn.Sequential(
            flow_stem,
            r3d.layer1,
            r3d.layer2,
            r3d.layer3,
            r3d.layer4,
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        # self.ann3 = nn.Linear(512, 768)
        # self.attention = CrossAttention(768, 512)

    def forward(self, x, fea, fea_len, *args, **kwargs):
        flow = args[0]
        # print(flow.shape)
        flow_out = self.res3d18(flow)
        # print(flow_out.shape)
        flow_fea = flow_out.flatten(2).permute(0,2,1)
        # print(flow_fea.shape)  ### [1, 4, 768]

        x = self.sparse_tubes_tokenizer(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.pos_embedding

        # with torch.no_grad():
        x_a = self.encoder(x,flow_fea)  ### (B, patch+1, 768) (b, 540, 768)

        # Attention pooling
        # x_a = self.attention(x, flow_fea)  ### (B, 768)
        # print(x_a.shape) ### (B, 540, 768)
        fea_spatial = self.ann2(x_a)

        # x = self.heads(x)

        fea = self.ann(fea)  # (B, seq, 2048) -> (B, seq, 128)
        fea = torch.cat([fea_spatial, fea], dim=1)

        outputs, _ = self.rnn(fea, self._get_initial_state(fea.size(0), fea.device))    ### (B, seq_len, hidden_dim)  (1, B, hidden_dim)
        q = self.q(outputs)  # frame quality (B, seq_len, 1)
        # print(q.shape)
        score = torch.zeros(fea.shape[0], device=fea.device)  # (B, 1)
        for i in range(fea.shape[0]):  #
            # print(fea_len[i].item()+x.shape[1])
            # qi = q[i, :int(fea_len[i].item()+x.shape[1])]  ### (seq_len_i, 1)
            qi = q[i, :]  ### (seq_len_i, 1)
            # print(qi.shape,fea_spatial.shape,fea.shape)
            score[i] = torch.mean(qi)

        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        output = np.ceil((self.video_shape[[1, 2, 3]] - offset - kernel_size + 1) / stride).astype(int)
        return output

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        position_embedding = [torch.zeros(1, self.hidden_dim)]

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                offset=self.offsets[i],
            )
            position_embedding.append(pos_embed)
            print(pos_embed.shape)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous()
        return position_embedding
