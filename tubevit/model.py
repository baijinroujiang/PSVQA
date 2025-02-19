from functools import partial
from typing import Callable, List, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import EncoderBlock
from typing_extensions import OrderedDict

from tubevit.positional_encoding import get_3d_sincos_pos_embed

from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat

class Encoder(nn.Module):

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

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.zero-initializing the final convolutional layer in each block prior to any residual connections can accelerate training. 
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads # inner_dim == SpatialTransformer.model_channels
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

    def forward(self, x, context=None, mask=None):# x:(b,T,C), context:(b,seq_len,context_dim)
        h = self.heads

        q = self.to_q(x)# q:(b,T,inner_dim)
        context = default(context, x)
        k = self.to_k(context)# (b,seq_len,inner_dim)
        v = self.to_v(context)# (b,seq_len,inner_dim)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))# n is seq_len for k and v

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale # (b*head,T,seq_len)

        if exists(mask):# false
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)# (b*head,T,inner_dim/head)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)# (b,T,inner_dim)
        return self.to_out(out)
    
class Conv1dFeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.,kernel_size = 9):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Conv1d(dim, inner_dim,kernel_size=kernel_size,padding=kernel_size//2),
            nn.GELU()
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Conv1d(inner_dim, dim_out,kernel_size=kernel_size,padding=kernel_size//2)
        )

    def forward(self, x): # x shape (B,C,T)
        return self.net(x)

class TubeEmbeddingLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        kernel_sizes = ((8, 8, 8), (16, 4, 4), (4, 12, 12)),
        strides = ((16, 32, 32), (6, 32, 32), (16, 32, 32)),
        offsets = ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides 
        self.offsets = offsets

        self.sparse_tubes_tokenizer = SparseTubesTokenizer(
            self.hidden_dim, self.kernel_sizes, self.strides, self.offsets
        )

    def forward(self, x):
        x = self.sparse_tubes_tokenizer(x)
        return x
    
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, head_dim, dropout=0., context_dim=None, gated_ff=True, checkpoint=True): # 1 self 1 cross or 2 self
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=head_dim, dropout=dropout)  # is a self-attention,if context is none
        self.ff = Conv1dFeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=head_dim, dropout=dropout)  # use as cross attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):# x shape:(B,T,C)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x

        x = self.ff(self.norm3(x).permute(0,2,1)).permute(0,2,1) + x
        return x
    
class TemporalEncoder(nn.Module):
    def __init__(
            self, 
            in_channels, 
            n_heads, 
            head_dim,
            depth=1, 
            dropout=0., 
            context_dim=None,
            norm_num_groups=32,
            use_linear_projection=False
            ):
        
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * head_dim
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, head_dim, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x,'b c t -> b t c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x,'b t c -> b c t')
        
        x = self.proj_out(x)
        return x + x_in

class Tube_fps(nn.Module):
    def __init__(
        self,
        video_shape,  # CTHW
        num_layers= 1,
        num_heads= 16,
        hidden_dim = 768,
        context_dim = None,

        spa_fea_dim=2048, 
        gru_hidden_size=32,
    ):
        super().__init__()
        self.video_shape = np.array(video_shape)  # CTHW
        self.context_dim = context_dim

        self.tubes_embedding = TubeEmbeddingLayer(hidden_dim)

        self.pos_embedding = self._generate_position_embedding()
        self.pos_embedding = torch.nn.Parameter(self.pos_embedding, requires_grad=False)
        self.register_parameter("pos_embedding", self.pos_embedding)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim), requires_grad=True)
        self.register_parameter("class_token", self.class_token)

        if context_dim is not None:
            self.fps_embedding = nn.Conv1d(1, context_dim, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([
            TemporalEncoder(hidden_dim, 
                            num_heads, 
                            hidden_dim//num_heads,
                            depth=1,
                            context_dim=context_dim
                            ) for _ in range(num_layers)
        ])

        self.gru_hidden_size = gru_hidden_size
        self.spa_proj_in = nn.Linear(spa_fea_dim, hidden_dim) 
        self.rnn = nn.GRU(hidden_dim, gru_hidden_size, batch_first=True)

        self.q_head = nn.Linear(gru_hidden_size, 1)

    def forward(self, x, spa_feas, fea_len, *args, **kwargs):

        x = self.tubes_embedding(x)
        n, t, c = x.shape

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1) 

        x = x + self.pos_embedding 

        if self.context_dim is not None:
            fps = torch.tensor(fea_len.unsqueeze(1), dtype=x.dtype)
            fps = self.fps_embedding(fps.unsqueeze(1))

        x = rearrange(x,'b t c -> b c t')
        for block in self.blocks:
            if self.context_dim is not None:
                x = block(x, fps) 
            else: 
                x = block(x) 
        x = rearrange(x,'b c t -> b t c')

        spa_fea = self.spa_proj_in(spa_feas) 
        x = torch.cat([x, spa_fea], dim=1) 

        outputs, _ = self.rnn(x, self._get_initial_state(x.size(0), x.device))  
        
        q = self.q_head(outputs) 
        
        score = torch.zeros(x.shape[0], device=x.device) 
        for i in range(x.shape[0]):  #
            qi = q[i, :int(fea_len[i].item()+t+1)]  
            score[i] = torch.mean(qi)
        return score
    
    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        output = np.ceil((self.video_shape[[1, 2, 3]] - offset - kernel_size + 1) / stride).astype(int)
        return output

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        hidden_dim = self.tubes_embedding.hidden_dim
        kernel_sizes = self.tubes_embedding.kernel_sizes
        strides = self.tubes_embedding.strides
        offsets = self.tubes_embedding.offsets

        position_embedding = [torch.zeros(1, hidden_dim)] # 1: class_token
        for i in range(len(kernel_sizes)):
            tube_shape = self._calc_conv_shape(kernel_sizes[i], strides[i], offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=hidden_dim,
                tube_shape=tube_shape,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                offset=offsets[i],
            )
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous()
        return position_embedding
    
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.gru_hidden_size, device=device)
        return h0

class Tube_fps_cross_fuse(Tube_fps):
    def __init__(
        self,
        video_shape,  # CTHW
        num_layers= 1,
        num_heads= 16,
        hidden_dim = 768,
        context_dim = None,

        spa_fea_dim=2048,
        gru_hidden_size=32,
    ):
        super().__init__(video_shape,
        num_layers,
        num_heads,
        hidden_dim,
        context_dim,
        spa_fea_dim, 
        gru_hidden_size)
        
        self.fea_fuse = Conv1dFeedForward(hidden_dim, hidden_dim, mult=1, kernel_size=3)

    def forward(self, x, spa_feas, fea_len, *args, **kwargs):

        x = self.tubes_embedding(x) 
        n, t, c = x.shape

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.pos_embedding # 

        if self.context_dim is not None:
            fps = torch.tensor(fea_len.unsqueeze(1), dtype=x.dtype)
            fps = self.fps_embedding(fps.unsqueeze(1))

        x = rearrange(x,'b t c -> b c t')
        for block in self.blocks:
            if self.context_dim is not None:
                x = block(x, fps) 
            else: 
                x = block(x) 
        x = rearrange(x,'b c t -> b t c')

        spa_fea = self.spa_proj_in(spa_feas)

        x = torch.cat([x, spa_fea], dim=1)
        x = rearrange(x,'b t c -> b c t')
        x = x + self.fea_fuse(x)
        x = rearrange(x,'b c t -> b t c')

        outputs, _ = self.rnn(x, self._get_initial_state(x.size(0), x.device)) 
        
        q = self.q_head(outputs) 
        
        score = torch.zeros(x.shape[0], device=x.device) 
        for i in range(x.shape[0]): 
            qi = q[i, :int(fea_len[i].item()+t+1)]  
            score[i] = torch.mean(qi)
        return score
