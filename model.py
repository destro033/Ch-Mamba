"""
Glossary:
    b: batch size                       
    l: sequence length                 
    d or d_model: hidden dim
    n or d_state: latent state dim      
    expand: expansion factor            
    d_in or d_inner: d * expand         
    A, B, C, D: state space parameters  
                                        
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                 

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

class ModelArgs:
    def __init__(self, d_model=128, n_layer=4, seq_len=96, d_state=16, expand=2, dt_rank='auto',
                 d_conv=4, pad_multiple=8, conv_bias=True, bias=False,
                 num_channels=24, patch_len=16, stride=8, forecast_len=96, sigma=0.5, reduction_ratio=8, verbose=False):
        self.d_model = d_model
        self.n_layer = n_layer
        self.seq_len = seq_len
        self.d_state = d_state
        self.v = verbose
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.pad_multiple = pad_multiple
        self.conv_bias = conv_bias
        self.bias = bias
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        self.forecast_len = forecast_len
        self.sigma = sigma
        self.reduction_ratio = reduction_ratio
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.forecast_len % self.pad_multiple != 0:
            self.forecast_len += (self.pad_multiple - self.forecast_len % self.pad_multiple)

class ChannelMixup(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            B, V, L = x.shape
            perm = torch.randperm(V)

            #print("Mixup perm: ", perm) 

            lambda_ = torch.normal(mean=0, std=self.sigma, size=(V,)).to(x.device)
            #print("Mixup lambda:", lambda_.shape) #(num_channels)

            x_mixed = x + lambda_.unsqueeze(1) * x[:, perm]
            return x_mixed
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool_out = self.avg_pool(x)
        #print("avg_pool output shape:", avg_pool_out.shape) #(B, d_model, 1)

        avg_pool_out = avg_pool_out.squeeze(-1)
        #print("avg_pool squeezed shape:", avg_pool_out.shape) #(B, d_model)

        avg_fc1 = self.fc1(avg_pool_out)
        #print("avg fc1 output shape:", avg_fc1.shape) #(B, d_model/2)

        avg_relu = self.relu(avg_fc1)
        #print("avg relu output shape:", avg_relu.shape) #(B, d_model/2)

        avg_out = self.fc2(avg_relu)
        #print("avg fc2 output shape:", avg_out.shape) #(B, d_model)

        max_pool_out = self.max_pool(x)
        #print("max_pool output shape:", max_pool_out.shape) #(B, d_model, 1)

        max_pool_out = max_pool_out.squeeze(-1)
        #print("max_pool squeezed shape:", max_pool_out.shape) #(B, d_model)

        max_fc1 = self.fc1(max_pool_out)
        #print("max fc1 output shape:", max_fc1.shape) #(B, d_model/2)

        max_relu = self.relu(max_fc1)
        #print("max relu output shape:", max_relu.shape) #(B, d_model/2)

        max_out = self.fc2(max_relu)
        #print("max fc2 output shape:", max_out.shape) #(B, d_model)

        out = avg_out + max_out
        #print("sum shape:", out.shape) #(B, d_model)

        out = self.sigmoid(out)
        #print("sigmoid output shape:", out.shape) #(B, d_model)

        out = out.unsqueeze(-1)
        #print("final output shape:", out.shape) #(B, d_model, 1)

        return out

class PatchMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([MambaBlock(args) for _ in range(args.n_layer)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

class CMambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.patch_mamba = PatchMamba(args)
        self.channel_attention = ChannelAttention(args.d_model, args.reduction_ratio)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        x = self.patch_mamba(x)
        attn = self.channel_attention(x.permute(0, 2, 1))
        x = x * attn.permute(0, 2, 1)
        return self.norm(x)

class CMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.channel_mixup = ChannelMixup(args.sigma)
        self.patch_embedding = nn.Linear(args.patch_len * args.num_channels, args.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, args.num_patches, args.d_model))
        
        self.c_mamba_blocks = nn.ModuleList([CMambaBlock(args) for _ in range(args.n_layer)])
        
        self.norm_f = RMSNorm(args.d_model)
        self.output_layer = nn.Linear(args.d_model * args.num_patches, args.num_channels * args.forecast_len)

    def forward(self, input_ids):

        print("input_ids", input_ids.shape) if self.args.v else None  #(B,V,L)

        x = self.channel_mixup(input_ids)
        print("after channel mixup", x.shape) if self.args.v else None  #(B,V,L)

        # Patching
        B, V, L = x.shape
        P = self.args.patch_len
        S = self.args.stride

        print("Patch len", P) if self.args.v else None
        print("stride", S) if self.args.v else None

        # Manual patching
        patches = []
        for i in range(0, L - P + 1, S):
            patch = x[:, :, i:i+P].reshape(B, -1)
            patches.append(patch)
        num_patches = (L - P) // S + 1
        print(f"Calculated number of patches: {num_patches}") if self.args.v else None

        x = torch.stack(patches, dim=1)  # (B, num_patches, V*P)
        print("after patching", x.shape) if self.args.v else None

        # Patch embedding
        x = self.patch_embedding(x)  # (B, num_patches, d_model)
        print("after patch embedding", x.shape) if self.args.v else None

        # Adjust positional encoding
        pos_encoding = self.pos_encoding[:, :x.size(1), :]
        print(f"Positional encoding shape: {pos_encoding.shape}") if self.args.v else None #(1, num_patches, d_model)

        # Add positional encoding
        x = x + pos_encoding
        print("after positional encoding", x.shape) if self.args.v else None  #(B, num_patches, d_model)

        # Apply CH-Mamba blocks
        for block in self.c_mamba_blocks:
            x = block(x)
        print("after CH-Mamba blocks", x.shape) if self.args.v else None #(B, num_patches, d_model)

        x = self.norm_f(x)
        print("after norm_f", x.shape) if self.args.v else None  #(B, num_patches, d_model)

        # Output layer
        x = x.reshape(x.shape[0], -1)
        print("before output layer", x.shape) if self.args.v else None  #(B, num_patches*d_model)

        logits = self.output_layer(x)
        print("after output layer", logits.shape) if self.args.v else None #(B, num_channels*forecast_length )

        logits = logits.reshape(-1, self.args.num_channels, self.args.forecast_len)
        print("final logits", logits.shape) if self.args.v else None  #(b, num_channels, forecast_length)

        return logits

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        #print("Mamba block input shape: ", x.shape)        #(B, num_patches, d_model)
        
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        #print("Mamba block split shape x: ", x.shape)      #(B, num_patches, d_inner)
        #print("Mamba block split shape res: ", res.shape)  #(B, num_patches, d_inner)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)
        #print("Mamba block output shape: ", output.shape)

        return output

    def ssm(self, x):
        """

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

    
        """
        (d_in, n) = self.A_log.shape
        #print("SSM A_log shape: ", self.A_log.shape)  #(d_inner, d_state)

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent 
        #     ∆, B, C are input-dependent 
        
        A = -torch.exp(self.A_log.float())  #(d_inner, d_state)
        D = self.D.float()

        #print("SSM A shape: ", A.shape)  #(d_inner, d_state)
        #print("SSM D shape: ", D.shape)  #(d_inner) 

        x_dbl = self.x_proj(x)   
        #print("SSM x_dbl shape: ", x_dbl.shape)   #(B, num_patches, dt_rank + 2*(d_state))
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, num_patches, dt_rank). B, C: (b, num_patches, d_state)
        #print("SSM delta shape: ", delta.shape)  #(B, num_patches, dt_rank)
        #print("SSM B shape: ", B.shape)   #(B, num_patches, d_state)
        #print("SSM C shape: ", C.shape)   #(B, num_patches, d_state)

        delta = F.softplus(self.dt_proj(delta)) 
        #print("SSM delta shape after softplus: ", delta.shape)  #(B, num_patches, d_inner)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        #print("SSM output shape: ", y.shape)  #(b, num_patches, d_inner)

        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        #print("Selective scan input shape: ", u.shape)  #(B, num_patches, d_inner)

        n = A.shape[1]
        #print("Selective scan A shape: ", A.shape)  #(d_inner, d_state)
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization.
        # - B is discretized using a simplified Euler discretization instead of ZOH. 
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        #print("Selective scan deltaA shape: ", deltaA.shape) #(B, num_patches, d_inner, d_state)
        #print("Selective scan deltaB_u shape: ", deltaB_u.shape) #(B, num_patches, d_inner, d_state)
        
        
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        #print("Selective scan x shape: ", x.shape)  #(B, d_inner, d_state)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  
        #print("Selective scan output shape: ", y.shape) #(b, num_patches, d_inner)
        
        y = y + u * D
        #print("Selective scan output shape after D: ", y.shape) #(b, num_patches, d_inner)
    
        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
