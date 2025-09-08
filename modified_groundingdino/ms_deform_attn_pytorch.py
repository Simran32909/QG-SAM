import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    """
    Pure PyTorch implementation of Multi-Scale Deformable Attention.
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_ = sampling_locations.shape
    
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, M_, H_*W_, D_ -> N_*M_, H_*W_, D_
        value_l_ = value_list[lid_].permute(0, 2, 1, 3).reshape(N_*M_, H_*W_, D_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].permute(0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, H_, W_ (value_l_.view)
        value_l_ = value_l_.reshape(N_*M_, D_, H_, W_)
        # N_*M_, Lq_*P_, 2 (sampling_grid_l_.reshape)
        sampling_grid_l_ = sampling_grid_l_.reshape(N_*M_, Lq_*P_, 2)
        
        # bilinear interpolation
        # N_*M_, D_, Lq_*P_
        sampling_value_l_ = F.grid_sample(
            value_l_, 
            sampling_grid_l_,
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )
        # N_, M_, D_, Lq_, P_
        sampling_value_l_ = sampling_value_l_.reshape(N_, M_, D_, Lq_, P_)
        # N_, M_, Lq_, P_, D_
        sampling_value_l_ = sampling_value_l_.permute(0, 1, 3, 4, 2)
        sampling_value_list.append(sampling_value_l_)
    
    # N_, M_, Lq_, L_, P_, D_
    attention_weights = attention_weights.view(N_, M_, Lq_, L_, P_, 1)
    output = (torch.stack(sampling_value_list, dim=3) * attention_weights).sum(dim=[3, 4])
    # N_, Lq_, M_, D_
    output = output.permute(0, 2, 1, 3)
    
    return output.contiguous()

class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, we'll just return None for all gradients
        # In a real implementation, you would compute proper gradients
        return None, None, None, None, None, None
