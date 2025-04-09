import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import SwinModel
import copy

class longitudinal_SwinT_B(nn.Module):
    def __init__(self, args):
        super(longitudinal_SwinT_B, self).__init__()
        model = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')
        modules1 = list(model.children())
        modules2 = list(modules1[1].children())
        modules2 = list(modules2[0].children())
        modules3 = list(modules2[0].children())
        modules4 = list(modules2[1].children())
        modules5 = list(modules2[2].children())
        modules6 = list(modules2[3].children())

        self.stage = []
        self.context_stage = []
        self.downsample = []
        self.context_downsample = []

        self.embeding = modules1[0]
        self.stage.append(nn.Sequential(*(list(modules3[0].children()))))
        self.downsample.append(modules3[1])
        self.stage.append(nn.Sequential(*(list(modules4[0].children()))))
        self.downsample.append(modules4[1])
        self.stage.append(nn.Sequential(*(list(modules5[0].children()))))
        self.downsample.append(modules5[1])
        self.stage.append(nn.Sequential(*(list(modules6[0].children()))))

        self.feature_num = [128, 256, 512, 1024]
        self.num_heads = [4, 8, 16, 32]
        self.context_fusion = []
        self.context_gate = []
        self.fusion = []
        self.gate = []

        for i in range(4):
            if i<3:
                self.context_downsample.append(copy.deepcopy(self.downsample[i]))
            self.context_stage.append(copy.deepcopy(self.stage[i]))
            self.context_fusion.append(PWAM(
                self.feature_num[i], self.feature_num[i], 768, self.feature_num[i], self.feature_num[i], self.num_heads[i]))
            self.context_gate.append(nn.Sequential(
                nn.Linear(self.feature_num[i], self.feature_num[i], bias=False),
                nn.ReLU(),
                nn.Linear(self.feature_num[i], self.feature_num[i], bias=False),
                nn.Tanh()
            ))
            self.fusion.append(FeatureFusion(self.feature_num[i]))
            self.gate.append(nn.Sequential(
                nn.Linear(self.feature_num[i], self.feature_num[i], bias=False),
                nn.ReLU(),
                nn.Linear(self.feature_num[i], self.feature_num[i], bias=False),
                nn.Tanh()
            ))

        self.stage = nn.ModuleList(self.stage)
        self.context_stage = nn.ModuleList(self.context_stage)
        self.downsample = nn.ModuleList(self.downsample)
        self.context_downsample = nn.ModuleList(self.context_downsample)
        self.context_fusion = nn.ModuleList(self.context_fusion)
        self.context_gate = nn.ModuleList(self.context_gate)
        self.fusion = nn.ModuleList(self.fusion)
        self.gate = nn.ModuleList(self.gate)

        self.layernorm1 = modules1[2]
        self.layernorm2 = copy.deepcopy(self.layernorm1)
        self.pooler1 = modules1[3]
        self.pooler2 = copy.deepcopy(self.pooler1)

    def forward(self, x1, x2, ctxt, context_attmasks, has_progress):
        '''
            current_img = (B, 3, 224, 224)
            context_img = (B, 3, 224, 224)
            context = (B, N_l, 768)
            context_attmask = (B, N_l)
        '''
        ctxt = ctxt.permute(0, 2, 1)
        context_attmasks = context_attmasks.unsqueeze(dim=-1)

        x1, x1_indim = self.embeding(x1)
        x2, x2_indim = self.embeding(x2)

        for i in range(4):
            h1, w1 = x1_indim
            if i>0:
                x1 = self.downsample[i-1](x1, x1_indim)
                x2 = self.context_downsample[i-1](x2, x2_indim)
                x1_indim = ((h1 + 1) // 2, (w1 + 1) // 2)
                x2_indim = ((h1 + 1) // 2, (w1 + 1) // 2)
            for j, layer_module in enumerate(self.stage[i]):
                layer_outputs = layer_module(x1, x1_indim)
                x1 = layer_outputs[0]
            for j, layer_module in enumerate(self.context_stage[i]):
                layer_outputs = layer_module(x2, x2_indim)
                x2 = layer_outputs[0]

            x2_residual = self.context_fusion[i](x2, ctxt, context_attmasks)
            alpha1 = self.context_gate[i](x2_residual) * has_progress.unsqueeze(-1).unsqueeze(-1)
            x2 = x2 + (alpha1 * x2_residual)

            x1_residual = self.fusion[i](x1.permute(0, 2, 1), x2.permute(0, 2, 1)).permute(0, 2, 1)
            alpha2 = self.gate[i](x1_residual) * has_progress.unsqueeze(-1).unsqueeze(-1)
            x1 = x1 + (alpha2 * x1_residual)

        x1 = self.layernorm1(x1)
        x2 = self.layernorm2(x2)
        
        avg_feats = self.pooler1(x1.transpose(1, 2)).flatten(1)
        avg_feat2 = self.pooler2(x2.transpose(1, 2)).flatten(1)

        return x1, avg_feats, avg_feat2


class PWAM(nn.Module):
    '''
    * LAVT: Language-Aware Vision Transformer for Referring Image Segmentation
    * Yang, Zhao and Wang, Jiaqi and Tang, Yansong and Chen, Kai and Zhao, Hengshuang and Torr, Philip HS
    * 2022
    * https://github.com/yz93/LAVT-RIS
    '''
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1), 
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )
        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels, 
                                                            l_in_channels,
                                                            key_channels,
                                                            value_channels, 
                                                            out_channels=value_channels,
                                                            num_heads=num_heads)
        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        vis = self.vis_project(x.permute(0, 2, 1))
        lang = self.image_lang_att(x, l, l_mask)
        lang = lang.permute(0, 2, 1)
        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)
        mm = mm.permute(0, 2, 1)
        return mm
    
class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1) 
        l_mask = l_mask.permute(0, 2, 1) 

        query = self.f_query(x)
        query = query.permute(0, 2, 1)
        key = self.f_key(l)
        value = self.f_value(l)
        key = key * l_mask 
        value = value * l_mask 
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map 

        sim_map = sim_map + (1e4*l_mask - 1e4)
        sim_map = F.softmax(sim_map, dim=-1)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)
        out = out.permute(0, 2, 1)
        out = self.W(out)
        out = out.permute(0, 2, 1)

        return out

class FeatureFusion(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super(FeatureFusion, self).__init__()
        self.LDC1 = nn.Sequential(
            conv3x3_learn(input_dim, input_dim),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            )
        self.LDC2 = nn.Sequential(
            conv3x3_learn(input_dim, input_dim),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            )
        self.map = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid(),
            )
        self.fc = nn.Linear(input_dim * 2, input_dim)
    
    def forward(self, x1, x2):
        b, c, hw = x1.size()

        x1_enh = self.LDC1(x1.reshape(b, c, int(hw**0.5), int(hw**0.5)))
        x2_enh = self.LDC2(x2.reshape(b, c, int(hw**0.5), int(hw**0.5)))
        x_diff = self.map((x1_enh - x2_enh).reshape(b, c, int(hw**0.5), int(hw**0.5)))

        x1_fuse = torch.mul(x1_enh, x_diff)
        x2_fuse = torch.mul(x2_enh, x_diff)

        x_concat = torch.cat((x1_fuse, x2_fuse), dim=1).reshape(b, 2*c, hw)  # (b, 2*c, H*W)

        x_concat = x_concat.permute(0, 2, 1)
        x_concat = self.fc(x_concat)
        x_concat = x_concat.permute(0, 2, 1)

        return x_concat
    

class conv3x3_learn(nn.Module):
    '''
    * Learnable Descriptive Convolutional Network for Face Anti-Spoofing
    * Huang, Pei-Kai and Ni, Hui-Yu and Ni, Yan-Qin and Hsu, Chiou-Ting
    * 2022
    * https://github.com/huiyu8794/LDCNet
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(conv3x3_learn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        
        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
 
    def forward(self, x):
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]
 
        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff