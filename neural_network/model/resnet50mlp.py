from collections import OrderedDict
from functools import partial
import torchvision.models as models
import torch
import torch.nn as nn
from timm.layers import DropPath
from torchvision.models import resnet50, ResNet50_Weights,resnet101,ResNet101_Weights,resnet18,ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
import timm




class SeedWeightvitalityPredictor(nn.Module):
    def __init__(self,weight_num=3,vitality_num=2,imageFeatureNumber = 2048,WeightFeatureNumber = 128):
        super(SeedWeightvitalityPredictor,self).__init__()

        imag_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.imageEncoder = nn.Sequential(*list(imag_model.children())[:-1])
        self.imageFeatureNumber = imageFeatureNumber

        self.WeightEncoder = nn.Sequential(
            nn.Linear(1,WeightFeatureNumber),
            nn.ReLU(),
            nn.Linear(WeightFeatureNumber,WeightFeatureNumber),
            nn.ReLU()
        )
        self.WeightFeatureNumber = WeightFeatureNumber
        self.WeightRegression = nn.Sequential(
            nn.Linear(self.imageFeatureNumber+self.WeightFeatureNumber,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,weight_num)
        )

    def forward(self,OriginalImage,OriginalWeight):
        ImageFeature = self.imageEncoder(OriginalImage)
        ImageFeature = ImageFeature.view(ImageFeature.size(0),-1)
        WeightFeature = self.WeightEncoder(OriginalWeight)

        CombinationFeature = torch.cat([ImageFeature,WeightFeature],dim = 1)
        WeightPredict = self.WeightRegression(CombinationFeature)

        return WeightPredict
class SeedResnet101(nn.Module):
    def __init__(self,vitality_num=2,imageFeatureNumber = 2048,WeightFeatureNumber = 128):
        super(SeedResnet101,self).__init__()

        imag_model = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.imageEncoder = nn.Sequential(*list(imag_model.children())[:-1],
                                          nn.Linear(2048, 2)
                                          )

    def forward(self,OriginalImage):
        image_feature = self.imageEncoder(OriginalImage)






def build_encoder():
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    return encoder

class LSTMWeightPredictor(nn.Module):
    def __init__(self, ImgFeatureDim=2048, WeightDim=1, HiddenDim=1000, NumLayers=1, FeatureNum=10):
        super(LSTMWeightPredictor, self).__init__()

        self.ImageEncoder1 = build_encoder()
        self.ImgProj1 = nn.Sequential(
            nn.Linear(ImgFeatureDim, 512),
            nn.ReLU(),
            nn.Linear(512, FeatureNum)
        )
        self.ImageEncoder2 = build_encoder()
        self.ImgProj2 = nn.Sequential(
            nn.Linear(ImgFeatureDim, 512),
            nn.ReLU(),
            nn.Linear(512, FeatureNum)
        )
        self.ImageEncoder3 = build_encoder()
        self.ImgProj3 = nn.Sequential(
            nn.Linear(ImgFeatureDim, 512),
            nn.ReLU(),
            nn.Linear(512, FeatureNum)
        )

        self.WeightEncoder = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU()
        )

        self.Mlp = nn.Sequential(
            nn.Linear(FeatureNum + WeightDim + 10, FeatureNum + WeightDim),
            nn.ReLU(),
            nn.Linear(FeatureNum + WeightDim, FeatureNum)
        )

        self.PosEmbed = nn.Parameter(torch.randn(1, 3, FeatureNum))

        self.lstm = nn.LSTM(
            input_size=FeatureNum,
            hidden_size=HiddenDim,
            num_layers=NumLayers,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(HiddenDim, 1)

    def forward(self, Image, WeightSeq):
        B = Image.size(0)

        ImgFeat1 = self.ImageEncoder1(Image).view(B, -1)
        ImgProj1 = self.ImgProj1(ImgFeat1)

        ImgFeat2 = self.ImageEncoder2(Image).view(B, -1)
        ImgProj2 = self.ImgProj2(ImgFeat2)

        ImgFeat3 = self.ImageEncoder3(Image).view(B, -1)
        ImgProj3 = self.ImgProj3(ImgFeat3)

        ImgFeat = torch.stack([ImgProj1, ImgProj2, ImgProj3], dim=1)

        ImgFeat = ImgFeat + self.PosEmbed

        W1 = self.WeightEncoder(WeightSeq[:, 0].unsqueeze(1))
        W2 = self.WeightEncoder(WeightSeq[:, 1].unsqueeze(1))
        W3 = self.WeightEncoder(WeightSeq[:, 2].unsqueeze(1))
        WEncoded = torch.stack([W1, W2, W3], dim=1)

        WeightSeq = WeightSeq.unsqueeze(2)  # [B, 3, 1]

        MlpInput = torch.cat([ImgFeat, WEncoded, WeightSeq], dim=2)


        MlpOut = self.Mlp(MlpInput)


        out, _ = self.lstm(MlpOut)
        out = self.dropout(out)
        pred = self.fc(out[:, -1, :])
        return pred


class AblationExperiment(nn.Module):
    def __init__(self, ImgFeatureDim=2048, WeightDim=1, HiddenDim=1000, NumLayers=1,FeatureNum=10):
        super(AblationExperiment, self).__init__()

        self.ImageEncoder1 = build_encoder()
        self.ImgProj1 = nn.Sequential(
            nn.Linear(ImgFeatureDim, 512),
            nn.ReLU(),
            nn.Linear(512, FeatureNum)
        )
        self.ImageEncoder2 = build_encoder()
        self.ImgProj2 = nn.Sequential(
            nn.Linear(ImgFeatureDim, 512),
            nn.ReLU(),
            nn.Linear(512, FeatureNum)
        )
        self.ImageEncoder3 = build_encoder()
        self.ImgProj3 = nn.Sequential(
            nn.Linear(ImgFeatureDim, 512),
            nn.ReLU(),
            nn.Linear(512, FeatureNum)
        )
        self.PosEmbed = nn.Parameter(torch.randn(1, 4, FeatureNum))

        self.lstm = nn.LSTM(input_size=WeightDim,
                            hidden_size=HiddenDim,
                            num_layers=NumLayers,
                            batch_first=True)

        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(HiddenDim, 1)

    def forward(self, WeightSeq):

        LstmInput = WeightSeq.unsqueeze(2)

        # LSTM
        out, _ = self.lstm(LstmInput)
        pred = self.fc(out[:, -1, :])
        return pred

class SeedVitalityClassifier(nn.Module):
    def __init__(self, backbone_name='resnest101e', num_classes=1):
        super(SeedVitalityClassifier, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return self.sigmoid(out)

class PatchEmbed(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_chanel=3,embed_dim=768,norm_layer=None):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0],img_size[1]//patch_size[1]) #224/16=14 14*14
        self.num_patches = self.grid_size[0]*self.grid_size[1] #14*14=196
        self.proj = nn.Conv2d(in_chanel,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        B,C,H,W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1],f'input image size is {H}*{W},but expect image size is {self.img_size[0]}*{self.img_size[1]}'
        #B,3,224,224->B,768,14,14->B,768,196->B,196,768
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bais=False,qk_scale=None,atte_drop_ration=0.,proj_drop_ration=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = qk_scale or head_dim**(-0.5)
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bais)
        self.att_drop = nn.Dropout(atte_drop_ration)
        self.proj_drop = nn.Dropout(proj_drop_ration)
        self.proj = nn.Linear(dim,dim)

    def forward(self,x):
        B,N,C = x.shape
        #B,N,C->B,N,3,num_heads,C//num_heads->3,B,num_heads,N,C//self.num_heads
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        #shape:B,num_heads,N,C//num_heads
        q,k,v = qkv[0],qkv[1],qkv[2]
        #B,num_heads,N,C//num_heads->B,num_heads,C//num_heads,N->q*k = B,num_heads,N,N
        attn = (q @ k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=-1)
        #atten*v=B,num_heads,N,C//num_heads->B,N,num_heads,C//num_heads->B,N,C
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()  # 修复：正确实例化激活函数
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,dim,num_heads,qkv_bais=False,qk_scale=None,drop_ratio=0.,attn_drop_ratio=0.,drop_path_ratio=0.,act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,mlp_ratio=4):
        super(Block,self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,num_heads=num_heads,qkv_bais=qkv_bais,qk_scale=qk_scale,atte_drop_ration=attn_drop_ratio,proj_drop_ration=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim=int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_ratio)

    def forward(self,x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chanel=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chanel, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=None, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop_ratio, bias=qkv_bias)
        self.drop_path = nn.Dropout(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            act_layer(),  # 正确实例化激活函数
            nn.Dropout(drop_ratio),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(drop_ratio)
        )

    def forward(self, x):
        # 正确的 MultiheadAttention 使用
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chanel=15, num_classes=1, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=None, qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super(VisionTransformer, self).__init__()

        self.channel_reducer = nn.Conv2d(15, 3, kernel_size=1)

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_token = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chanel=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_token, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i], act_layer=act_layer, norm_layer=norm_layer, mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.channel_reducer(x)
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = x
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



class ResNet101_18Channel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.2):
        super().__init__()
        # 降维卷积，把18通道压缩到3通道
        self.channel_reducer = nn.Conv2d(15, 3, kernel_size=1)

        # 加载ResNet101
        if pretrained:
            weights = ResNet101_Weights.IMAGENET1K_V2
            self.backbone = resnet101(weights=weights)
        else:
            self.backbone = resnet101(weights=None)

        # 修改全连接层，加入Dropout
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.channel_reducer(x)
        return self.backbone(x)

def build_resnet101_18channel(num_classes=2, pretrained=False, dropout_p=0.2):
    return ResNet101_18Channel(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout_p)

class Resnet50_18channel(nn.Module):
    def __init__(self,num_classes=2,pretrained=True,dropout_p=0.2):
        super().__init__()

        self.channel_reducer = nn.Conv2d(15,3,kernel_size=1)

        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features,num_classes)
        )

    def forward(self,x):
        x = self.channel_reducer(x)
        return self.backbone(x)

def build_resnet50_18channel(num_classes=2,pretrained=True,dropout_p=0.2):
    return Resnet50_18channel(num_classes=num_classes,pretrained=pretrained,dropout_p=dropout_p)

class Resnet18_15channel(nn.Module):
    def __init__(self,num_classes=2,pretrained=True,dropout_p=0.2):
        super().__init__()

        self.channel_reducer = nn.Conv2d(15,3,kernel_size=1)

        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V2
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet50(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features,num_classes)
        )

    def forward(self,x):
        x = self.channel_reducer(x)
        return self.backbone(x)

def build_resnet18_15channel(num_classes=2,pretrained=True,dropout_p=0.2):
    return Resnet18_15channel(num_classes=num_classes,pretrained=pretrained,dropout_p=dropout_p)


class EfficientNetB0_15channel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.2):
        super().__init__()

        self.channel_reducer = nn.Conv2d(15, 3, kernel_size=1)

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)

        # 获取分类器的输入特征数
        in_features = self.backbone.classifier[1].in_features

        # 替换分类器
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.channel_reducer(x)
        return self.backbone(x)


def build_efficientnetb0_15channel(num_classes=2, pretrained=True, dropout_p=0.2):
    return EfficientNetB0_15channel(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout_p)


class InceptionV3_15channel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.2, aux_logits=False):
        super().__init__()

        # 将15通道转换为3通道
        self.channel_reducer = nn.Conv2d(15, 3, kernel_size=1)

        # 尺寸调整：224x224 -> 299x299
        self.resize = transforms.Resize((299, 299), antialias=True)

        if pretrained:
            weights = Inception_V3_Weights.IMAGENET1K_V1
            self.backbone = inception_v3(weights=weights, aux_logits=aux_logits)
        else:
            self.backbone = inception_v3(weights=None, aux_logits=aux_logits)

        # Inception v3的输入特征数
        in_features = self.backbone.fc.in_features

        # 替换分类器
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

        if aux_logits:
            in_features_aux = self.backbone.AuxLogits.conv.in_channels
            self.backbone.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    def forward(self, x):
        x = self.channel_reducer(x)
        x = self.resize(x)
        return self.backbone(x)


def build_inceptionv3_15channel(num_classes=2, pretrained=True, dropout_p=0.2, aux_logits=False):
    return InceptionV3_15channel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
        aux_logits=aux_logits
    )


from torchvision.models import vit_b_16, ViT_B_16_Weights
class ViT_15channel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.2):
        super().__init__()

        self.channel_reducer = nn.Conv2d(15, 3, kernel_size=1)

        self.resize = transforms.Resize((224, 224), antialias=True)

        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None)

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.channel_reducer(x)
        x = self.resize(x)
        return self.backbone(x)


def build_vit_15channel(num_classes=2, pretrained=True, dropout_p=0.2):

    return ViT_15channel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p
    )
