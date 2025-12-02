import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from functools import reduce
from operator import mul
import tqdm

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if cfg.TRAINER.BMIPLOSS.CONNECT_METHOD == "joint":
        BMIP_length = cfg.TRAINER.BMIPLOSS.N_CTX * 2
        connect_method = "joint"
    else:
        BMIP_length = cfg.TRAINER.BMIPLOSS.N_CTX
        connect_method = "other"
    # print(f"weight_threshold: {cfg.TRAINER.BMIPLOSS.WEIGHT_THRESHOLD}")
    # weight_threshold = cfg.TRAINER.BMIPLOSS.WEIGHT_THRESHOLD
    design_details = {"trainer": 'BMIP',
                      "vision_depth": 0,
                      "language_depth": 0,
                      "vision_ctx": 0,
                      "language_ctx": 0,
                      "bitp_length": BMIP_length,
                      "connect_method": connect_method}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BMIPLOSS.N_CTX
        ctx_init = cfg.TRAINER.BMIPLOSS.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_v_dim = clip_model.visual.class_embedding.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        val_v = math.sqrt(6. / float(3 * reduce(mul, [16], 1) + ctx_v_dim))  # noqa
        val_t = math.sqrt(6. / float(3 * reduce(mul, [16], 1) + ctx_dim))  # noqa
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.BMIPLOSS.PROMPT_DEPTH >= 1, "For BMIP, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.BMIPLOSS.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # if cfg.TRAINER.BMIPLOSS.CONNECT_METHOD == "joint":
        #     tmp_ctx = cfg.TRAINER.BMIPLOSS.N_CTX * 2
        # else:
        #     tmp_ctx = cfg.TRAINER.BMIPLOSS.N_CTX
        if cfg.TRAINER.BMIPLOSS.CONNECT_METHOD != "joint":
            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
        else:
            if ctx_init and (n_ctx * 2) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * (n_ctx * 2))
        print('BMIP design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of BMIP context words (tokens): {n_ctx}")
        if cfg.TRAINER.BMIPLOSS.CONNECT_METHOD == "joint":
            print(f"Total ctx number is : {n_ctx * 2}")

        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj_t = nn.Linear(ctx_dim, ctx_v_dim)
        self.proj_v = nn.Linear(ctx_v_dim, ctx_dim)
        self.proj_t.half()
        self.proj_v.half()
        self.ctx_t = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # (n_cls, n_tkn, dim)

        # Visual prompts are 768, as they are projected from 512
        self.ctx_v = nn.Parameter(torch.empty(embedding.shape[1], ctx_v_dim))
        nn.init.normal_(self.ctx_v, std=0.02)
        self.ctx_prompts_visual = self.ctx_v[:n_ctx, :]
        # Minimum can be 1, which defaults to shallow BMIP
        # compound prompts

        # Visual prompts are 768, as they are projected from 512
        if cfg.SHARE.SHARE_PARAMETER_VISUAL:
            self.compound_prompts_visual = nn.ParameterList([self.ctx_prompts_visual
                                                             for _ in range(self.compound_prompts_depth - 1)])
            self.single_layer_v = nn.Linear(ctx_v_dim, ctx_dim)
            self.exchange_layer_v = nn.Linear(ctx_v_dim * 2, ctx_v_dim)
        else:
            self.compound_prompts_visual = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_v_dim))
                                                             for _ in range(self.compound_prompts_depth - 1)])
            single_exchange_layer_v = nn.Linear(ctx_v_dim * 2, ctx_v_dim)
            self.compound_exchange_v = _get_clones(single_exchange_layer_v, self.compound_prompts_depth - 1)
            for single_para in self.compound_prompts_visual:
                # nn.init.normal_(single_para, std=0.02)
                nn.init.uniform_(single_para, -val_v, val_v)

            single_layer_v = nn.Linear(ctx_v_dim, ctx_dim)
            self.compound_prompt_projections_v = _get_clones(single_layer_v, self.compound_prompts_depth - 1)

        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
            
        if cfg.SHARE.SHARE_PARAMETER_TEXT:
            self.single_layer_t = nn.Linear(ctx_dim, ctx_v_dim)
            self.exchange_layer_t = nn.Linear(ctx_dim * 2, ctx_dim)
        else:
            # Also make corresponding projection layers, for each prompt
            single_layer_t = nn.Linear(ctx_dim, ctx_v_dim)
            single_exchange_layer_t = nn.Linear(ctx_dim * 2, ctx_dim)
            self.compound_prompt_projections_t = _get_clones(single_layer_t, self.compound_prompts_depth - 1)
            self.compound_exchange_t = _get_clones(single_exchange_layer_t, self.compound_prompts_depth - 1)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.share_para_visual = cfg.SHARE.SHARE_PARAMETER_VISUAL
        self.share_para_text = cfg.SHARE.SHARE_PARAMETER_TEXT
        # self.weight_threshold = cfg.TRAINER.BMIPLOSS.WEIGHT_THRESHOLD

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx_t = self.ctx_t.to(torch.float16)
        ctx_v = self.ctx_v[:self.n_ctx, :].to(torch.float16)
        # print(ctx_t.shape)
        # print(ctx_v.shape)

        if ctx_t.dim() == 2:
            ctx_t = ctx_t.unsqueeze(0).expand(self.n_cls, -1, -1)
            # ctx_v = ctx_v.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx_t, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        if self.share_para_text:
            for index in range(self.compound_prompts_depth - 1):
                visual_deep_prompts.append(self.single_layer_t(self.compound_prompts_text[index]))  # 768
        else:
            for index, layer in enumerate(self.compound_prompt_projections_t):
                visual_deep_prompts.append(layer(self.compound_prompts_text[index]))  # 768
        # visual_deep_prompts = visual_deep_prompts.to(torch.float16)

        text_deep_prompts = []

        if self.share_para_visual:
            for index in range(self.compound_prompts_depth - 1):
                text_deep_prompts.append(self.single_layer_v(self.compound_prompts_visual[index]))  # 512
        else:
            for index, layer in enumerate(self.compound_prompt_projections_v):
                text_deep_prompts.append(layer(self.compound_prompts_visual[index]))  # 512
        text_deep_prompts.append(ctx_t)
        # text_deep_prompts = text_deep_prompts.to(torch.float16)
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        shared_ctx_v = self.proj_v(ctx_v)
        # print(self.learnable_textw.grad)
        return prompts, self.proj_t(
            self.ctx_t), self.compound_prompts_text, visual_deep_prompts, ctx_v, shared_ctx_v, \
                self.compound_prompts_visual, text_deep_prompts
    # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.total_ctx = self.prompt_learner.n_ctx * 2
        self.connect_method = cfg.TRAINER.BMIPLOSS.CONNECT_METHOD
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        # self.tau = nn.Parameter(torch.tensor(1.0))

    # def compute_contrast_confidence(self, image, patch_size=16):
    #     """
    #     Use local standard deviation as clarity proxy.
    #     image: [B, 3, H, W]
    #     Returns: [B, N]
    #     """
    #     B, C, H, W = image.shape
    #     gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    #     # Unfold to patches
    #     patches = gray.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    #     # [B, 1, N_h, N_w, p, p]
    #     patches = patches.contiguous().view(B, -1, patch_size, patch_size)  # [B, N, p, p]

    #     # Compute std per patch
    #     std = patches.view(B, -1, patch_size * patch_size).std(dim=-1)  # [B, N]

    #     # Optional: clip to avoid noise amplification
    #     std = torch.clamp(std, min=0.0, max=0.3)  # adjust max based on your data range

    #     # Normalize to [0,1]
    #     confidence = std / (std.max(dim=1, keepdim=True).values + 1e-6)
    #     return confidence.to(self.dtype)

    def compute_contrast_confidence(self, image, patch_size=16, k=1.5):
        """
        Use local std with adaptive thresholding.
        k: multiplier for adaptive threshold (try 1.0 ~ 2.0)
        """
        B, C, H, W = image.shape
        gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Unfold to patches
        patches = gray.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, -1, patch_size * patch_size)  # [B, N, p²]

        # Compute std per patch
        std = patches.std(dim=-1)  # [B, N]

        # Adaptive threshold: mean + k * std of std
        std_mean = std.mean(dim=1, keepdim=True)   # [B, 1]
        std_std = std.std(dim=1, keepdim=True)     # [B, 1]
        adaptive_max = std_mean + k * std_std       # [B, 1]

        # Clamp with adaptive max
        std_clamped = torch.minimum(std, adaptive_max)

        # Normalize to [0,1]
        confidence = std_clamped / (adaptive_max + 1e-6)
        return confidence.to(self.dtype)

    def vision_to_text_fusion(self, text_features, patch_features):
        """
        text_features: [M, d]
        patch_features: [B, N, d]
        return: [B, M, d]
        """
        B, N, d = patch_features.shape
        M = text_features.shape[0]

        # 计算相似度: [B, M, N]
        attn = torch.einsum('bnd,md->bmn', patch_features, text_features)
        attn = F.softmax(attn, dim=-1)  # [B, M, N]

        # 聚合 patch 特征 -> [B, M, d]
        patch_agg = torch.einsum('bmn,bnd->bmd', attn, patch_features)

        # 融合 text + vision
        fused = patch_agg + text_features.unsqueeze(0)  # [B, M, d]

        # 在 batch 维度上取平均 -> [M, d]
        fused = fused.mean(dim=0)
        return fused
    
    def text_to_vision_fusion(self, p_a, patch_features):
        """
        p_a: [M, d]
        patch_features: [B, N, d]
        return: [B, N, d]
        """
        B, N, d = patch_features.shape
        M = p_a.shape[0]

        # patch-to-text 相似度: [B, N, M]
        attn = torch.einsum('bnd,md->bnm', patch_features, p_a)

        # softmax over text dim (M)
        attn = F.softmax(attn, dim=-1)  # [B, N, M]

        # 注意力加权 text -> [B, N, d]
        text_agg = torch.einsum('bnm,md->bnd', attn, p_a)

        # 融合 patch + weighted text
        fused = text_agg + patch_features  # [B, N, d]

        return fused

    def forward(self, image, label=None):
        eps = 1e-6
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        # print(f"logit_scale: {logit_scale}!!!!!!!!!!!!!")

        prompts, shared_ctx, deep_compound_prompts_text_t, deep_compound_prompts_vision_t, prompts_v, shared_ctx_v, \
            deep_compound_prompts_vision_v, deep_compound_prompts_text_v, \
            = self.prompt_learner()

        deep_compound_prompts_text = [list(deep_compound_prompts_text_t), list(deep_compound_prompts_text_v)]
        deep_compound_prompts_vision = [list(deep_compound_prompts_vision_v), list(deep_compound_prompts_vision_t)]

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        # image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        # patch_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, return_patches=True)

        image_features, patch_features = self.image_encoder(
            image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, return_patches=True
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        patch_features = F.normalize(patch_features, dim=-1)
        
        # >>> 新增：统一为模型 dtype（通常是 float16）
        image_features = image_features.to(self.dtype)
        text_features = text_features.to(self.dtype)
        patch_features = patch_features.to(self.dtype)
        # print(patch_features.shape) # [4, 196, 512]
        # print(image_features.shape) # [4, 512]
        # print(text_features.shape) # [15, 512]
        # <<< 新增结束

        if self.cfg.TRAINER.BMIPLOSS.LOSS_METHOD == "contrast": 
            patch_confidence = self.compute_contrast_confidence(image, patch_size=16, k=1.5)  # [B, 196]
            patch_features = patch_features * patch_confidence.unsqueeze(-1)      # [B, 196, d]

        p_a = self.vision_to_text_fusion(text_features, patch_features)  # [M, d]
        f_a = self.text_to_vision_fusion(p_a, patch_features)  # [B, N, d]

        # 归一化
        p_a = p_a / p_a.norm(dim=-1, keepdim=True)
        f_a = f_a / f_a.norm(dim=-1, keepdim=True)

        # 全局特征
        f_a_global = f_a.mean(dim=1)  # [B, d]

        # 三个 logits
        base_logits = image_features @ text_features.t()
        if self.cfg.TRAINER.BMIPLOSS.LOSS_TYPE == 1:
            new_logits = image_features @ p_a.t()
        elif self.cfg.TRAINER.BMIPLOSS.LOSS_TYPE == 2:
            new_logits = f_a_global @ text_features.t()
        elif self.cfg.TRAINER.BMIPLOSS.LOSS_TYPE == 3:
            new_logits = f_a_global @ p_a.t()
        else:
            new_logits = 0
        
        logits_3 = logit_scale * (base_logits + 0.2 * new_logits)        # 原始融合 logits

        if self.prompt_learner.training:
            return F.cross_entropy(logits_3, label)
        
        logits_1 = logit_scale * (f_a_global @ text_features.t())       # f_a + text_features
        logits_2 = logit_scale * (image_features @ p_a.t())              # p_a + image_features

        # --- 新增：收集中间变量 ---
        intermediates = {
            "patch_features": patch_features,               # [B, N, d]
            "f_a": f_a,                                     # [B, N, d]
            "p_a": p_a,                                     # [M, d]
            "image_features": image_features,               # [B, d]
            "text_features": text_features,                 # [M, d]
        }

        if self.cfg.TRAINER.BMIPLOSS.LOSS_METHOD == "contrast":
            intermediates["confidence"] = patch_confidence   # [B, N]
        else:
            intermediates["confidence"] = None

        attn_v2t = torch.einsum('bnd,md->bmn', patch_features, text_features)
        attn_v2t = F.softmax(attn_v2t, dim=-1)  # [B, M, N]
        attn_t2v = torch.einsum('bnd,md->bnm', patch_features, p_a)
        attn_t2v = F.softmax(attn_t2v, dim=-1)  # [B, N, M]
        intermediates["attn_v2t"] = attn_v2t
        intermediates["attn_t2v"] = attn_t2v

        if self.prompt_learner.training:
            return logits_1, logits_2, logits_3
        else:
            return logits_1, logits_2, logits_3, intermediates
        # --- 新增：收集中间变量 ---

        # return logits_1, logits_2, logits_3



def _get_clones(module, N):
    # 深度复制，不会共享
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class BMIPLOSS(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.no_improvement= 0
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        assert cfg.TRAINER.BMIPLOSS.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.device = cfg.DEVICE
        # print(self.device)
        classnames = self.dm.dataset.classnames

        print(f"BMIP: Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BMIPLOSS.PREC == "fp32" or cfg.TRAINER.BMIPLOSS.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("BMIP: Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                # if "VPT" in name:
                if "VPT" in name or "learnable_linear1" in name or "learnable_linear2" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        # 对enabled进行排序
        enabled = sorted(enabled)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.BMIPLOSS.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        # print(f"image.shape: {image.shape}")
        # print(f"label.shape:{label.shape}")
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.BMIPLOSS.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # 删除
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def get_features(self, images, norm=True):
        model = self.model
        model.eval()
        with torch.no_grad():
            prompts, shared_ctx, deep_compound_prompts_text_t, deep_compound_prompts_vision_t, prompts_v, shared_ctx_v, deep_compound_prompts_vision_v, deep_compound_prompts_text_v = self.prompt_learner()
            # 两个prompts加和,其中一个是ParameterList一个是list，请帮我转化为合适的形式
            deep_compound_prompts_text = []
            deep_compound_prompts_text = [list(deep_compound_prompts_text_t), list(deep_compound_prompts_text_v)]

            deep_compound_prompts_vision = []
            deep_compound_prompts_vision = [list(deep_compound_prompts_vision_v), list(deep_compound_prompts_vision_t)]

            text_features = model.text_encoder(prompts, model.prompt_learner.tokenized_prompts, deep_compound_prompts_text)
            image_features = model.image_encoder(images.type(model.dtype), shared_ctx, deep_compound_prompts_vision)
            # print(image_features.shape)
            # print(text_features.shape)
            if norm:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST_EARLY
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        # print("test????????????? {}, {}".format(do_test, self.cfg.TRAIN.PATIENT_COUNT))
        if do_test and self.cfg.TRAIN.PATIENT_COUNT != -1:
            
            with torch.no_grad():
                curr_result = self.test(split="val")
            # 加入提前停止，如果验证集上的性能没有提升次数大于3次，则停止训练
                if self.cfg.TRAIN.EARLY_STOPPING:
                    if curr_result > self.best_result:
                        self.no_improvement = 0
                    else:
                        self.no_improvement += 1
                        if self.no_improvement >= self.cfg.TRAIN.PATIENT_COUNT and self.epoch >= 5:
                            print(f"Early stopping at epoch {self.epoch}")
                            self.epoch = self.max_epoch + 1
                            self.early_stop = True
                            return
                print(f"now no improvement {self.no_improvement} times")
                if curr_result > self.best_result + 0.01:
                    self.best_result = curr_result
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        val_result=curr_result,
                        model_name="model-best.pth.tar"
                    )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.early_stop = False

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if self.early_stop:   # <-- 检查是否触发早停
                break
        self.after_train()
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if isinstance(output, (tuple, list)):
                logits_1, logits_2, logits_3 = output[:3]
            else:
                logits_1 = logits_2 = logits_3 = output
            # # 🔹 分别取每个的预测类别
            # pred_1 = logits_1.argmax(dim=1)
            # pred_2 = logits_2.argmax(dim=1)
            # pred_3 = logits_3.argmax(dim=1)

            # # 🔹 堆叠 [3, B]
            # preds = torch.stack([pred_1, pred_2, pred_3], dim=0)

            # # 🔹 多数投票
            # final_pred = torch.mode(preds, dim=0)[0]  # shape: [B]

            # # 🔹 转为 one-hot logits（兼容 evaluator）
            # num_classes = logits_1.shape[1]
            # final_logits = torch.zeros(final_pred.size(0), num_classes, device=self.device)
            # final_logits.scatter_(1, final_pred.unsqueeze(1), 1.0)
            
            # self.evaluator.process(final_logits, label)
            self.evaluator.process(logits_3, label)
        
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]



