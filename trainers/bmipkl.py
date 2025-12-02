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

_tokenizer = _Tokenizer()

# 新增
def load_original_clip(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    # 关键：不传 design_details，或传全 0
    design_details = {
        "trainer": 'IVLP',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


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
    if cfg.TRAINER.BMIPKL.CONNECT_METHOD == "joint":
        BMIP_length = cfg.TRAINER.BMIPKL.N_CTX * 2
        connect_method = "joint"
    else:
        BMIP_length = cfg.TRAINER.BMIPKL.N_CTX
        connect_method = "other"
    # print(f"weight_threshold: {cfg.TRAINER.BMIPKL.WEIGHT_THRESHOLD}")
    # weight_threshold = cfg.TRAINER.BMIPKL.WEIGHT_THRESHOLD
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

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text, no_text=False, no_visual=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined, no_text, no_visual)
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
        n_ctx = cfg.TRAINER.BMIPKL.N_CTX
        ctx_init = cfg.TRAINER.BMIPKL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_v_dim = clip_model.visual.class_embedding.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        val_v = math.sqrt(6. / float(3 * reduce(mul, [16], 1) + ctx_v_dim))  # noqa
        val_t = math.sqrt(6. / float(3 * reduce(mul, [16], 1) + ctx_dim))  # noqa
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.BMIPKL.PROMPT_DEPTH >= 1, "For BMIP, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.BMIPKL.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # if cfg.TRAINER.BMIPKL.CONNECT_METHOD == "joint":
        #     tmp_ctx = cfg.TRAINER.BMIPKL.N_CTX * 2
        # else:
        #     tmp_ctx = cfg.TRAINER.BMIPKL.N_CTX
        if cfg.TRAINER.BMIPKL.CONNECT_METHOD != "joint":
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
        if cfg.TRAINER.BMIPKL.CONNECT_METHOD == "joint":
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
        # self.weight_threshold = cfg.TRAINER.BMIPKL.WEIGHT_THRESHOLD

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
        self.connect_method = cfg.TRAINER.BMIPKL.CONNECT_METHOD
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg


    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text_t, deep_compound_prompts_vision_t, prompts_v, shared_ctx_v, \
            deep_compound_prompts_vision_v, deep_compound_prompts_text_v, \
            = self.prompt_learner()

        deep_compound_prompts_text = [list(deep_compound_prompts_text_t), list(deep_compound_prompts_text_v)]
        deep_compound_prompts_vision = [list(deep_compound_prompts_vision_v), list(deep_compound_prompts_vision_t)]

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        local_text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text, no_text=True, no_visual=False)
        local_visual_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, no_text=False, no_visual=True)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        local_text_features = local_text_features/ local_text_features.norm(dim=-1, keepdim=True)
        local_image_features = local_visual_features/ local_visual_features.norm(dim=-1, keepdim=True)

        logits_trainable = logit_scale * image_features @ text_features.t()
        text_logits = logit_scale * image_features @ local_text_features.t()
        visual_logits = logit_scale * local_image_features @ text_features.t()

        if not self.prompt_learner.training:
            return logits_trainable

        ce_loss = F.cross_entropy(logits_trainable, label)
        return ce_loss, logits_trainable, text_logits, visual_logits


def _get_clones(module, N):
    # 深度复制，不会共享
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class BMIPKL(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.no_improvement= 0
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        assert cfg.TRAINER.BMIPKL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.device = cfg.DEVICE
        # print(self.device)
        classnames = self.dm.dataset.classnames

        print(f"BMIP: Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BMIPKL.PREC == "fp32" or cfg.TRAINER.BMIPKL.PREC == "amp":
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

        self.scaler = GradScaler() if cfg.TRAINER.BMIPKL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.BMIPKL.PREC
        T = 1.0  # temperature, can be tuned

        kl_weight = getattr(self.cfg.TRAINER.BMIPKL, "KL_LOSS_WEIGHT", 1.0)

        if prec == "amp":
            with autocast():
                ce_loss, logits_trainable, text_logits, visual_logits = model(image, label)
                # KL loss with log_target=True
                log_prob_train = F.log_softmax(logits_trainable / T, dim=1)
                log_prob_text = F.log_softmax(text_logits / T, dim=1)
                log_prob_visual = F.log_softmax(visual_logits / T, dim=1)
                text_kl_loss = F.kl_div(log_prob_train, log_prob_text, reduction='sum', log_target=True)
                text_kl_loss = text_kl_loss * (T * T) / logits_trainable.numel()
                visual_kl_loss = F.kl_div(log_prob_train, log_prob_visual, reduction='sum', log_target=True)
                visual_kl_loss = visual_kl_loss * (T * T) / logits_trainable.numel()
                loss = ce_loss + kl_weight * text_kl_loss + kl_weight * visual_kl_loss
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            ce_loss, logits_trainable, text_logits, visual_logits = model(image, label)
            # KL loss with log_target=True
            log_prob_train = F.log_softmax(logits_trainable / T, dim=1)
            log_prob_text = F.log_softmax(text_logits / T, dim=1)
            log_prob_visual = F.log_softmax(visual_logits / T, dim=1)
            text_kl_loss = F.kl_div(log_prob_train, log_prob_text, reduction='sum', log_target=True)
            text_kl_loss = text_kl_loss * (T * T) / logits_trainable.numel()
            visual_kl_loss = F.kl_div(log_prob_train, log_prob_visual, reduction='sum', log_target=True)
            visual_kl_loss = visual_kl_loss * (T * T) / logits_trainable.numel()
            loss = ce_loss + kl_weight * text_kl_loss + kl_weight * visual_kl_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "text_kl_loss": text_kl_loss.item(),
            "visual_kl_loss": visual_kl_loss.item()
        }

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
            
            # 新增
            if "zero_shot_text_features" in state_dict:
                del state_dict["zero_shot_text_features"]

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
            print(image_features.shape)
            print(text_features.shape)
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



