import torch
import sys
import os
import argparse

# 1. 导入必要的库和你训练时用的trainer/model定义
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform
from dassl.utils import setup_logger
from dassl.data.data_manager import build_data_loader # 用于获取 classnames
from dassl.data.datasets import DATASET_REGISTRY # 用于获取 dataset class
from train import setup_cfg

def main(args):
    # 2. 加载配置
    cfg = setup_cfg(args)
    # 加载你训练时用的 config 文件！这是关键，确保 DATASET.NAME 等配置正确
    cfg.merge_from_file("PATH/TO/YOUR_TRAIN_CONFIG.yaml") # 例如: configs/trainers/bmip_loss/xxx.yaml
    # 如果需要，可以用命令行参数覆盖一些设置，如数据集路径
    # cfg.DATASET.ROOT = "your_dataset_root_path" 
    cfg.freeze()

    # 3. 设置日志 (可选)
    setup_logger(cfg.OUTPUT_DIR)

    # 4. 加载数据集管理器，获取 classnames
    # 这里复用了 dassl 的逻辑来加载数据集
    dataset_class = DATASET_REGISTRY.get(cfg.DATASET.NAME)
    dataset = dataset_class(cfg) # 实例化数据集类
    classnames = dataset.classnames # 获取类别名称列表
    print(f"Loaded {len(classnames)} classes: {classnames[:5]}...") # 打印前几个类别确认

    # 选取一个可用的数据划分（优先 test，其次 val，最后 train），提取其中图像路径
    if hasattr(dataset, "test") and len(dataset.test) > 0:
        sample_split = dataset.test
        split_name = "test"
    elif hasattr(dataset, "val") and len(dataset.val) > 0:
        sample_split = dataset.val
        split_name = "val"
    else:
        sample_split = dataset.train_x
        split_name = "train"

    sample_records = sample_split[:4]  # 取前4张做示例
    image_paths = [record.impath for record in sample_records]
    print(f"从 {split_name} 划分中选取 {len(image_paths)} 张图片: {image_paths}")

    # 5. 构建模型 (与 model.txt 中的 build_model 逻辑类似)
    from trainers.bmip_loss import CustomCLIP, load_clip_to_cpu

    clip_model = load_clip_to_cpu(cfg)
    model = CustomCLIP(cfg, classnames, clip_model)

    # 6. 加载训练好的权重
    model_dir = "PATH/TO/YOUR_MODEL_DIR" # 例如: output/bmiploss/oxford_pets/RN50/best.pth.tar 所在的目录
    model_file = "model-best.pth.tar" # 或者你想要加载的具体文件名
    model_path = os.path.join(model_dir, "MultiModalPromptLearner", model_file) # 假设权重保存在 MultiModalPromptLearner 下

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # 删除 token_prefix/suffix，因为它们在加载时会被忽略
    for k in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
        state_dict.pop(k, None)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to("cuda") # 或 "cpu"

    # 7. 准备你的几张图片
    from PIL import Image
    transform = build_transform(cfg, is_train=False) # 使用测试时的 transform

    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = transform(img) # [C, H, W]
        images.append(img)
    images = torch.stack(images).to("cuda") # [B, C, H, W]

    # 8. (可选) 修改模型 forward 以返回中间结果 (参考上一个回答)
    # ... (修改 CustomCLIP.forward 的代码) ...

    # 9. 前向传播
    with torch.no_grad():
        # 如果你没有修改 forward，它会返回 logits_1, logits_2, logits_3
        # 如果你修改了 forward 返回 intermediates，它会返回 (logits_1, logits_2, logits_3, intermediates)
        logits_1, logits_2, logits_3 = model(images) 
        # 或者 (logits_1, logits_2, logits_3, intermediates) = model(images)

        # 获取预测结果
        pred_3 = logits_3.argmax(dim=1) # 选择 logits_3 的预测
        predicted_classes = [classnames[i] for i in pred_3.cpu().numpy()]
        print(f"Predictions: {predicted_classes}")

    # 10. 可视化中间结果 (如果修改了 forward)
    # if 'intermediates' in locals():
    #     confidence = intermediates["confidence"]
    #     attn_v2t = intermediates["attn_v2t"]
    #     # ... (可视化代码，如上一个回答所述) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument("--device", default='cuda:0', type=str, help="the device you will use")
    parser.add_argument("--best", action="store_true", help="load best model")
    parser.add_argument("--draw", action="store_true", help="draw distribution")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)