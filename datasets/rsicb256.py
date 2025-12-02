import os
import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "airport_runway": "airport runway",
    "artificial_grassland": "artificial grassland",
    "bare_land": "bare land",
    "city_building": "city building",
    "dry_farm": "dry farm",
    "green_farmland": "green farmland",
    "parkinglot": "parking lot",
    "river_protection_forest": "river protection forest",
    "sandbeach": "sand beach",
    "snow_mountain": "snow mountain",
    "sparse_forest": "sparse forest",
    "storage_room": "storage room"
}


@DATASET_REGISTRY.register()  # 注册数据集到DASSL框架的注册表
class RSICB256(DatasetBase):
    dataset_dir = "RSICB256"  # 数据集在文件系统中的根目录名称

    def __init__(self, cfg):
        # 解析数据集根目录（从配置中获取并标准化路径）
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)  # 完整数据集路径
        self.image_dir = os.path.join(self.dataset_dir, "images")  # 图像存放目录
        self.split_path = os.path.join(self.dataset_dir, "split_chen_RSICB256.json")  # 数据分割文件路径
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")  # 少样本缓存目录
        
        # 创建少样本缓存目录（如果不存在）
        mkdir_if_missing(self.split_fewshot_dir)
        
        # 加载预定义的数据分割或生成新分割
        if os.path.exists(self.split_path):
            print(f"Loading pre-defined data split from {self.split_path}")
            # 使用OxfordPets的工具函数读取分割文件
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            print(f"Generating new data split for RSICB256 and saving to {self.split_path}")
            # 使用DTD的工具函数读取图像并生成分割（应用类名映射）
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            # 使用OxfordPets的工具函数保存分割结果
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        
        # 处理少样本学习场景（当设置了样本数时）
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED  # 随机种子，确保结果可复现
            # 少样本数据缓存文件路径
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading pre-generated few-shot data from {preprocessed}")
                # 读取缓存的少样本数据
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                print(f"Generating few-shot data with {num_shots} shots per class")
                # 从完整数据集中采样生成少样本训练集
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                # 生成少样本验证集（最多4个样本/类，避免验证集样本过少）
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                # 缓存少样本数据以避免重复生成
                data = {"train": train, "val": val}
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 执行类别子采样（根据配置选择需要的类别）
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        
        # 调用父类初始化方法，完成数据集注册
        super().__init__(train_x=train, val=val, test=test)
    
    def update_classname(self, dataset_old):
        """
        根据NEW_CNAMES映射表更新数据集的类名
        
        Args:
            dataset_old: 原始数据集（包含旧类名）
        
        Returns:
            dataset_new: 类名更新后的新数据集
        """
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            # 使用 get() 方法，若类名不在映射表中，返回原始名称
            cname_new = NEW_CNAMES.get(cname_old, cname_old)  # 关键逻辑
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new