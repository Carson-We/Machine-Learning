from datasets import load_dataset
from tqdm import tqdm
import time

# ==================== 配置区 ====================
DATASET_DIR = "/Users/carsonwu/Developer/Own/code/L.E.P.A.U.T.E./dataset/lepaute_train_dataset"

REPO_ID = "dev1virtuoso/lepaute-dataset"   # 你的用户名已正确
# ===============================================

print("正在初始化 imagefolder 加载器...")

# 增加进度条（imagefolder 内部扫描会显示部分进度）
dataset = load_dataset(
    "imagefolder",
    data_dir=DATASET_DIR,
    split=None,
    # streaming=False,   # 不要开启 streaming
)

print("\n数据集加载完成！结构如下：")
print(dataset)

# 可选：手动显示每个 split 的大小
for split_name in dataset.keys():
    print(f"{split_name} split 包含 {len(dataset[split_name])} 个样本")

print("\n开始上传到 Hugging Face（这会比较久）...")

# 上传（带进度）
dataset.push_to_hub(
    repo_id=REPO_ID,
    max_shard_size="3GB",
    private=False,
    commit_message="Upload lepaute dataset with frames (197GB)",
    num_proc=1,                    # 根据你的 CPU 核数调整，建议不要超过 8
)
print("上传完成！")