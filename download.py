import os
import requests
import random
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm

# ===================== 配置区域 =====================
# 1. 设置目标数量
TARGET_COUNTS = {
    'train': 5000, 
    'val': 1200     
}

# 2. 路径配置
BASE_DIR = "/data/10T/yhy/stable_signature/coco"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# 3. 其他参数
MIN_SIZE = 256
RANDOM_SEED = 42  # 核心参数：保证分类永远固定
# ===================================================

def setup_directories():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

def get_truth_split(coco):
    """
    生成“上帝视角”的分类标准。
    返回两个集合：train_ids, val_ids
    """
    print("正在构建分类标准 (Seed=42)...")
    # 1. 筛选 ID
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids_with_person = set()
    for cat_id in cat_ids:
        img_ids_with_person.update(coco.getImgIds(catIds=cat_id))
    
    all_img_ids = set(coco.getImgIds())
    non_person_ids = list(all_img_ids - img_ids_with_person)
    
    # 2. 打乱 (标准排序后再打乱，保证绝对一致性)
    non_person_ids.sort()
    random.seed(RANDOM_SEED)
    random.shuffle(non_person_ids)
    
    # 3. 切分 (前80%给Train, 后20%给Val)
    split_idx = int(len(non_person_ids) * 0.8)
    
    train_set = set(non_person_ids[:split_idx])
    val_set = set(non_person_ids[split_idx:])
    
    print(f"分类标准已建立: Train池 {len(train_set)} 张, Val池 {len(val_set)} 张")
    return train_set, val_set

def move_file(fname, src_dir, dst_dir, desc="移动"):
    """安全移动文件"""
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)
    
    if not os.path.exists(src_path):
        return False
        
    # 如果目标处已经有了
    if os.path.exists(dst_path):
        # 直接删除源文件（因为目标处已有，相当于合并）
        os.remove(src_path)
        return True
    
    # 移动文件
    shutil.move(src_path, dst_path)
    return True

def reorganize_existing_files(train_set, val_set):
    """
    【核心逻辑】检查现有的所有文件，如果不走错了房间，就强制移回去。
    """
    print("\nStep 1: 检查并自动归位现有图片...")
    
    moves_count = 0
    
    # --- 检查 Val 文件夹 ---
    # 这里的图片应该都在 val_set 里，如果在 train_set 里，就是走错门了
    val_files = [f for f in os.listdir(VAL_DIR) if f.endswith('.jpg')]
    for f in val_files:
        try:
            # 文件名转ID (去除前导0)
            img_id = int(f.split('.')[0])
            
            if img_id in train_set:
                # 错误：它属于 Train，却在 Val 里
                move_file(f, VAL_DIR, TRAIN_DIR)
                moves_count += 1
        except ValueError:
            pass

    # --- 检查 Train 文件夹 ---
    # 这里的图片应该都在 train_set 里，如果在 val_set 里，就是走错门了
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')]
    for f in train_files:
        try:
            img_id = int(f.split('.')[0])
            
            if img_id in val_set:
                # 错误：它属于 Val，却在 Train 里
                move_file(f, TRAIN_DIR, VAL_DIR)
                moves_count += 1
        except ValueError:
            pass
            
    print(f"整理完成: 共修正了 {moves_count} 张图片的位置。")

def download_missing(mode, target_ids, coco):
    """下载缺失的图片"""
    target_dir = TRAIN_DIR if mode == 'train' else VAL_DIR
    target_num = TARGET_COUNTS[mode]
    
    # 计算当前数量
    current_files = [f for f in os.listdir(target_dir) if f.endswith('.jpg')]
    current_count = len(current_files)
    needed = target_num - current_count
    
    print(f"\nStep 2: 检查 {mode} 集缺口...")
    print(f"  - 目标: {target_num}")
    print(f"  - 现有: {current_count}")
    
    if needed <= 0:
        print(f"  - ✅ 数量已达标，无需下载。")
        return

    print(f"  - ⚠️ 需要下载: {needed} 张")
    
    # 开始下载
    success = 0
    # 将 target_ids 转换为列表以便遍历
    id_list = list(target_ids)
    # 简单排序或打乱均可，这里直接遍历
    
    pbar = tqdm(total=needed, desc=f"下载 {mode}")
    
    for img_id in id_list:
        if success >= needed:
            break
            
        img_info = coco.loadImgs(img_id)[0]
        fname = img_info['file_name']
        save_path = os.path.join(target_dir, fname)
        
        # 如果已经存在（可能刚被移过来），跳过
        if os.path.exists(save_path):
            continue
            
        # 尺寸检查
        if img_info['width'] < MIN_SIZE or img_info['height'] < MIN_SIZE:
            continue
            
        try:
            response = requests.get(img_info['coco_url'], stream=True, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                success += 1
                pbar.update(1)
        except:
            pass
            
    pbar.close()

def main():
    setup_directories()
    
    # 准备 COCO API
    ann_file = 'annotations/instances_train2017.json'
    if not os.path.exists(ann_file):
        # ... (此处省略下载解压代码，假设你已有，或复制之前的下载逻辑)
        print("请确保 annotations 文件夹已存在 (可使用之前的代码下载)")
        return 
        
    coco = COCO(ann_file)
    
    # 1. 获取上帝视角的分类标准
    train_set_ids, val_set_ids = get_truth_split(coco)
    
    # 2. 【关键】根据标准，清洗现有文件
    reorganize_existing_files(train_set_ids, val_set_ids)
    
    # 3. 补全 Val 集
    download_missing('val', val_set_ids, coco)
    
    # 4. 补全 Train 集
    download_missing('train', train_set_ids, coco)

    print("\n✅ 所有任务完成！数据已彻底清洗并补全。")

if __name__ == "__main__":
    main()