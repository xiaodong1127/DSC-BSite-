# -*- coding: utf-8 -*-
# @Description  : predicting process with metrics

import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, roc_auc_score, matthews_corrcoef
from model import ContinueModel

# 参数设置
threshold = 0.5
input_pkl = "./Dataset/SM/esm_test_UniProtSMB.pkl"   # 存有 (feature, label) 的文件
ckpt_path = "./Models/UniProtSMB/random_seed/6.ckpt"
output_file = "CLAPE_SMB_result.txt"

# ===== 加载数据 =====
print("=====Loading pre-extracted ESM features & labels=====")
with open(input_pkl, "rb") as f:
    data = pickle.load(f)  # {protein_id: (feature, label)}
print(f"Loaded {len(data)} proteins from {input_pkl}")

seq_ids = []
features = []
labels = []

for pid, (feat, label) in data.items():
    seq_ids.append(pid)

    # feature 处理
    if isinstance(feat, torch.Tensor):
        arr = feat.float().numpy()
    elif isinstance(feat, np.ndarray):
        arr = feat.astype(np.float32)
    elif isinstance(feat, list):
        arr = np.array(feat, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported feature type: {type(feat)}")

    features.append(torch.tensor(arr).unsqueeze(0))  # [1, L, d_esm]

    # label 处理
    if isinstance(label, (int, np.integer)):
        # 如果是超大整数，按字符串展开成 0/1 序列
        label_str = str(label)
        label_arr = np.array([int(ch) for ch in label_str], dtype=int)
    elif isinstance(label, str):
        label_arr = np.array([int(ch) for ch in label], dtype=int)
    else:
        # 已经是 list / numpy array
        label_arr = np.array(label, dtype=int)

    labels.append(label_arr)


print("Done!")

# ===== 加载分类模型 =====
print("=====Loading classification model=====")
predictor = ContinueModel()
predictor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
predictor.eval()
print("Done!")

# ===== 推理过程 =====
all_preds = []
all_trues = []
results = []

print("=====Predicting Small molecules-binding sites=====")
for f, true_label in tqdm(zip(features, labels), total=len(features)):
    with torch.no_grad():
        out = predictor(f)[0].squeeze(0).numpy()[:, 1]  # 残基正类概率

    pred_bin = (out > threshold).astype(int)

    results.append(''.join(map(str, pred_bin)))  # 保存结果
    all_preds.extend(pred_bin.tolist())
    all_trues.extend(true_label.tolist())

print("Done!")

# ===== 计算指标 =====
print("=====Calculating metrics=====")
recall = recall_score(all_trues, all_preds, zero_division=0)
precision = precision_score(all_trues, all_preds, zero_division=0)
try:
    auroc = roc_auc_score(all_trues, all_preds)
except ValueError:
    auroc = float('nan')
mcc = matthews_corrcoef(all_trues, all_preds)

print(f"Recall: {recall:.4f}, Precision: {precision:.4f}, AUROC: {auroc:.4f}, MCC: {mcc:.4f}")

# ===== 保存结果 =====
print(f"=====Writing result files into {output_file}=====")
with open(output_file, 'w') as f:
    for i, pid in enumerate(seq_ids):
        f.write(pid + '\n')
        f.write(results[i] + '\n')
print(f"Congrats! All process done! Your result file is saved as {output_file}")
