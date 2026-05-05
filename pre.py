# -*- coding: utf-8 -*-
# @Time         : 2024/6/24 12:00
# @Author       : Jue Wang and Yufan Liu (modified for checkpoint saving)
# @Description  : Generate protein sequence embeddings by ESM-2 on CPU with resume support

import os
import pickle
import torch
import esm
from tqdm import tqdm

# 指定 CPU
device = torch.device('cpu')

# esm-v2
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()

batch_converter = alphabet.get_batch_converter()

# 数据路径
Name = 'train_UniProtSMB'
data_file = "./Raw_data/UniProtSMB/" + Name + ".txt"
save_file = "./Dataset/SM/esm_" + Name + ".pkl"

# 如果已有部分结果，加载；否则新建
if os.path.exists(save_file):
    data_dict = pickle.load(open(save_file, 'rb'))
    print(f"继续运行：已完成 {len(data_dict)} 条数据")
else:
    data_dict = {}
    print("新建结果字典，开始计算")

# 读取原始数据
data = open(data_file, 'r').readlines()

for i in tqdm(range(len(data))):
    if data[i].startswith('>'):
        pid = data[i].strip()[1:]
        if pid in data_dict:  # 已经算过的直接跳过
            continue

        seq = [(pid, data[i + 1].strip())]
        batch_labels, batch_strs, batch_tokens = batch_converter(seq)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # 存储序列嵌入和标签
        data_dict[pid] = (
            token_representations.squeeze(0)[1:-1, :].cpu(),  # 序列嵌入
            data[i + 2].strip()  # 标签
        )

        # 每处理一个序列就保存一次，避免中途丢失
        with open(save_file, 'wb') as f:
            pickle.dump(data_dict, f)

print(f"全部完成，共保存 {len(data_dict)} 条数据到 {save_file}")
