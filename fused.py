# fused_flexible_egnn.py
import os
import pickle
import torch
import numpy as np

def fuse_esm_struct_for_egnn(esm_pkl, struct_dir, out_pkl, method='mask'):
    """
    method: 'mask'|'pad_zero'|'truncate'
      - 'mask' (推荐): 保留序列全长，用0占位结构缺失位置，返回 dict: {coords, x, mask, label}
      - 'pad_zero': 保留序列全长，用0占位
      - 'truncate': 只保留两者交集
    """
    assert method in ('mask','pad_zero','truncate')

    esm_data = pickle.load(open(esm_pkl, 'rb'))  # {pid: (esm_feat, label)}
    fused_data = {}
    warnings = []
    processed = 0

    for protein_id, (esm_feat, label) in esm_data.items():
        struct_path = os.path.join(struct_dir, f"{protein_id}.pkl")
        if not os.path.exists(struct_path):
            warnings.append(f"结构特征缺失 {protein_id}，跳过")
            continue

        struct_dict = pickle.load(open(struct_path, 'rb'))
        if "node_features" not in struct_dict or "residue_index" not in struct_dict:
            warnings.append(f"{protein_id} 缺少 node_features 或 residue_index，跳过")
            continue

        # -------------------------
        # ESM 特征 (1280维)
        # -------------------------
        esm_feat_tensor = torch.as_tensor(esm_feat, dtype=torch.float)
        L_seq = esm_feat_tensor.size(0)

        # -------------------------
        # 坐标
        # -------------------------
        node_feat = struct_dict["node_features"]
        coords_raw = node_feat[:, :3] if isinstance(node_feat, np.ndarray) else node_feat[:, :3].cpu().numpy()
        coords_tensor = torch.zeros((L_seq, 3), dtype=torch.float)

        residue_index = struct_dict["residue_index"]
        if isinstance(residue_index, torch.Tensor):
            residue_index_list = [int(x) for x in residue_index.cpu().numpy()]
        else:
            residue_index_list = [int(x) for x in np.asarray(residue_index).flatten()]

        mask = torch.zeros((L_seq,), dtype=torch.bool)
        out_of_range = 0
        for i, rid in enumerate(residue_index_list):
            idx0 = int(rid) - 1
            if 0 <= idx0 < L_seq:
                coords_tensor[idx0] = torch.tensor(coords_raw[i], dtype=torch.float)
                mask[idx0] = True
            else:
                out_of_range += 1
        if out_of_range > 0:
            warnings.append(f"{protein_id}: {out_of_range}/{len(residue_index_list)} 个残基 residue_index 超出序列范围")

        # -------------------------
        # 填充策略
        # -------------------------
        x = esm_feat_tensor  # 只用 ESM 特征

        if method == 'truncate':
            idx_keep = mask.nonzero(as_tuple=False).squeeze(1)
            if idx_keep.numel() == 0:
                warnings.append(f"{protein_id}: 无可对齐残基，跳过")
                continue
            coords_tensor = coords_tensor[idx_keep]
            x = x[idx_keep]
            mask = None  # truncate 后不需要 mask

        fused_data[protein_id] = {
            'coords': coords_tensor,  # [L_seq,3]
            'x': x,                   # [L_seq,1280]
            'mask': mask,             # [L_seq] 或 None
            'label': label
        }

        processed += 1

    # 保存
    with open(out_pkl, 'wb') as f:
        pickle.dump(fused_data, f)

    print(f"处理完成: 共处理 {processed} 个蛋白，保存到 {out_pkl}，条目数 {len(fused_data)}")
    if warnings:
        print("警告示例（部分）:")
        for w in warnings[:20]:
            print(" -", w)
    return fused_data

# -----------------------------
# 示例：直接运行
# -----------------------------
if __name__ == "__main__":
    splits = ['train', 'valid', 'test']
    method = 'mask'
    for split in splits:
        esm_file = f'./Dataset/SM/esm_{split}_UniProtSMB.pkl'
        struct_folder = f'./Dataset/{split}_U_structures'
        out_file = f'./Dataset/fused_{split}_UniProtSMB.pkl'
        print(f"\n=== processing {split} (method={method}) ===")
        fused = fuse_esm_struct_for_egnn(esm_file, struct_folder, out_file, method=method)

        # 打印示例
        cnt = 0
        for pid, rec in fused.items():
            coords = rec['coords']
            x = rec['x']
            mask = rec['mask']
            label = rec['label']
            num_ones = int(mask.sum()) if mask is not None else 'N/A'
            num_zeros = (mask.numel() - mask.sum()) if mask is not None else 'N/A'
            print(f"Sample {pid}: coords={tuple(coords.shape)}, x={tuple(x.shape)}, "
                  f"mask ones={num_ones}, zeros={num_zeros}, total={mask.shape[0] if mask is not None else 'N/A'}")
            cnt += 1
            if cnt >= 3:
                break

    print("\n所有 split 处理完成。")
