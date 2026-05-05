# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
from Bio import PDB
from Bio.PDB.Atom import Atom
from torch_geometric.utils import add_self_loops

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 1️⃣ 提取电荷分布和极性特征
# -------------------------------
def get_charge_and_polarity(residue):
    positive_charged = ['ARG', 'LYS', 'HIS']
    negative_charged = ['ASP', 'GLU']
    neutral = ['ALA', 'CYS', 'PHE', 'GLY', 'ILE', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR']

    if residue.get_resname() in positive_charged:
        charge = 1
    elif residue.get_resname() in negative_charged:
        charge = -1
    else:
        charge = 0

    hydrophilic = ['SER', 'THR', 'ASN', 'GLN', 'GLY', 'CYS']
    hydrophobic = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'TYR']

    if residue.get_resname() in hydrophilic:
        polarity = 1
    elif residue.get_resname() in hydrophobic:
        polarity = -1
    else:
        polarity = 0

    return charge, polarity

# -------------------------------
# 2️⃣ 提取节点特征（只保留 CA 坐标）
# -------------------------------
def extract_node_features(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    coords = []
    aa_types = []
    charges = []
    polarities = []
    volumes = []

    aa_list = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']

    aa_volumes = {
        'ALA': 67.5, 'CYS': 114.0, 'ASP': 111.0, 'GLU': 138.0, 'PHE': 166.0, 'GLY': 47.0,
        'HIS': 135.0, 'ILE': 166.0, 'LYS': 146.0, 'LEU': 166.0, 'MET': 162.0, 'ASN': 114.0,
        'PRO': 112.0, 'GLN': 143.0, 'ARG': 174.0, 'SER': 89.0, 'THR': 112.0, 'VAL': 140.0,
        'TRP': 204.0, 'TYR': 193.0
    }

    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    ca_atom = residue['CA']

                    if ca_atom:
                        coords.append(ca_atom.get_coord())
                        aa_onehot = [1 if residue.get_resname() == aa else 0 for aa in aa_list]
                        aa_types.append(aa_onehot)
                        charge, polarity = get_charge_and_polarity(residue)
                        charges.append(charge)
                        polarities.append(polarity)
                        volumes.append(aa_volumes.get(residue.get_resname(), 0))

    coords = np.array(coords)
    aa_types = np.array(aa_types)
    charges = np.array(charges).reshape(-1, 1)
    polarities = np.array(polarities).reshape(-1, 1)
    volumes = np.array(volumes).reshape(-1, 1)

    node_features = np.concatenate([coords, aa_types, charges, polarities, volumes], axis=1)
    return node_features, coords

# -------------------------------
# 3️⃣ 构建邻接矩阵 + 边特征
# -------------------------------
def construct_edges(coords, distance_cutoff=8.0):
    N = coords.shape[0]
    edge_index = []
    edge_features = []

    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < distance_cutoff:
                edge_index.append([i, j])
                edge_index.append([j, i])

                # ---- 基础方向 ----
                direction = (coords[j] - coords[i]) / (dist + 1e-6)

                # ---- backbone 向量 (i-1 -> i) ----
                if i > 0:
                    backbone_vec = coords[i] - coords[i - 1]
                    backbone_vec = backbone_vec / (np.linalg.norm(backbone_vec) + 1e-6)
                    angle = np.arccos(
                        np.clip(np.dot(backbone_vec, direction), -1.0, 1.0)
                    )
                else:
                    angle = 0.0  # 第一个残基没有 backbone 方向

                seq_gap = abs(j - i)

                edge_feat = np.concatenate([[dist], direction, [seq_gap], [angle]], axis=0)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)

    edge_index = np.array(edge_index).T
    edge_features = np.array(edge_features)
    return edge_index, edge_features


# -------------------------------
# 4️⃣ 保存图特征
# -------------------------------
def save_graph_features(pdb_file, save_path, distance_cutoff=8.0):
    if os.path.exists(save_path):
        print(f"File {save_path} already exists, skipping...")
        return

    try:
        node_features, coords = extract_node_features(pdb_file)
        edge_index, edge_features = construct_edges(coords, distance_cutoff=distance_cutoff)

        # residue_index 对齐序列
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        residue_index = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue):
                        residue_index.append(residue.get_id()[1])
        residue_index = torch.tensor(residue_index, dtype=torch.long, device=device)

        node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        edge_features = torch.tensor(edge_features, dtype=torch.float32, device=device)

        features = {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "residue_index": residue_index
        }

        with open(save_path, 'wb') as f:
            pickle.dump(features, f)

        print(f"Graph features saved to {save_path}")

    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return

# -------------------------------
# 5️⃣ 批量处理 PDB 文件
# -------------------------------
if __name__ == "__main__":
    splits = ["train", "valid", "test"]

    for split in splits:
        pdb_dir = f"./Raw_data/UniProtSMB/{split}_U_structures"
        save_dir = f"./Dataset/{split}_U_structures"
        os.makedirs(save_dir, exist_ok=True)

        processed_files = 0
        skipped_files = []

        if not os.path.exists(pdb_dir):
            print(f"Directory {pdb_dir} does not exist, skipping...")
            continue

        for pdb_file in os.listdir(pdb_dir):
            if pdb_file.endswith(".pdb"):
                pdb_path = os.path.join(pdb_dir, pdb_file)
                save_path = os.path.join(save_dir, pdb_file.replace(".pdb", ".pkl"))
                save_graph_features(pdb_path, save_path, distance_cutoff=8.0)
                if os.path.exists(save_path):
                    processed_files += 1
                else:
                    skipped_files.append(pdb_file)

        print(f"[{split}] Total PDB files processed: {processed_files}")
        if skipped_files:
            print(f"[{split}] Skipped files due to errors: {', '.join(skipped_files)}")
        else:
            print(f"[{split}] No files were skipped.")

