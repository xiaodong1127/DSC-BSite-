import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
# from triplet import TripletClassificationModel
from a_test import TripletClassificationModel
import count

# ================== Parameters ==================
threshold = 0.5
esm_pkl = './Dataset/SM/esm_test_UniProtSMB.pkl'
struct_pkl = './Raw_data/UniProtSMB/test_coords_mask.pkl'
model_ckpt = './triplet_classification/SM/full/11-21-04-15-36/epoch=24-step=198600.ckpt'
output_file = 'CLAPE_SMB_result.txt'

# ================== Device ==================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================== Load precomputed features ==================
with open(esm_pkl, 'rb') as f:
    esm_dict = pickle.load(f)
with open(struct_pkl, 'rb') as f:
    struct_dict = pickle.load(f)

print(f"Loaded {len(esm_dict)} proteins with ESM features from {esm_pkl}")
print(f"Loaded {len(struct_dict)} proteins with structure features from {struct_pkl}")

# 对齐 id
common_ids = sorted(set(esm_dict.keys()) & set(struct_dict.keys()))
print(f"Found {len(common_ids)} common proteins")

# ================== Prepare features ==================
esm_feats = [esm_dict[k][0].to(device) for k in common_ids]
labels_str = [esm_dict[k][1] for k in common_ids]

coords = [struct_dict[k]['coords'].to(device) for k in common_ids]
masks = [struct_dict[k]['mask'].to(device) for k in common_ids]

labels = [np.array([int(c) for c in s], dtype=np.int32) for s in labels_str]

# ================== Load checkpoint ==================
ckpt = torch.load(model_ckpt, map_location=device)
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# 根据训练代码获取 samples_per_class
samples_per_class = count.count('./Raw_data/train_UniProtSMB.txt')

model_params = {
    'alpha': None,
    'margin': 0.5,
    'clw': 0.2,
    'clf_lr': 1e-4,
    'loss_lr': 1e-5,
    'gamma': 2,
    'loss': 'focal',
    'samples_per_class': samples_per_class,
    'batch_size': 1
}

predictor = TripletClassificationModel(**model_params).to(device)

# 只加载 full_model 权重
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("full_model."):
        name = k.replace("full_model.", "")
        if name in predictor.full_model.state_dict():
            new_state_dict[name] = v
predictor.full_model.load_state_dict(new_state_dict)
predictor.eval()
print("Model loaded successfully!")

# ================== Prediction ==================
all_scores = []

print("=====Predicting Small molecules-binding sites=====")
with torch.no_grad():
    for f, c, m, l in tqdm(zip(esm_feats, coords, masks, labels), total=len(common_ids)):
        # batch_size=1, 保持 tensor 维度 [1,L,d] / [1,L,3] / [1,L]
        f_tensor = f.unsqueeze(0)  # [1, L, 1280]
        c_tensor = c.unsqueeze(0)  # [1, L, 3]
        m_tensor = m.unsqueeze(0)  # [1, L]
        l_tensor = torch.tensor(l, device=device).unsqueeze(0)  # [1, L]

        score, _ = predictor.full_model(f_tensor, c_tensor, m_tensor)  # logits, embedding
        prob = score.squeeze(0).cpu().numpy()[:, 1]  # 正类概率
        all_scores.append(prob)

# ================== Flatten labels and scores ==================
labels_flat = np.concatenate(labels)
all_scores_flat = np.concatenate(all_scores)

# ================== Evaluation function ==================
def compute_metrics(labels_flat, preds_flat, scores_flat):
    recall = recall_score(labels_flat, preds_flat, zero_division=0)
    precision = precision_score(labels_flat, preds_flat, zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, zero_division=0)
    mcc = matthews_corrcoef(labels_flat, preds_flat)
    try:
        auroc = roc_auc_score(labels_flat, scores_flat)
    except:
        auroc = float('nan')
    return recall, precision, f1, mcc, auroc

# ----- Default threshold evaluation -----
preds_default = [(s > threshold).astype(int) for s in all_scores]
preds_flat_default = np.concatenate(preds_default)
recall, precision, f1, mcc, auroc = compute_metrics(labels_flat, preds_flat_default, all_scores_flat)
print(f"=====Evaluation Metrics @ threshold={threshold}=====")
print(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, AUROC: {auroc:.4f}")

# ----- Auto threshold selection -----
best_f1 = -1
best_threshold = threshold
for t in np.arange(0.0, 1.01, 0.01):
    preds_t = [(s > t).astype(int) for s in all_scores]
    preds_flat_t = np.concatenate(preds_t)
    f1_t = f1_score(labels_flat, preds_flat_t, zero_division=0)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = t

preds_best = [(s > best_threshold).astype(int) for s in all_scores]
preds_flat_best = np.concatenate(preds_best)
recall_b, precision_b, f1_b, mcc_b, auroc_b = compute_metrics(labels_flat, preds_flat_best, all_scores_flat)
print(f"=====Evaluation Metrics @ best threshold={best_threshold:.4f}=====")
print(f"Recall: {recall_b:.4f}, Precision: {precision_b:.4f}, F1: {f1_b:.4f}, MCC: {mcc_b:.4f}, AUROC: {auroc_b:.4f}")

# ================== Save results ==================
print(f"=====Writing result files into {output_file}=====")
with open(output_file, 'w') as f_out:
    for i, pid in enumerate(common_ids):
        f_out.write(pid + '\n')
        f_out.write(' '.join(map(str, labels[i])) + '\n')
        f_out.write(' '.join(map(str, preds_best[i])) + '\n')

print(f"All process done! Your result file is saved as {output_file}")
