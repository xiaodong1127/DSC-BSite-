import pickle
import torch

esm_pkl = './Dataset/SM/esm_test_UniProtSMB.pkl'

esm_data = pickle.load(open(esm_pkl, 'rb'))

# 随便取一个蛋白
protein_id = list(esm_data.keys())[0]
esm_feat, label = esm_data[protein_id]

# 查看类型和 shape
print(type(esm_feat))
if isinstance(esm_feat, torch.Tensor):
    print("shape:", esm_feat.shape)
else:
    esm_feat = torch.tensor(esm_feat)
    print("shape:", esm_feat.shape)
