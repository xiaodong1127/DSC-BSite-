# check_fused_dataset.py
import pickle
import numpy as np

dataset_files = [
    './Dataset/fused_train_UniProtSMB.pkl',
    './Dataset/fused_valid_UniProtSMB.pkl'
]

for file in dataset_files:
    print(f"\nChecking file: {file}")
    try:
        data = pickle.load(open(file, 'rb'))
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        continue

    if not isinstance(data, dict):
        print("Data is not a dictionary!")
        continue

    valid_count = 0
    invalid_count = 0
    for pid, rec in data.items():
        try:
            if len(rec) not in [3, 4]:
                raise ValueError(f"Expected 3 or 4 elements, got {len(rec)}")

            x = np.array(rec[0], dtype=float)
            label = np.array(rec[1], dtype=int)
            mask = np.array(rec[2], dtype=bool)
            if len(rec) == 4:
                coords = np.array(rec[3], dtype=float)
            else:
                coords = x[:, :3]  # fallback

            valid_count += 1
        except Exception as e:
            print(f"[Warning] Skipping protein {pid} due to invalid data: {e}")
            invalid_count += 1

    print(f"Total proteins: {len(data)}, valid: {valid_count}, invalid: {invalid_count}")
