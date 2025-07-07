import os
import re
import random
import h5py
from typing import Dict, Optional
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModel

def load_additional_labels_from_csv(csv_path):

    lymph_node_status = {}
    vascular_status    = {}
    perineural_status  = {}

    df = pd.read_csv(csv_path, dtype=str)
    if '病理号' not in df.columns:
        raise ValueError("CSV 文件必须包含 '病理号' 列")

    col_map = {'lymph': None, 'vascular': None, 'perineural': None}
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ['淋巴结', 'lymph']):
            col_map['lymph'] = col
        if any(k in low for k in ['脉管', 'vascular']):
            col_map['vascular'] = col
        if any(k in low for k in ['神经束', 'perineural']):
            col_map['perineural'] = col

    for k, v in col_map.items():
        if v is None:
            raise ValueError(f"CSV 中未找到 {k} 标签列")

    for _, row in df.iterrows():
        pid = str(row['病理号']).strip()
        m = re.search(r'(\d+)', pid)
        if not m:
            continue
        pid = m.group(1)

        def parse_bool(val):
            if pd.isna(val):
                return None
            s = str(val).lower()
            if any(x in s for x in ['1','是','有','yes','positive']):
                return 1
            if any(x in s for x in ['0','否','无','no','negative']):
                return 0
            try:
                return int(float(s))
            except:
                return None

        ln = parse_bool(row[col_map['lymph']])
        va = parse_bool(row[col_map['vascular']])
        pe = parse_bool(row[col_map['perineural']])
        if ln is not None:
            lymph_node_status[pid] = ln
        if va is not None:
            vascular_status[pid] = va
        if pe is not None:
            perineural_status[pid] = pe

    print(f"Loaded labels from CSV: "
          f"{len(lymph_node_status)} lymph, "
          f"{len(vascular_status)} vascular, "
          f"{len(perineural_status)} perineural")
    return lymph_node_status, vascular_status, perineural_status

# Lazy loading dataset for pathology multimodal data
class PathologyMultimodalDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 lymph_node_status: Optional[Dict] = None,
                 vascular_status: Optional[Dict] = None,
                 perineural_status: Optional[Dict] = None,
                 missing_probs: Optional[Dict[int, float]] = None):
        super().__init__()
        self.data_dir = data_dir
        self.missing_probs = missing_probs or {}
        
        self.file_paths = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.h5'):
                    self.file_paths.append(os.path.join(root, f))
        
        self.lymph_node_status = lymph_node_status or {}
        self.vascular_status = vascular_status or {}
        self.perineural_status = perineural_status or {}

        self.titan_model = None
        self.titan_tokenizer = None
        self.text_embedding_cache = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _initialize_encoder(self):
        if self.titan_model is None:
            print(f"Process {os.getpid()}: Initializing TITAN model...")
            self.titan_model = AutoModel.from_pretrained(
                'MahmoodLab/TITAN', trust_remote_code=True
            ).to(self.device)
            self.titan_model.eval()
            self.titan_tokenizer = self.titan_model.text_encoder.tokenizer

    def _reconstruct_text(self, metadata: Dict) -> str:
        return (
            f"Tumor Location：{metadata.get('Tumor Location', '未知')}，"
            f"Gross Type：{metadata.get('Gross Type', '未知')}，"
            f"Histological Type：{metadata.get('Histological Type', '未知')}，"
            f"Invasion Depth：{metadata.get('Invasion Depth', '未知')}"
        )

    def _encode_text(self, text: str) -> torch.Tensor:
        self._initialize_encoder()
        
        if text in self.text_embedding_cache:
            return self.text_embedding_cache[text]

        tokenized = self.titan_tokenizer([text]).to(self.device)
        with torch.no_grad():
            embedding = self.titan_model.encode_text(tokenized, normalize=True).squeeze(0).cpu()
        
        self.text_embedding_cache[text] = embedding
        return embedding

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict:
        h5_path = self.file_paths[idx]
        
        with h5py.File(h5_path, 'r') as h5f:
            image_feat = torch.from_numpy(np.array(h5f['tensor']))
            metadata = {k: v for k, v in h5f['pathReport'].attrs.items()}

        fname = os.path.basename(h5_path)
        base_name = os.path.splitext(fname)[0].upper()
        pid_match = re.search(r'(\d+)', base_name)
        pid = pid_match.group(1) if pid_match else base_name
        
        # main task
        label = 1 if 'A' in base_name else 0

        reconstructed_text = self._reconstruct_text(metadata)
        # if miss modality
        missing_modality = 0
        text_feat = None
        if random.random() < self.missing_probs.get(label, 0.0):
            missing_modality = 1
            text_feat = torch.zeros(768)
        else:
            text_feat = self._encode_text(reconstructed_text)

        # reconstruct
        sample = {
            'image': image_feat,
            'text_feat': text_feat.unsqueeze(0),
            'missing_modality': torch.tensor(missing_modality, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'original_filename': fname,
            'pathology_id': pid,
            'lymph_node_label': self.lymph_node_status.get(pid, -1),
            'vascular_thrombus': self.vascular_status.get(pid, -1),
            'perineural_invasion': self.perineural_status.get(pid, -1),
        }
        return sample

def create_folds(dataset: PathologyMultimodalDataset, n_folds=5, train_ratio=0.7, base_dir='folds'):
    os.makedirs(base_dir, exist_ok=True)
    n_samples = len(dataset)
    indices = np.arange(n_samples)

    for fold in range(n_folds):
        np.random.shuffle(indices)
        split_point = int(train_ratio * n_samples)
        train_idx, val_idx = indices[:split_point], indices[split_point:]
        
        records = []
        from tqdm import tqdm
        for phase, phase_indices in [('train', train_idx), ('val', val_idx)]:
            for i in tqdm(phase_indices, desc=f"Creating fold {fold} {phase} set"):
                sample_info = dataset[i]
                rec = {
                    'original_filename': sample_info['original_filename'],
                    'pathology_id': sample_info['pathology_id'],
                    'label': int(sample_info['label'].item()),
                    'set': phase,
                    'missing_modality': int(sample_info['missing_modality'].item()),
                    'lymph_node_label': int(sample_info['lymph_node_label']),
                    'vascular_thrombus': int(sample_info['vascular_thrombus']),
                    'perineural_invasion': int(sample_info['perineural_invasion']),
                }
                records.append(rec)
        
        df = pd.DataFrame(records)
        fold_dir = os.path.join(base_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        csv_path = os.path.join(fold_dir, 'split.csv')
        df.to_csv(csv_path, index=False)
        print(f"Fold {fold} split file created at: {csv_path}")


if __name__ == "__main__":
    CSV_LABEL_PATH = "/home/aletolia/documents/code_notes/visualization/MMP/crossTest/pathReport_modified_cleaned.csv"
    H5_DATA_DIR = "/home/aletolia/documents/code_notes/visualization/MMP/crossTest/output_conch2"
    FOLDS_DIR = "folds_new"

    ln_labels, va_labels, pe_labels = load_additional_labels_from_csv(CSV_LABEL_PATH)

    dataset = PathologyMultimodalDataset(
        data_dir=H5_DATA_DIR,
        lymph_node_status=ln_labels,
        vascular_status=va_labels,
        perineural_status=pe_labels,
        missing_probs={0: 0.0, 1: 0.0} 
    )
    print(f"Dataset initialized with {len(dataset)} samples found.")
    print("Note: Data is not loaded into memory yet. It will be loaded on-the-fly.")

    if dataset:
        create_folds(dataset, n_folds=5, train_ratio=0.7, base_dir=FOLDS_DIR)
        print(f"\nSuccessfully created cross-validation splits in '{FOLDS_DIR}' directory.")