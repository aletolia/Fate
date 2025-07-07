import os
import re
import random
import math
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
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

class H5TextEncoder:
    def __init__(self,
                 missing_probs=None,
                 lymph_node_status=None,
                 vascular_status=None,
                 perineural_status=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.titan = AutoModel.from_pretrained(
            'MahmoodLab/TITAN', trust_remote_code=True
        ).to(self.device)
        self.titan.eval()

        self.missing_probs     = missing_probs    or {}
        self.lymph_node_status = lymph_node_status or {}
        self.vascular_status   = vascular_status   or {}
        self.perineural_status = perineural_status or {}

    def _reconstruct_text(self, metadata):
        return (
            f"Tumor Location：{metadata.get('Tumor Location', '未知')}，"
            f"Gross Type：{metadata.get('Gross Type', '未知')}，"
            f"Tumor Size：{metadata.get('Tumor Size', '未知')}。"
            f"Histological Type：{metadata.get('Histological Type', '未知')}，"
            f"Histological Grade：{metadata.get('Histological Grade', '未知')}，"
            f"Invasion Depth：{metadata.get('Invasion Depth', '未知')}，"
            f"Lauren Classification：{metadata.get('Lauren Classification', '未知')}"
        )

    def _encode_text(self, text):
        tokenized = self.titan.text_encoder.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.titan.encode_text(tokenized, normalize=True)
        return emb[0].cpu()

    def process_h5(self, h5_path):
        with h5py.File(h5_path, 'r') as h5f:
            image_feat = torch.from_numpy(np.array(h5f['tensor']))
            meta_grp   = h5f['pathReport']
            metadata   = {k: v for k, v in meta_grp.attrs.items()}
            
        text = self._reconstruct_text(metadata)
        fname = os.path.basename(h5_path)
        base  = os.path.splitext(fname)[0].upper()
        m = re.search(r'(\d+)', base)
        pid = m.group(1) if m else base
        label = 1 if 'A' in base else 0

        ln_label = self.lymph_node_status.get(pid, None)
        va_label = self.vascular_status.get(pid, None)
        pe_label = self.perineural_status.get(pid, None)

        mp = self.missing_probs.get(label, 0.0)
        if random.random() < mp:
            text_feat = torch.zeros(768)
            missing_modality = 1
        else:
            text_feat = self._encode_text(text)
            missing_modality = 0

        result = {
            'image': image_feat,
            'text_feat': text_feat,
            'metadata': metadata,
            'reconstructed_text': text,
            'file_path': h5_path,
            'label': torch.tensor(label, dtype=torch.long),
            'missing_modality': torch.tensor(missing_modality, dtype=torch.long),
            'pathology_id': pid,
            'original_filename': fname,
        }
        if ln_label is not None:
            result['lymph_node_label'] = torch.tensor(ln_label, dtype=torch.long)
        if va_label is not None:
            result['vascular_thrombus'] = torch.tensor(va_label, dtype=torch.long)
        if pe_label is not None:
            result['perineural_invasion'] = torch.tensor(pe_label, dtype=torch.long)

        return result

class WSIPathDataset:
    def __init__(self, folder_path, encoder=None):
        self.encoder = encoder or H5TextEncoder()
        self.data = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.endswith('.h5'):
                    try:
                        self.data.append(self.encoder.process_h5(os.path.join(root, f)))
                    except Exception as e:
                        print(f"Error processing {f}: {e}")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

    def get_labels(self):
        return [d['label'] for d in self.data]
    
    def get_image_features(self):
        return [d['image'] for d in self.data]
    
    def get_text_embeddings(self):
        return [d['text_feat'] for d in self.data]
    
    def get_metadata(self):
        return [d['metadata'] for d in self.data]
    
    def get_missing_flags(self):
        return [d['missing_modality'] for d in self.data]
    
    def get_lymph_node_labels(self):
        return [d['lymph_node_label'] for d in self.data if 'lymph_node_label' in d]
    
    def get_vascular_labels(self):
        return [d['vascular_thrombus'] for d in self.data if 'vascular_thrombus' in d]
    
    def get_perineural_labels(self):
        return [d['perineural_invasion'] for d in self.data if 'perineural_invasion' in d]

def create_folds(dataset, n_folds=5, train_ratio=0.7, base_dir='folds'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    N = len(dataset)
    idxs = np.arange(N)

    for fold in range(n_folds):
        np.random.shuffle(idxs)
        split = int(train_ratio * N)
        train_idx, val_idx = idxs[:split], idxs[split:]
        recs = []
        for phase, inds in [('train', train_idx), ('val', val_idx)]:
            for i in inds:
                d = dataset.data[i]
                rec = {
                    'original_filename': d['original_filename'],
                    'pathology_id': d['pathology_id'],
                    'label': int(d['label'].item()),
                    'set': phase,
                    'missing_modality': int(d['missing_modality'].item()),
                    'lymph_node_label': int(d.get('lymph_node_label', -1).item()) if 'lymph_node_label' in d else -1,
                    'vascular_thrombus': int(d.get('vascular_thrombus', -1).item()) if 'vascular_thrombus' in d else -1,
                    'perineural_invasion': int(d.get('perineural_invasion', -1).item()) if 'perineural_invasion' in d else -1,
                }
                recs.append(rec)
        df = pd.DataFrame(recs)
        odir = os.path.join(base_dir, f'fold_{fold}')
        os.makedirs(odir, exist_ok=True)
        csv_p = os.path.join(odir, 'split.csv')
        df.to_csv(csv_p, index=False)
        print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)} -> {csv_p}")

if __name__ == "__main__":
    csv_path = "/home/aletolia/documents/code_notes/visualization/MMP/crossTest/pathReport_modified_cleaned.csv"
    ln, va, pe = load_additional_labels_from_csv(csv_path)

    encoder = H5TextEncoder(
        missing_probs={0:0.0, 1:0.0},
        lymph_node_status=ln,
        vascular_status=va,
        perineural_status=pe,
    )
    dataset = WSIPathDataset("/home/aletolia/documents/code_notes/visualization/MMP/crossTest/output_conch2", encoder=encoder)
    print(f"Loaded {len(dataset)} samples")

    create_folds(dataset, n_folds=5, train_ratio=0.7, base_dir='folds')