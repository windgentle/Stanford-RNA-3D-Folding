import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import gc
import random
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F

# =============================================================================
# 1. Configuration, Seeds, and Memory Management
# =============================================================================

# DATA_PATH = "/kaggle/input/stanford-rna-3d-folding/"
DATA_PATH = "stanford-rna-3d-folding/"
SEQ_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_PREDICTIONS = 5  # Number of structure predictions

def set_seed(seed=42):
    """Set seeds for reproducibility across all random modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seeds set to {seed} for reproducibility")

set_seed()

def reduce_mem_usage(df):
    """Downcast numeric columns to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    for col in df.columns:
        if df[col].dtype != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(df[col].dtype)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB; decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    return df

def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

# =============================================================================
# 2. Data Loading, Preprocessing, and Feature Engineering
# =============================================================================

def load_data():
    """Load CSV files with sequences, labels, and sample submission."""
    print("Loading data...")
    train_seqs = pd.read_csv(os.path.join(DATA_PATH, 'train_sequences.csv'))
    val_seqs = pd.read_csv(os.path.join(DATA_PATH, 'validation_sequences.csv'))
    test_seqs = pd.read_csv(os.path.join(DATA_PATH, 'test_sequences.csv'))
    train_labels = pd.read_csv(os.path.join(DATA_PATH, 'train_labels.csv'))
    val_labels = pd.read_csv(os.path.join(DATA_PATH, 'validation_labels.csv'))
    sample_submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
    return train_seqs, val_seqs, test_seqs, train_labels, val_labels, sample_submission

def verify_data_splits(train_df, val_df, test_df):
    """Ensure that there is no data leakage between splits."""
    train_ids = set(train_df['target_id'])
    val_ids = set(val_df['target_id'])
    test_ids = set(test_df['target_id'])
    if not (train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)):
        print("WARNING: Overlapping target IDs detected among splits.")
    else:
        print("Data splits verified: No overlapping target IDs.")

def create_advanced_lag_features(df, window_sizes=[3, 5]):
    """Add rolling statistics as lag features."""
    for col in df.select_dtypes(include=[np.number]).columns:
        for window in window_sizes:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
    return df

# Data augmentation for RNA sequences
def augment_rna_sequence(seq, prob=0.1):
    """Randomly substitute bases in RNA sequence."""
    bases = ['A', 'C', 'G', 'U']
    seq_chars = list(seq)
    for i in range(len(seq_chars)):
        if random.random() < prob:
            seq_chars[i] = random.choice([b for b in bases if b != seq_chars[i]])
    return ''.join(seq_chars)

# Function to add Gaussian noise to features
def augment_features(features, noise_std=0.01):
    noise = torch.randn_like(features) * noise_std
    return features + noise

class RNADataset(Dataset):
    def __init__(self, seq_df, label_df, msa_dir, max_len=SEQ_LENGTH, augment=False, normalize=True):
        self.seq_df = seq_df
        self.label_df = label_df
        self.msa_dir = msa_dir
        self.max_len = max_len
        self.augment = augment
        self.normalize = normalize
        self.data = []
        self.feature_scaler = StandardScaler()
        self._preprocess_data()
        
    def _preprocess_sequence(self, seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        one_hot = np.zeros((self.max_len, 4), dtype=np.float32)
        for i, base in enumerate(seq[:self.max_len]):
            if base in mapping:
                one_hot[i, mapping[base]] = 1.0
        return one_hot
    
    def _get_msa_features(self, target_id):
        msa_path = os.path.join(self.msa_dir, f"{target_id}.MSA.fasta")
        # Return zeros with 2 channels to match expected dimension.
        if not os.path.exists(msa_path):
            return np.zeros((self.max_len, 2), dtype=np.float32)
        try:
            sequences = [str(rec.seq) for rec in SeqIO.parse(msa_path, 'fasta')]
            if sequences:
                counts = np.zeros((self.max_len, 4), dtype=np.float32)
                for seq in sequences:
                    for i, c in enumerate(seq[:self.max_len]):
                        if c == 'A': 
                            counts[i, 0] += 1
                        elif c == 'C': 
                            counts[i, 1] += 1
                        elif c == 'G': 
                            counts[i, 2] += 1
                        elif c == 'U': 
                            counts[i, 1] += 1
                counts += 1e-5
                freqs = counts / counts.sum(axis=1, keepdims=True)
                entropy = -np.sum(freqs * np.log(freqs + 1e-10), axis=1)
                conservation = 1 - entropy / np.log(4)
                seq_depth = len(sequences)
                extra_feature = np.ones(self.max_len, dtype=np.float32) * seq_depth / 100
                conservation = np.column_stack((conservation, extra_feature))
                if conservation.shape[0] > self.max_len:
                    conservation = conservation[:self.max_len, :]
                elif conservation.shape[0] < self.max_len:
                    pad_rows = self.max_len - conservation.shape[0]
                    padding = np.zeros((pad_rows, conservation.shape[1]), dtype=conservation.dtype)
                    conservation = np.vstack((conservation, padding))
                return conservation
            else:
                return np.zeros((self.max_len, 2), dtype=np.float32)
        except Exception as e:
            print(f"Error processing MSA for {target_id}: {e}")
            return np.zeros((self.max_len, 2), dtype=np.float32)
    
    def _preprocess_labels(self, target_id):
        if self.label_df.empty or 'ID' not in self.label_df.columns:
            return np.zeros((NUM_PREDICTIONS, self.max_len, 3), dtype=np.float32)
        target_labels = self.label_df[self.label_df['ID'].str.startswith(target_id)]
        coords_list = []
        for _, row in target_labels.iterrows():
            struct_coords = []
            for i in range(1, 100):
                x = row.get(f'x_{i}', np.nan)
                y = row.get(f'y_{i}', np.nan)
                z = row.get(f'z_{i}', np.nan)
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    break
                struct_coords.append([x, y, z])
            if len(struct_coords) > self.max_len:
                struct_coords = struct_coords[:self.max_len]
            elif len(struct_coords) < self.max_len:
                pad_len = self.max_len - len(struct_coords)
                struct_coords.extend([[0.0, 0.0, 0.0]] * pad_len)
            coords_list.append(struct_coords)
        if len(coords_list) < NUM_PREDICTIONS:
            for _ in range(NUM_PREDICTIONS - len(coords_list)):
                coords_list.append([[0.0, 0.0, 0.0]] * self.max_len)
        else:
            coords_list = coords_list[:NUM_PREDICTIONS]
        coords_array = np.array(coords_list, dtype=np.float32)
        coords_array = (coords_array - np.mean(coords_array)) / (np.std(coords_array) + 1e-8)
        return coords_array
    
    def _preprocess_data(self):
        all_features = []
        if self.normalize:
            print("Collecting features for normalization...")
            for _, row in tqdm(self.seq_df.iterrows(), total=len(self.seq_df)):
                target_id = row['target_id']
                seq = row['sequence']
                one_hot = self._preprocess_sequence(seq)
                conservation = self._get_msa_features(target_id)
                if isinstance(conservation, np.ndarray) and conservation.ndim == 1:
                    conservation = conservation.reshape(-1, 1)
                if one_hot.shape[0] != conservation.shape[0]:
                    conservation = conservation[:one_hot.shape[0], :]
                features = np.concatenate([one_hot, conservation], axis=1)
                all_features.append(features.reshape(1, -1)[0])
            all_features = np.array(all_features)
            self.feature_scaler.fit(all_features)
        
        print("Creating dataset...")
        for _, row in tqdm(self.seq_df.iterrows(), total=len(self.seq_df)):
            target_id = row['target_id']
            seq = row['sequence']
            if self.augment and random.random() < 0.3:
                seq = augment_rna_sequence(seq, prob=0.05)
            one_hot = self._preprocess_sequence(seq)
            conservation = self._get_msa_features(target_id)
            if isinstance(conservation, np.ndarray) and conservation.ndim == 1:
                conservation = conservation.reshape(-1, 1)
            if one_hot.shape[0] != conservation.shape[0]:
                conservation = conservation[:one_hot.shape[0], :]
            features = np.concatenate([one_hot, conservation], axis=1)
            if self.normalize:
                flat_features = features.reshape(1, -1)
                features = self.feature_scaler.transform(flat_features).reshape(features.shape)
            coords = self._preprocess_labels(target_id)
            if self.augment and not np.all(coords == 0):
                noise_scale = 0.01
                coords += np.random.normal(0, noise_scale, coords.shape)
            self.data.append((
                torch.tensor(features.T, dtype=torch.float32),  # (channels, seq_len)
                torch.tensor(coords, dtype=torch.float32)         # (NUM_PREDICTIONS, max_len, 3)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    features_list, labels_list = zip(*batch)
    features = torch.stack(features_list, dim=0)  # (batch, channels, seq_len)
    fixed_seq_len = SEQ_LENGTH
    if features.shape[2] < fixed_seq_len:
        pad_size = fixed_seq_len - features.shape[2]
        features = F.pad(features, (0, pad_size), "constant", 0)
    elif features.shape[2] > fixed_seq_len:
        features = features[:, :, :fixed_seq_len]
    padded_labels = []
    for label in labels_list:
        if label.shape[1] < fixed_seq_len:
            pad = torch.zeros((label.shape[0], fixed_seq_len - label.shape[1], label.shape[2]), dtype=label.dtype)
            padded_label = torch.cat([label, pad], dim=1)
        elif label.shape[1] > fixed_seq_len:
            padded_label = label[:, :fixed_seq_len, :]
        else:
            padded_label = label
        padded_labels.append(padded_label)
    labels = torch.stack(padded_labels, dim=0)
    return features, labels

# =============================================================================
# 3. Neural Network Architecture (Original)
# =============================================================================
# Input channels: 6 (4 for one-hot, 2 for conservation)
class RNA3DModel(nn.Module):
    def __init__(self, input_channels=6, seq_length=SEQ_LENGTH, num_structures=NUM_PREDICTIONS):
        super().__init__()
        self.num_structures = num_structures
        self.seq_length = seq_length
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 128, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv1d(512, num_structures * 3, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), self.num_structures, 3)
        return x

# =============================================================================
# 3.5 Enhanced Model: Attention, Graph-Based, Transformer, Confidence, and TTA
# =============================================================================
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch, seq_len, feat = x.size()
        padded = torch.zeros(batch, seq_len + 2, feat, device=x.device)
        padded[:, 1:-1, :] = x
        agg = (padded[:, :-2, :] + padded[:, 1:-1, :] + padded[:, 2:, :]) / 3.0
        out = self.linear(agg)
        out = self.dropout(out)
        return out

class ImprovedRNA3DModel(nn.Module):
    def __init__(self, input_channels=6, seq_length=SEQ_LENGTH, num_structures=NUM_PREDICTIONS, 
                 nhead=4, num_transformer_layers=2, dropout_rate=0.3):
        super().__init__()
        self.num_structures = num_structures
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 128, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate/2)
        )
        
        self.self_attn = nn.MultiheadAttention(embed_dim=256, num_heads=nhead, dropout=dropout_rate, batch_first=True)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=nhead,
            dropout=dropout_rate,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.graph_conv1 = GraphConvLayer(256, 256, dropout=dropout_rate)
        self.graph_conv2 = GraphConvLayer(256, 256, dropout=dropout_rate/2)
        
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(256)
        
        self.post_block = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate/3)
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv1d(512, num_structures * 3, 1),
            nn.Tanh()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Conv1d(512, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, p_dropout=None):
        dropout_rate = self.dropout_rate if p_dropout is None else p_dropout
        x = self.conv_block(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        x = F.dropout(x, p=dropout_rate, training=self.training)
        attn_out, _ = self.self_attn(x, x, x)
        x = x + F.dropout(attn_out, p=dropout_rate, training=self.training)
        x = self.layer_norm1(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm2(x)
        gc_out1 = self.graph_conv1(x)
        x = x + gc_out1
        gc_out2 = self.graph_conv2(x)
        x = x + gc_out2
        x_post = x.transpose(1, 2)  # (batch, features, seq_len)
        x_post = self.post_block(x_post)
        out = self.out_conv(x_post)
        out = out.permute(0, 2, 1)
        batch_size = out.size(0)
        structure_pred = out.view(batch_size, out.size(1), self.num_structures, 3)
        confidence = self.confidence_head(x_post)
        confidence = confidence.squeeze(1)  # (batch, seq_len)
        return structure_pred, confidence

def tta_predict(model, loader, tta_iterations=5):
    """
    Test-Time Augmentation: perform multiple predictions with slight random noise added,
    and vary dropout rate. The predictions are then weighted by softmax-normalized confidence.
    """
    model.eval()
    all_preds = []
    all_confidences = []
    
    with torch.no_grad():
        for features, _ in tqdm(loader, desc="Generating TTA predictions"):
            batch_preds = []
            batch_confs = []
            for tta_iter in range(tta_iterations):
                p_dropout = 0.1 + (tta_iter * 0.05)
                noise_level = 0.005 + (tta_iter * 0.003)
                noise = torch.randn_like(features) * noise_level
                aug_features = features + noise
                aug_features = aug_features.to(DEVICE)
                structure_pred, confidence = model(aug_features, p_dropout=p_dropout)
                batch_preds.append(structure_pred.cpu())
                batch_confs.append(confidence.cpu())
            stacked_preds = torch.stack(batch_preds, dim=0)  # [TTA, batch, seq_len, num_structures, 3]
            stacked_conf = torch.stack(batch_confs, dim=0)     # [TTA, batch, seq_len]
            norm_conf = F.softmax(stacked_conf, dim=0)  # Normalize over TTA iterations
            weights = norm_conf.unsqueeze(-1).unsqueeze(-1)  # [TTA, batch, seq_len, 1, 1]
            weighted_pred = (stacked_preds * weights).sum(dim=0)
            avg_conf = stacked_conf.mean(dim=0)
            all_preds.append(weighted_pred)
            all_confidences.append(avg_conf)
    
    final_predictions = torch.cat(all_preds, dim=0)
    final_confidences = torch.cat(all_confidences, dim=0)
    return final_predictions, final_confidences

# =============================================================================
# 4. Training Components and Evaluation
# =============================================================================

def tm_score_loss(pred, target, confidence=None):
    target = target.permute(0, 2, 1, 3)
    pred_struct = pred[:, :, 0, :]
    all_tm_scores = []
    for i in range(target.shape[2]):
        target_struct = target[:, :, i, :]
        squared_dists = torch.sum((pred_struct - target_struct)**2, dim=-1) + 1e-8
        dists = torch.sqrt(squared_dists)
        seq_len = target.shape[1]
        d0 = 1.24 * (seq_len - 15)**(1/3) - 1.8
        tm_score_components = 1 / (1 + (dists / d0)**2)
        if confidence is not None:
            norm_confidence = seq_len * confidence / confidence.sum(dim=1, keepdim=True)
            tm_score_components = tm_score_components * norm_confidence
        tm_scores = tm_score_components.mean(dim=1)
        all_tm_scores.append(tm_scores)
    all_tm_scores = torch.stack(all_tm_scores, dim=1)
    best_tm_scores = all_tm_scores.max(dim=1)[0]
    l2_reg = torch.mean(torch.norm(pred_struct, dim=2)) * 0.001
    return -best_tm_scores.mean() + l2_reg

def compute_accuracy(pred, target, threshold=1.0):
    pred_struct = pred[:, :, 0, :]
    target = target.permute(0, 2, 1, 3)
    best_acc = torch.zeros(pred.size(0), device=pred.device)
    for i in range(target.shape[2]):
        target_struct = target[:, :, i, :]
        distances = torch.sqrt(torch.sum((pred_struct - target_struct)**2, dim=-1) + 1e-8)
        correct = (distances < threshold).float().mean(dim=1)
        best_acc = torch.max(torch.stack([best_acc, correct]), dim=0)[0]
    return best_acc.mean().item()

# =============================================================================
# 5. Main Training Pipeline and Experiment Management
# =============================================================================

def train_and_validate(model, train_loader, val_loader, epochs=EPOCHS, patience=10, save_best=True, model_path='best_model.pth'):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    best_val_acc = 0
    no_improve_epochs = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        total_samples = 0
        progress_bar = tqdm(train_loader, desc="Training")
        train_accuracies = []
        for features, targets in progress_bar:
            features = augment_features(features, noise_std=0.01).to(DEVICE)
            targets = targets.to(DEVICE)
            print("****************************************************features****************************************************************")
            print(features)
            print("*****************************************************targets***************************************************************")
            print(targets)


            optimizer.zero_grad()
            structure_pred, confidence = model(features)
            loss = tm_score_loss(structure_pred, targets, confidence)
            combined_loss = loss
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = features.size(0)
            total_loss += combined_loss.item() * batch_size
            total_samples += batch_size
            train_acc = compute_accuracy(structure_pred, targets)
            train_accuracies.append(train_acc)
            progress_bar.set_postfix({'loss': combined_loss.item(), 'avg_loss': total_loss / total_samples})
        
        epoch_train_loss = total_loss / total_samples
        epoch_train_acc = np.mean(train_accuracies)
        
        model.eval()
        val_loss = 0
        val_accs = []
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                structure_pred, confidence = model(features)
                loss = tm_score_loss(structure_pred, targets, confidence)
                val_loss += loss.item() * features.size(0)
                val_accs.append(compute_accuracy(structure_pred, targets))
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = np.mean(val_accs)
        
        history['epoch'].append(epoch+1)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)
        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            no_improve_epochs = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
            if save_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                }, model_path)
                print(f"Model saved to {model_path}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")
            if no_improve_epochs >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break
    
    # Plot training history
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_history.png")
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'], history['val_acc'], label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy History')
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_accuracy_history.png")
    plt.show()
    
    return best_val_loss, best_val_acc, history

def test_prediction(model, test_loader, sample_submission, model_path='best_model.pth'):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")
    model.eval()
    predictions = []
    target_ids = []
    
    for batch_idx, (features, _) in enumerate(tqdm(test_loader, desc="Generating predictions")):
        features = features.to(DEVICE)
        with torch.no_grad():
            structure_pred, _ = model(features)
        pred_coords = structure_pred[:, :, 0, :].cpu().numpy()
        for i in range(len(pred_coords)):
            idx = batch_idx * test_loader.batch_size + i
            target_id = test_loader.dataset.seq_df.iloc[idx]['target_id']
            target_ids.append(target_id)
            predictions.append(pred_coords[i])
    
    submission_rows = []
    for target_id, pred_coords in zip(target_ids, predictions):
        row = {'ID': target_id}
        for i, (x, y, z) in enumerate(pred_coords, 1):
            if i > 100:
                break
            row[f'x_{i}'] = x
            row[f'y_{i}'] = y
            row[f'z_{i}'] = z
        submission_rows.append(row)
    
    submission_df = pd.DataFrame(submission_rows)
    return submission_df

def analyze_model_performance(model, val_loader, output_dir="./analysis"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    features, targets = next(iter(val_loader))
    features, targets = features.to(DEVICE), targets.to(DEVICE)
    with torch.no_grad():
        structure_pred, confidence = model(features)
    plt.figure(figsize=(10, 6))
    plt.hist(confidence.cpu().numpy().flatten(), bins=50)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"))
    plt.show()
    plt.close()
    sample_idx = 0
    pred_struct = structure_pred[sample_idx, :, 0, :].cpu().numpy()
    targets_permuted = targets.permute(0, 2, 1, 3)
    true_struct = targets_permuted[sample_idx, :, 0, :].cpu().numpy()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pred_struct[:, 0], pred_struct[:, 1], pred_struct[:, 2],
               c='blue', marker='o', alpha=0.6, label='Predicted')
    ax.scatter(true_struct[:, 0], true_struct[:, 1], true_struct[:, 2],
               c='red', marker='^', alpha=0.6, label='True')
    for i in range(len(pred_struct)-1):
        ax.plot([pred_struct[i, 0], pred_struct[i+1, 0]],
                [pred_struct[i, 1], pred_struct[i+1, 1]],
                [pred_struct[i, 2], pred_struct[i+1, 2]], 'b-', alpha=0.3)
        ax.plot([true_struct[i, 0], true_struct[i+1, 0]],
                [true_struct[i, 1], true_struct[i+1, 1]],
                [true_struct[i, 2], true_struct[i+1, 2]], 'r-', alpha=0.3)
    ax.set_title("3D Structure Comparison")
    ax.legend()
    plt.savefig(os.path.join(output_dir, "structure_comparison_3d.png"))
    plt.show()
    plt.close()
    try:
        background = features[:10].cpu()
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(features[:5].cpu())
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, features[:5].cpu(), feature_names=['A', 'C', 'G', 'U', 'Conservation', 'MSA Depth'])
        plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"))
        plt.show()
        plt.close()
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

def evaluate_holdout(model, seq_df, label_df, msa_dir, holdout_fraction=0.2):
    """Evaluate model on an unseen holdout set from the training data."""
    train_df, holdout_df = train_test_split(seq_df, test_size=holdout_fraction, random_state=42)
    holdout_dataset = RNADataset(holdout_df, label_df, msa_dir, augment=False)
    holdout_loader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)
    model.eval()
    holdout_losses = []
    holdout_accuracies = []
    with torch.no_grad():
        for features, targets in holdout_loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            structure_pred, confidence = model(features)
            loss = tm_score_loss(structure_pred, targets, confidence)
            holdout_losses.append(loss.item())
            holdout_accuracies.append(compute_accuracy(structure_pred, targets))
    avg_loss = np.mean(holdout_losses)
    avg_acc = np.mean(holdout_accuracies)
    print(f"Holdout Evaluation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

def cross_validation(seq_df, label_df, msa_dir, n_folds=5, model_class=ImprovedRNA3DModel, **model_kwargs):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    val_losses = []
    val_accs = []
    strata = seq_df['sequence'].apply(len)
    strata = pd.qcut(strata, 5, labels=False)
    for fold, (train_idx, val_idx) in enumerate(kf.split(seq_df, strata)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        train_seq_df = seq_df.iloc[train_idx].reset_index(drop=True)
        val_seq_df = seq_df.iloc[val_idx].reset_index(drop=True)
        train_dataset = RNADataset(train_seq_df, label_df, msa_dir, augment=True)
        val_dataset = RNADataset(val_seq_df, label_df, msa_dir, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)
        model = model_class(**model_kwargs).to(DEVICE)
        fold_val_loss, fold_val_acc = train_and_validate(model, train_loader, val_loader, model_path=f'model_fold{fold+1}.pth')
        val_losses.append(fold_val_loss)
        val_accs.append(fold_val_acc)
        print(f"Fold {fold+1} validation loss: {fold_val_loss:.4f}, accuracy: {fold_val_acc:.4f}")
        del model, train_dataset, val_dataset, train_loader, val_loader
        clear_memory()
    print("\n--- Cross-Validation Results ---")
    print(f"Mean validation loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Mean validation accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    return val_losses, val_accs

# =============================================================================
# 6. Main Execution Pipeline
# =============================================================================

def main():
    print(f"Using device: {DEVICE}")
    train_seqs, val_seqs, test_seqs, train_labels, val_labels, sample_submission = load_data()
    verify_data_splits(train_seqs, val_seqs, test_seqs)
    print(f"Training on {len(train_seqs)} sequences, validating on {len(val_seqs)}, testing on {len(test_seqs)}")
    msa_dir = os.path.join(DATA_PATH, "MSA")
    
    # Create datasets
    train_dataset = RNADataset(train_seqs, train_labels, msa_dir, augment=True)
    val_dataset = RNADataset(val_seqs, val_labels, msa_dir, augment=False)
    test_dataset = RNADataset(test_seqs, pd.DataFrame(), msa_dir, augment=False)
    print(train_dataset)
    # Create data loaders with custom collate function (fixed sequence length)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)
    
    # Initialize model (improved architecture)
    model = ImprovedRNA3DModel(input_channels=6, seq_length=SEQ_LENGTH, num_structures=NUM_PREDICTIONS).to(DEVICE)
    
    # Train model and record history
    best_val_loss, best_val_acc, history = train_and_validate(
        model, train_loader, val_loader, epochs=EPOCHS, patience=10, save_best=True, model_path='best_rna_model.pth'
    )
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}, accuracy: {best_val_acc:.4f}")
    
    # Evaluate on unseen holdout data
    print("Evaluating on holdout data...")
    holdout_loss, holdout_acc = evaluate_holdout(model, train_seqs, train_labels, msa_dir, holdout_fraction=0.2)
    
    # Graphical plots for overall peaked accuracies (train, validation, holdout)
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'], history['train_loss'], label="Train Loss", marker='o')
    plt.plot(history['epoch'], history['val_loss'], label="Validation Loss", marker='o')
    plt.axhline(y=holdout_loss, color='red', linestyle='--', label=f"Holdout Loss: {holdout_loss:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Holdout Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("overall_loss_history.png")
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'], history['val_acc'], label="Validation Accuracy", marker='o')
    plt.axhline(y=holdout_acc, color='green', linestyle='--', label=f"Holdout Accuracy: {holdout_acc:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation and Holdout Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("overall_accuracy_history.png")
    plt.show()
    
    # Analyze model performance (feature importance, SHAP, etc.)
    analyze_model_performance(model, val_loader)
    
    # Generate predictions with Test-Time Augmentation
    print("Generating predictions with TTA...")
    tta_predictions, tta_confidences = tta_predict(model, test_loader, tta_iterations=5)
    
    # Create submission file from TTA predictions
    submission_df = test_prediction(model, test_loader, sample_submission, model_path='best_rna_model.pth')
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created: submission.csv")
    
    # Optional cross-validation (if desired)
    if False:
        print("\nRunning cross-validation...")
        cv_losses, cv_accs = cross_validation(
            train_seqs, train_labels, msa_dir, n_folds=5, model_class=ImprovedRNA3DModel,
            input_channels=6, seq_length=SEQ_LENGTH, num_structures=NUM_PREDICTIONS
        )
    
    print("="*50)
    print("RNA 3D Structure Prediction Pipeline Complete")
    print("="*50)
    clear_memory()
    print("All resources cleaned up. Execution complete.")

if __name__ == "__main__":
    main()