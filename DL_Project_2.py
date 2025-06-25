import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 고정 시드
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset 정의
class TrajectoryDataset(Dataset):
    def __init__(self, X_df, y_df=None, max_len=60):
        self.vehicle_ids = X_df['vehicle_id'].unique()
        self.X_group = X_df.groupby('vehicle_id')
        self.y_dict = dict(zip(y_df['vehicle_id'], y_df['change_section'])) if y_df is not None else None
        self.max_len = max_len

    def __len__(self):
        return len(self.vehicle_ids)

    def __getitem__(self, idx):
        vid = self.vehicle_ids[idx]
        data = self.X_group.get_group(vid).sort_values('time_step')[['position_y', 'speed', 'acceleration']].values
        pad = np.zeros((self.max_len, 3))
        pad[:len(data)] = data
        X = torch.tensor(pad, dtype=torch.float)
        if self.y_dict:
            y = torch.tensor(self.y_dict[vid], dtype=torch.long)
            return X, y
        else:
            return X, vid

# TCN 블록 정의 (Residual 연결 및 크기 조정)
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.residual is not None:
            res = self.residual(x)
        else:
            res = x

        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]
        return out + res

# TCN 정의
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_c = num_inputs if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            layers.append(
                TemporalBlock(in_c, out_c, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        return out.transpose(1, 2)

# Attention 메커니즘 정의
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn_weights = self.softmax(self.attn(x))
        return torch.sum(x * attn_weights, dim=1)

# 모델 정의 (TCN + BiGRU + Attention)
class HybridModel(nn.Module):
    def __init__(self, input_dim=3, tcn_channels=[64, 128, 256], gru_hidden=128, num_classes=4):
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, tcn_channels, kernel_size=3, dropout=0.2)
        self.bigru = nn.GRU(tcn_channels[-1], gru_hidden, batch_first=True, bidirectional=True)
        self.attention = Attention(gru_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.tcn(x)
        x, _ = self.bigru(x)
        x = self.attention(x)
        return self.classifier(x)

# 평가 함수
def evaluate_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("📊 Evaluation Metrics:")
    print(f" - Accuracy : {acc:.4f}")
    print(f" - Precision: {prec:.4f}")
    print(f" - Recall   : {rec:.4f}")
    print(f" - F1-score : {f1:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}

# 학습 루프
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
        val_f1 = evaluate_on_validation(model, val_loader)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "model.pt")
            print("✅ Best model saved.")

# Validation 평가
def evaluate_on_validation(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            out = model(X)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    result = evaluate_classification(y_true, y_pred)
    return result["f1_score"]

# Test 예측 및 제출
def predict_and_submit(model, test_loader, test_X, output_path="submission.csv"):
    model.eval()
    vehicle_ids, predictions = [], []
    with torch.no_grad():
        for X, vids in test_loader:
            X = X.to(DEVICE)
            out = model(X)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            predictions.extend(preds)
            vehicle_ids.extend(vids)

    test_vehicle_ids = test_X['vehicle_id'].unique()
    submission_dict = dict(zip(vehicle_ids, predictions))
    ordered_predictions = [submission_dict.get(vid, 0) for vid in test_vehicle_ids]
    
    assert len(test_vehicle_ids) == len(ordered_predictions), "Mismatch in number of vehicles"
    assert len(test_vehicle_ids) == len(set(test_vehicle_ids)), "Duplicate vehicle_ids in test_X"
    assert len(test_vehicle_ids) == len(vehicle_ids), "Number of vehicles in submission does not match test_X"

    submission = pd.DataFrame({
        'vehicle_id': test_vehicle_ids,
        'change_section': ordered_predictions
    })

    submission['vehicle_id'] = submission['vehicle_id'].astype(str)
    submission['change_section'] = submission['change_section'].astype(str)
    submission.to_csv(output_path, index=False)
    print(f"📝 Submission saved to {output_path}")

# 데이터 로드 및 전처리
def load_and_preprocess_data(train_X_path, train_y_path, val_X_path, val_y_path, test_X_path):
    train_X = pd.read_csv(train_X_path)
    train_y = pd.read_csv(train_y_path)
    val_X = pd.read_csv(val_X_path)
    val_y = pd.read_csv(val_y_path)
    test_X = pd.read_csv(test_X_path)

    scaler = StandardScaler()
    features = ['position_y', 'speed', 'acceleration']
    train_X[features] = scaler.fit_transform(train_X[features])
    val_X[features] = scaler.transform(val_X[features])
    test_X[features] = scaler.transform(test_X[features])

    train_dataset = TrajectoryDataset(train_X, train_y, max_len=60)
    val_dataset = TrajectoryDataset(val_X, val_y, max_len=60)
    test_dataset = TrajectoryDataset(test_X, max_len=60)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, test_X

# 메인 실행 함수
def main():
    train_X_path = "train_X.csv"
    train_y_path = "train_y.csv"
    val_X_path = "validation_X.csv"
    val_y_path = "validation_y.csv"
    test_X_path = "test_X.csv"

    train_loader, val_loader, test_loader, test_X = load_and_preprocess_data(
        train_X_path, train_y_path, val_X_path, val_y_path, test_X_path
    )

    model = HybridModel(input_dim=3, tcn_channels=[64, 128, 256], gru_hidden=128, num_classes=4)

    train_model(model, train_loader, val_loader, epochs=20, lr=1e-3)

    model.load_state_dict(torch.load("model.pt"))

    predict_and_submit(model, test_loader, test_X, output_path="submission.csv")

if __name__ == "__main__":
    main()