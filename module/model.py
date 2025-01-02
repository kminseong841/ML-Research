import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from IPython.display import clear_output

# Main Architecture
class OzoneModel(nn.Module):
  def __init__(self, n_filter1, n_filter2, dense_units, mask_ts, n_channels):
    super(OzoneModel, self).__init__()

    self.conv1 = nn.Conv3d(
            in_channels=n_channels,
            out_channels=n_filter1,
            kernel_size=(2, 3, 3)
            )

    self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))

    self.conv2 = nn.Conv3d(
        in_channels = n_filter1,
        out_channels = n_filter2,
        kernel_size = (2,3,3)
    )

    self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))

    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features=n_filter2 * 1 * 6 * 6, out_features=dense_units)
    self.fc2 = nn.Linear(in_features=dense_units, out_features=32 * 32)


     # 마스크 등록 (평탄화된 형태)
    mask_ts = torch.tensor(mask_ts, dtype=torch.float32)
    mask_flat = mask_ts.view(-1)  # (1024,)
    self.register_buffer('mask', mask_flat)  # 마스크 등록

  def forward(self, x, onlyStation = True):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)
    if onlyStation:
      x = x * self.mask
    else:
      x = x

    return x

  def update_prediction(self, pred):
    self.prediction = pred

# Loss function
class maskedHuber(nn.Module):
    def __init__(self, mask, delta=1.0):
        super(maskedHuber, self).__init__()
        self.register_buffer('mask', mask)
        self.delta = delta

    def forward(self, preds, targets):
        diff = preds - targets
        valid_columns = (self.mask != 0)
        diff = diff[:, valid_columns]

        abs_diff = torch.abs(diff)

        # Huber loss 공식:
        # |diff| <= delta: 0.5 * diff^2
        # |diff| > delta: delta * |diff| - 0.5 * delta^2
        quadratic_part = torch.minimum(abs_diff, torch.tensor(self.delta, device=abs_diff.device))
        linear_part = abs_diff - quadratic_part

        huber = 0.5 * quadratic_part ** 2 + self.delta * linear_part
        return torch.mean(huber)

# 모델 학습 함수
def train_model(model, train_data, valid_data, criterion, optimizer, epochs, patience, device='cuda' if torch.cuda.is_available() else 'cpu'):
  model.to(device)

  # for early stopping
  best_val_loss = float('inf')
  best_val_metric = float('inf')
  epochs_no_improve = 0
  best_model_state = None

  for epoch in range(epochs):
    model.train()
    train_loss_sum_batch = 0.0
    train_metric_sum_batch = 0.0

    # training (per batch)
    for inputs, targets in train_data:
      inputs = inputs.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()
      pred = model(inputs)
      loss = criterion(pred, targets)
      loss.backward()
      optimizer.step()
      mae = torch.mean(torch.abs(pred-targets))
        
        # Batch마다 평균낸 loss를 다시 합친다
      train_loss_sum_batch += loss.item()*inputs.size(0)
      train_metric_sum_batch += mae.item()*inputs.size(0)

    train_loss = train_loss_sum_batch / len(train_data.dataset)
    train_metric = train_metric_sum_batch / len(train_data.dataset)


    # evaluate (per batch)
    model.eval()
    val_loss_sum_batch = 0.0
    val_metric_sum_batch = 0.0

    with torch.no_grad():
      for inputs, targets in valid_data:
        inputs = inputs.to(device)
        targets = targets.to(device)
        pred = model(inputs)
        loss = criterion(pred, targets)
        mae = torch.mean(torch.abs(pred-targets))
    
        # Batch마다 평균낸 loss를 다시 합친다
        val_loss_sum_batch += loss.item() * inputs.size(0)
        val_metric_sum_batch += mae.item() * inputs.size(0)

    val_loss = val_loss_sum_batch / len(valid_data.dataset)
    val_metric = val_metric_sum_batch / len(valid_data.dataset)

    print(f'Epoch {epoch+1}/{epochs} | Train HuberLoss: {train_loss:.4f} | Train MAE: {train_metric:.4f} | Val HuberLoss: {val_loss:.4f}| Val MAE: {val_metric:.4f}')

    # Early Stopping 체크 (둘 다 증가하지 않으면 학습을 멈춘다.)
    if val_loss < best_val_loss or val_metric < best_val_metric:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_metric < best_val_metric:
            best_val_metric = val_metric
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

  # 베스트 모델 로드
  if best_model_state is not None:
      model.load_state_dict(best_model_state)

  return model

# 모델 평가 함수
def evaluate_model(model, test_data, criterion, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    test_loss_per_batch = 0.0
    test_mae_per_batch = 0.0

    with torch.no_grad():
        for inputs, targets in test_data:
            inputs = inputs.to(device)
            targets = targets.to(device)

            pred = model(inputs)
            loss = criterion(pred, targets)
            mae = torch.mean(torch.abs(pred-targets))

            test_loss_per_batch += loss.item() * inputs.size(0)
            test_mae_per_batch += mae.item() * inputs.size(0)

    test_loss = test_loss_per_batch / len(test_data.dataset)
    test_mae = test_mae_per_batch / len(test_data.dataset)

    print(f'Test HuberLoss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

    return test_loss, test_mae

# 예측 수행 함수 정의
def predict(model, data_loader, onlyStation=True, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)  # TensorDataset의 경우 batch는 튜플 형태
            outputs = model(inputs, onlyStation=onlyStation)
            predictions = outputs.cpu().numpy()
            all_predictions.append(predictions)

    # 모든 배치의 예측을 하나의 배열로 합침
    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions

# 데이터 텐서로 변환 및 데이터로더 생성
def convertLoader(x, y=None, batch_size=16, shuffle=False):
    x = np.transpose(x, (0, 4, 1, 2, 3))

    # NumPy 배열을 PyTorch 텐서로 변환
    x_tensor = torch.tensor(x, dtype=torch.float32)

    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
    else:
        dataset = TensorDataset(x_tensor)

    # DataLoader 생성
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# 교차 검증을 통해 모델을 학습하는 함수
def TrainML(xdata, ydata, bit_mask, n_features):
    n = 5
    tscv = TimeSeriesSplit(n_splits=n)

    model_list = []
    train_S = np.zeros((n, 2))
    valid_S = np.zeros((n, 2))

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(xdata)):
        clear_output(wait=True)
        xtrain, xvalid = xdata[train_idx], xdata[valid_idx]
        ytrain, yvalid = ydata[train_idx], ydata[valid_idx]

        train_loader = convertLoader(xtrain, ytrain)
        valid_loader = convertLoader(xvalid, yvalid)

        # **bit_mask와 n_features를 인자로 사용**
        model = OzoneModel(
            n_filter1 = 32,
            n_filter2 = 128,
            dense_units = 64,
            mask_ts = bit_mask,
            n_channels = n_features + 1
        )

        trained_model = train_model(
            model = model,
            train_data = train_loader,
            valid_data = valid_loader,
            criterion = maskedHuber(mask=model.mask, delta=0.35),
            optimizer = Adam(model.parameters(), lr=1e-4),
            epochs = 300,
            patience = 10
        )

        model_list.append(model)

        train_score = evaluate_model(
            model = trained_model,
            test_data = train_loader,
            criterion = maskedHuber(mask=model.mask, delta=0.35)
        )
        valid_score = evaluate_model(
            model = trained_model,
            test_data = valid_loader,
            criterion = maskedHuber(mask=model.mask, delta=0.35)
        )
        train_S[fold, :] = train_score
        valid_S[fold, :] = valid_score

    print(f"Mean Train MAE --> {np.mean(train_S, axis = 0)[1]:.4f}")
    print(f"Mean Validation MAE --> {np.mean(valid_S, axis = 0)[1]:.4f}")

    return model_list


# 예측 수행
def PredictML(model_list, xdata, onlyStation=True):
	# Pred_list 선언
	n_model = len(model_list)
	n_data = xdata.shape[0]
	pred_list = np.zeros((n_model, n_data, 1024))

	data_loader = convertLoader(xdata)

	for idx, model in enumerate(model_list):
		pred_list[idx] = predict(model, data_loader, onlyStation=onlyStation)

	prediction = np.mean(pred_list, axis = 0)

	return prediction

# 평가 수행
def EvaluateML(model_list, xdata, ydata):
  n_model = len(model_list)
  # 교차 검증 동안 score를 저장
  test_S = np.zeros((n_model,2))

  # convert loader
  test_loader = convertLoader(xdata, ydata)

  for idx, model in enumerate(model_list):
    print(f"Model{idx+1} Prediction: ")
    test_S[idx, :] = evaluate_model(model = model, test_data=test_loader, criterion=maskedHuber(mask=model.mask, delta=0.35))

  print(f"Mean Test MAE --> {np.mean(test_S, axis = 0)[1]:.4f}")