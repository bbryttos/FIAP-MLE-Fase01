import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class MLPTrainer:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        random_state: int = 42,
    ):
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.history: dict[str, list] = {"train_loss": [], "val_loss": []}

    def _to_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        drop = shuffle  # drop last incomplete batch only during training
        return DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=shuffle, drop_last=drop)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "MLPTrainer":
        train_loader = self._to_loader(X_train, y_train, shuffle=True)
        val_loader = self._to_loader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_weights = None

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(X_train)

            val_loss = self._eval_loss(val_loader, len(X_val))
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %3d | Train Loss: %.4f | Val Loss: %.4f",
                    epoch, train_loss, val_loss,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("Early stopping at epoch %d (best val_loss=%.4f)", epoch, best_val_loss)
                    break

        if best_weights:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
        return self

    def _eval_loss(self, loader: DataLoader, n: int) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self.model(X_batch)
                total_loss += self.criterion(logits, y_batch).item() * len(X_batch)
        return total_loss / n

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
