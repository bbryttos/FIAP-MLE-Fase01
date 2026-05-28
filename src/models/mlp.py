import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
torch.manual_seed(SEED)


class ChurnMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# Alias para compatibilidade com testes e API existentes
MLP = ChurnMLP


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int] | None = None,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 200,
    dropout: float = 0.3,
    patience: int = 15,
    device: str = "cpu",
) -> tuple["ChurnMLP", dict[str, list[float]]]:
    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_vl = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True)

    model = ChurnMLP(X_train.shape[1], hidden_dims, dropout).to(device)

    # Pondera classe positiva para compensar o desbalanceamento (~26% churn)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    stopper = EarlyStopping(patience=patience)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), y_vl).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 20 == 0:
            logger.info(
                "Epoch {}/{} — train_loss={:.4f} val_loss={:.4f}",
                epoch, max_epochs, train_loss, val_loss,
            )

        if stopper.step(val_loss, model):
            logger.info("Early stopping at epoch {} (best val_loss={:.4f})", epoch, stopper.best_loss)
            break

    if stopper.best_state:
        model.load_state_dict(stopper.best_state)

    return model, history


def predict_proba(model: ChurnMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).cpu().numpy()
    return probs


class MLPTrainer:
    """Wrapper de treino compatível com a interface original (testes existentes)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
        device: str | None = None,
        random_state: int = 42,
        **_kwargs,
    ):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims or [64, 32, 16]
        self._dropout = dropout_rate
        self._lr = lr
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ChurnMLP | None = None
        self.history: dict[str, list] = {"train_loss": [], "val_loss": []}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "MLPTrainer":
        self.model, self.history = train_mlp(
            X_train, y_train.astype(np.float32),
            X_val, y_val.astype(np.float32),
            hidden_dims=self._hidden_dims,
            lr=self._lr,
            batch_size=self._batch_size,
            max_epochs=self._max_epochs,
            dropout=self._dropout,
            patience=self._patience,
            device=self._device,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Call fit() first"
        return predict_proba(self.model, X, device=self._device)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
