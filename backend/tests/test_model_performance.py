from pathlib import Path
import pandas as pd

#import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# =========================
# Configuração
# =========================
MODEL_PATH = Path("/home/mgasilva/code/puc_mvp/best_pneumonia_model.pth")
VAL_CSV = Path("/home/mgasilva/code/puc_mvp/data/val_labels.csv")

BATCH_SIZE = 32
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds mínimos exigidos
MIN_RECALL_PNEUMONIA = 0.90
MIN_ROC_AUC = 0.85
MIN_F1 = 0.80
MIN_ACCURACY = 0.80


# =========================
# Dataset
# =========================
class XRayValidationDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, label


# =========================
# Modelo
# =========================
def build_model() -> nn.Module:
    """
    Replica a arquitetura usada no app.py:
    EfficientNet-B0 com saída binária.
    """
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    return model


def load_trained_model(model_path: Path) -> nn.Module:
    assert model_path.exists(), f"Modelo não encontrado em: {model_path}"

    model = build_model()
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# =========================
# Inferência no conjunto de validação
# =========================
def run_inference(model: nn.Module, dataloader: DataLoader):
    y_true = []
    y_pred = []
    y_prob_pneumonia = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            pred_labels = torch.argmax(probs, dim=1)
            prob_pneumonia = probs[:, 1]

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred_labels.cpu().numpy().tolist())
            y_prob_pneumonia.extend(prob_pneumonia.cpu().numpy().tolist())

    return y_true, y_pred, y_prob_pneumonia


# =========================
# Fixture compartilhada
# =========================
@pytest.fixture(scope="module")
def validation_metrics():
    assert VAL_CSV.exists(), f"CSV de validação não encontrado em: {VAL_CSV}"

    dataset = XRayValidationDataset(VAL_CSV)
    assert len(dataset) > 0, "O conjunto de validação está vazio."

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = load_trained_model(MODEL_PATH)

    y_true, y_pred, y_prob = run_inference(model, dataloader)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_pneumonia": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    return metrics


# =========================
# Testes
# =========================
def test_model_accuracy(validation_metrics):
    accuracy = validation_metrics["accuracy"]
    assert accuracy >= MIN_ACCURACY, (
        f"Accuracy abaixo do mínimo: {accuracy:.3f} < {MIN_ACCURACY:.3f}"
    )


def test_model_recall_for_pneumonia(validation_metrics):
    recall = validation_metrics["recall_pneumonia"]
    assert recall >= MIN_RECALL_PNEUMONIA, (
        f"Recall para PNEUMONIA abaixo do mínimo: "
        f"{recall:.3f} < {MIN_RECALL_PNEUMONIA:.3f}"
    )


def test_model_f1(validation_metrics):
    f1 = validation_metrics["f1"]
    assert f1 >= MIN_F1, (
        f"F1-score abaixo do mínimo: {f1:.3f} < {MIN_F1:.3f}"
    )


def test_model_roc_auc(validation_metrics):
    roc_auc = validation_metrics["roc_auc"]
    assert roc_auc >= MIN_ROC_AUC, (
        f"ROC AUC abaixo do mínimo: {roc_auc:.3f} < {MIN_ROC_AUC:.3f}"
    )

