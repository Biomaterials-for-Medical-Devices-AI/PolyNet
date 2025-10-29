import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


from polynet.options.enums import ProblemTypes


def predict_network(model, loader):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    y_pred, y_true, idx, y_score = [], [], [], []

    with torch.no_grad():

        for batch in loader:
            batch = batch.to(device)

            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch_index=batch.batch,
                edge_attr=batch.edge_attr,
                monomer_weight=batch.weight_monomer,
            )

            out = out.cpu().detach()

            if model.problem_type == ProblemTypes.Classification:
                if out.dim() == 1 or out.size(1) == 1:
                    probs = torch.sigmoid(out)
                    preds = (probs >= 0.5).long()
                    y_score.append(probs.numpy().flatten())
                else:
                    probs = torch.softmax(out, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    y_score.append(probs.numpy())  # Shape: [batch_size, num_classes]
                y_pred.append(preds.numpy().flatten())
            else:
                out = out.numpy().flatten()
                y_pred.append(out)
                y_score = None  # No probabilities for regression

            y_true.append(batch.y.cpu().detach().numpy().flatten())
            idx.append(batch.idx)

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        idx = np.concatenate(idx, axis=0)
        if model.problem_type == ProblemTypes.Classification:
            y_score = np.concatenate(y_score, axis=0)
        else:
            y_score = None

    return idx, y_true, y_pred, y_score


def plot_training_curve(train_losses, val_losses, test_losses, best_epoch=None):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(test_losses, label="Test Loss")
    if best_epoch:
        plt.axvline(x=best_epoch, color="red", linestyle="--", label="Best Epoch")
    plt.xlabel("Epoch", size=12)
    plt.ylabel("Loss", size=12)
    plt.legend()
    plt.show()


def plot_parity_plot(df):
    sns.lmplot(x="True", y="Predicted", data=df, hue="set")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.show()
