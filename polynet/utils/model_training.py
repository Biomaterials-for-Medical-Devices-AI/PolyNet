from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from polynet.options.enums import ProblemTypes, Results


def train_network(model, train_loader, loss_fn, optimizer, device):
    train_loss = 0
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch_index=batch.batch,
            edge_attr=batch.edge_attr,
            monomer_weight=batch.weight_monomer,
        )

        if model.problem_type == ProblemTypes.Regression:
            loss = torch.sqrt(loss_fn(out.squeeze(1), batch.y.float()))
        elif model.problem_type == ProblemTypes.Classification:
            loss = loss_fn(out, batch.y.long())

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    return train_loss / len(train_loader.dataset)


def eval_network(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch_index=batch.batch,
                edge_attr=batch.edge_attr,
                monomer_weight=batch.weight_monomer,
            )

            if model.problem_type == ProblemTypes.Regression:
                loss = torch.sqrt(loss_fn(out.squeeze(1), batch.y.float()))
            elif model.problem_type == ProblemTypes.Classification:
                loss = loss_fn(out, batch.y.long())

            test_loss += loss.item() * batch.num_graphs

    return test_loss / len(test_loader.dataset)


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
        if model.problem_type == "classification":
            y_score = np.concatenate(y_score, axis=0)
        else:
            y_score = None

    results = {Results.Predicted.value: y_pred, Results.Label.value: y_true, "Index": idx}

    results = pd.DataFrame(results)

    return results


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


def train_model(
    model, train_loader, val_loader, test_loader, loss, optimizer, scheduler, device, epochs=250
):
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_network(model, train_loader, loss, optimizer, device)
        val_loss = eval_network(model, val_loader, loss, device)
        test_loss = eval_network(model, test_loader, loss, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())

        scheduler.step(val_loss)

        print(
            f"Epoch: {epoch:03d}, LR: {scheduler.get_last_lr()[0]:3f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    model.load_state_dict(best_model_state)

    return model
