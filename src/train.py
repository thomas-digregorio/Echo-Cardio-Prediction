import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import argparse
import sys
import yaml
from pathlib import Path
from transformers import get_cosine_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dali_loader import get_dataloader
from src.basic_model import EchoNetRegressor


def load_config(config_path: str | None) -> dict:
    """Load config from YAML file, with defaults."""
    default_config = {
        "data": {"data_dir": None},
        "training": {
            "epochs": 10, "batch_size": 4, "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5, "weight_decay": 0.01, "max_grad_norm": 1.0, "warmup_ratio": 0.1
        },
        "model": {"freeze_backbone": False},
        "amp": {"enabled": True, "dtype": "float16"},
        "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.01},
        "checkpoints": {"output_dir": "./checkpoints", "save_steps": 500, "eval_steps": 1000},
        "logging": {"project": "EchoNet-VideoMAE"},
        "loss": {"use_clinical_weights": True}
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        # Merge configs (user overrides defaults)
        for key in user_config:
            if key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(user_config[key])
            else:
                default_config[key] = user_config[key]
    
    return default_config


def get_clinical_weight(ef: torch.Tensor) -> torch.Tensor:
    """Clinical importance weighting for EF values."""
    weights = torch.ones_like(ef)
    weights = torch.where(ef < 30, torch.tensor(3.0, device=ef.device), weights)
    weights = torch.where((ef >= 30) & (ef < 40), torch.tensor(2.0, device=ef.device), weights)
    weights = torch.where(ef > 70, torch.tensor(2.5, device=ef.device), weights)
    return weights


def weighted_mse_loss(preds: torch.Tensor, targets: torch.Tensor, use_weights: bool = True) -> torch.Tensor:
    """MSE loss weighted by clinical importance."""
    if use_weights:
        weights = get_clinical_weight(targets)
        mse = (preds - targets) ** 2
        return (weights * mse).mean()
    return nn.functional.mse_loss(preds, targets)


def compute_per_bin_mae(preds: list, labels: list) -> dict:
    """Compute MAE for each EF bin."""
    import numpy as np
    preds = np.array(preds)
    labels = np.array(labels)
    
    bins = {
        "severe_hf": (0, 30),
        "moderate_hf": (30, 40),
        "mild": (40, 55),
        "normal": (55, 70),
        "hyperdynamic": (70, 100)
    }
    
    results = {}
    for name, (low, high) in bins.items():
        mask = (labels >= low) & (labels < high)
        if mask.sum() > 0:
            bin_mae = np.abs(preds[mask] - labels[mask]).mean()
            results[f"mae_{name}"] = bin_mae
            results[f"count_{name}"] = int(mask.sum())
    
    return results


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: torch.device, use_amp: bool = True, max_batches: int = 50) -> dict:
    """
    Run evaluation on validation/test set.
    
    Args:
        max_batches: Limit number of batches to avoid DALI hanging issues.
                     Set to -1 for full evaluation.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    try:
        for batch_idx, batch in enumerate(val_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
                
            frames = batch[0]['frames']
            labels = batch[0]['label'].to(dtype=torch.float32)
            
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    preds = model(frames)
            else:
                preds = model(frames)
            
            loss = weighted_mse_loss(preds, labels)
            mae = torch.abs(preds - labels).mean()
            
            batch_size = frames.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            total_samples += batch_size
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    except StopIteration:
        pass
    finally:
        # Always reset DALI iterator
        try:
            val_loader.dali_iter.reset()
        except Exception:
            pass
    
    model.train()
    
    if total_samples == 0:
        return {"loss": float('inf'), "mae": float('inf')}
    
    results = {
        "loss": total_loss / total_samples,
        "mae": total_mae / total_samples,
    }
    
    # Add per-bin MAE
    results.update(compute_per_bin_mae(all_preds, all_labels))
    
    return results


class EarlyStopping:
    """Early stopping to halt training when val_loss stops improving."""
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    cfg_train = config["training"]
    cfg_amp = config["amp"]
    cfg_ckpt = config["checkpoints"]
    cfg_early = config["early_stopping"]

    # Init WandB
    wandb.init(project=config["logging"]["project"], config=config)

    # Data paths
    if config["data"]["data_dir"] is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config["data"]["data_dir"] = os.path.join(base_dir, "EchoNet-Dynamic")
    
    csv_path = os.path.join(config["data"]["data_dir"], "FileList.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find FileList.csv at {csv_path}")

    print(f"Loading data from: {config['data']['data_dir']}")
    
    # Data loaders
    train_loader = get_dataloader(
        data_dir=config["data"]["data_dir"], csv_path=csv_path,
        batch_size=cfg_train["batch_size"], training=True, split="TRAIN"
    )
    val_loader = get_dataloader(
        data_dir=config["data"]["data_dir"], csv_path=csv_path,
        batch_size=cfg_train["batch_size"], training=False, split="VAL"
    )
    test_loader = get_dataloader(
        data_dir=config["data"]["data_dir"], csv_path=csv_path,
        batch_size=cfg_train["batch_size"], training=False, split="TEST"
    )
    print(f"Train: {len(train_loader)*cfg_train['batch_size']}, Val: {len(val_loader)*cfg_train['batch_size']}, Test: {len(test_loader)*cfg_train['batch_size']}")

    # Model
    model = EchoNetRegressor(freeze_backbone=config["model"]["freeze_backbone"])
    model.to(device)

    # Optimizer with differential LR
    if config["model"]["freeze_backbone"]:
        trainable_params = list(model.head.parameters()) + list(model.attention.parameters())
        optimizer = optim.AdamW(trainable_params, lr=cfg_train["learning_rate"], weight_decay=cfg_train["weight_decay"])
    else:
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': cfg_train["learning_rate"]},
            {'params': model.attention.parameters(), 'lr': cfg_train["learning_rate"] * 100},
            {'params': model.head.parameters(), 'lr': cfg_train["learning_rate"] * 100}
        ], weight_decay=cfg_train["weight_decay"])
    
    # Scheduler
    total_steps = len(train_loader) * cfg_train["epochs"] // cfg_train["gradient_accumulation_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg_train["warmup_ratio"] * total_steps),
        num_training_steps=total_steps
    )
    print(f"Scheduler: {int(cfg_train['warmup_ratio'] * total_steps)} warmup, {total_steps} total steps")

    # Mixed Precision
    use_amp = cfg_amp["enabled"] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_amp else None
    amp_dtype = torch.float16 if cfg_amp.get("dtype", "float16") == "float16" else torch.bfloat16
    print(f"Mixed Precision: {'Enabled (' + cfg_amp.get('dtype', 'float16') + ')' if use_amp else 'Disabled'}")

    # Early stopping
    early_stopper = EarlyStopping(patience=cfg_early["patience"], min_delta=cfg_early["min_delta"]) if cfg_early["enabled"] else None

    # Training
    os.makedirs(cfg_ckpt["output_dir"], exist_ok=True)
    print("Starting Training...")
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(cfg_train["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_train['epochs']}")
        
        for batch in pbar:
            frames = batch[0]['frames']
            labels = batch[0]['label'].to(dtype=torch.float32)

            # Forward with AMP
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    preds = model(frames)
                    loss = weighted_mse_loss(preds, labels, config["loss"]["use_clinical_weights"])
                loss = loss / cfg_train["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
            else:
                preds = model(frames)
                loss = weighted_mse_loss(preds, labels, config["loss"]["use_clinical_weights"])
                loss = loss / cfg_train["gradient_accumulation_steps"]
                loss.backward()

            # Update weights
            if (step + 1) % cfg_train["gradient_accumulation_steps"] == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train["max_grad_norm"])
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # Only step scheduler if optimizer actually stepped (scaler didn't skip)
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                
                optimizer.zero_grad()
                
                loss_val = loss.item() * cfg_train["gradient_accumulation_steps"]
                wandb.log({
                    "train_loss": loss_val,
                    "pred_mean": preds.mean().item(),
                    "pred_std": preds.std().item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": step, "epoch": epoch
                })
                pbar.set_postfix({"loss": f"{loss_val:.2f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            
            step += 1
            
            # Checkpoint
            if step % cfg_ckpt["save_steps"] == 0:
                torch.save(model.state_dict(), os.path.join(cfg_ckpt["output_dir"], f"model_step_{step}.pt"))
            
            # Validation
            if step % cfg_ckpt["eval_steps"] == 0:
                val_metrics = evaluate(model, val_loader, device, use_amp)
                wandb.log({f"val_{k}": v for k, v in val_metrics.items()} | {"step": step})
                print(f"\n[Step {step}] Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}%")
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save(model.state_dict(), os.path.join(cfg_ckpt["output_dir"], "best_model.pt"))
                    print("  -> New best model!")
                
                # Early stopping check
                if early_stopper and early_stopper(val_metrics['loss']):
                    print(f"\nEarly stopping triggered after {early_stopper.counter} evals without improvement.")
                    break
        
        train_loader.dali_iter.reset()
        
        if early_stopper and early_stopper.should_stop:
            break
        
        torch.save(model.state_dict(), os.path.join(cfg_ckpt["output_dir"], f"model_epoch_{epoch+1}.pt"))

    # Final Test Evaluation
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    test_metrics = evaluate(model, test_loader, device, use_amp)
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
    
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.2f}%")
    print("\nPer-bin MAE Breakdown:")
    for key, val in test_metrics.items():
        if key.startswith("mae_"):
            bin_name = key.replace("mae_", "")
            count = test_metrics.get(f"count_{bin_name}", 0)
            print(f"  {bin_name}: {val:.2f}% (n={count})")

    torch.save(model.state_dict(), os.path.join(cfg_ckpt["output_dir"], "last_model.pt"))
    wandb.finish()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)
