"""
Federated convergence analysis.
Plots convergence curves for FedAvg vs FedProx.

Usage:
    python training/evaluate_convergence.py
"""

import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import LOG_DIR

def plot_convergence():
    """Plot convergence curves from training logs."""
    
    # Load training history (you'll need to save this during training)
    fedavg_log = LOG_DIR / "fedavg_history.json"
    fedprox_log = LOG_DIR / "fedprox_history.json"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if fedavg_log.exists():
        with open(fedavg_log, "r") as f:
            fedavg = json.load(f)
        rounds = list(range(1, len(fedavg['val_kappa']) + 1))
        axes[0].plot(rounds, fedavg['val_kappa'], 'b-o', label='FedAvg', markersize=4)
        axes[1].plot(rounds, fedavg['val_loss'], 'b-o', label='FedAvg', markersize=4)
    
    if fedprox_log.exists():
        with open(fedprox_log, "r") as f:
            fedprox = json.load(f)
        rounds = list(range(1, len(fedprox['val_kappa']) + 1))
        axes[0].plot(rounds, fedprox['val_kappa'], 'r-s', label='FedProx', markersize=4)
        axes[1].plot(rounds, fedprox['val_loss'], 'r-s', label='FedProx', markersize=4)
    
    axes[0].set_xlabel('Federated Round')
    axes[0].set_ylabel('Validation Kappa (κ)')
    axes[0].set_title('Convergence: Quadratic Kappa')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Federated Round')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Convergence: Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = LOG_DIR / "convergence_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Convergence plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_convergence()