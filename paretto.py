import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import os
import torch
from pathlib import Path

TARGET_LOSS = 0.15
DELTA_R = 0.05
MAX_EPS = 50000
ANGLES = np.linspace(0, np.pi/2, 9)
N_EVAL = 30

VAR_EVAL = np.linspace(0.0, 1.0, 11)
MAG_EVAL = np.linspace(0.0, 50.0, 11)

def polar_to_cartesian(r: float, theta: float) -> Tuple[float, float]:
    var = r * np.cos(theta)
    mag = r * np.sin(theta) * 50.0  # Scale magnitude to [0, 50]
    return var, mag

def train_model(model: Any, var: float, mag: float) -> Tuple[Any, float, bool]:
    # Train the model on the given variance and magnitude
    return model, 0.0, True  # Placeholder for actual training logic

def train_along_ray(theta: float, save_dir: str = "checkpoints") -> Dict:
    model = None  # Placeholder for model initialization
    r = 0.0
    history = []
    while True:
        var, mag = polar_to_cartesian(r, theta)
        print(f"Training at r={r:.2f}, var={var:.2f}, mag={mag:.2f}")
        model, loss, success = train_model(model, var, mag)
        history.append({"r": r, "var": var, "mag": mag, "loss": loss, "success": success})

        save_policy(model, f"{save_dir}/theta_{theta:.3f}_r_{r:.3f}.pt", theta=theta, max_r=r)

        if not success:
            print(f"Training failed at r={r:.2f}, var={var:.2f}, mag={mag:.2f}")
            break

        r += DELTA_R
        final_path = os.path.join(save_dir, f"policy_theta_{theta:.3f}.pt")
        save_policy(
            model,
            final_path,
            theta=theta,
            max_r=r,
            extra_info={"history": history},
        )
    
    return {"theta": theta, "max_r": r, "final_model": model, "path": final_path, "history": history}

def evaluate_model(model: Any, var: float, mag: float) -> float:
    # Evaluate the model on the given variance and magnitude
    return 0.0  # Placeholder for actual evaluation logic

def run_full_experiment() -> List[Dict]:
    results = []
    for theta in ANGLES:
        ray_result = train_along_ray(theta)
        results.append(ray_result)

        save_policy(ray_result["final_model"], f"policy_theta_{theta:.3f}.pt")

    return results

def cross_evaluate(results: List[Dict]) -> np.ndarray:
    #evaluate each model on the grid
    n_angles = len(results)
    heatmap = np.full((n_angles, len(VAR_EVAL), len(MAG_EVAL)), np.nan)

    for i, res in enumerate(results):
        model = res["final_model"]
        for j, var in enumerate(VAR_EVAL):
            for k, mag in enumerate(MAG_EVAL):
                heatmap[i, j, k] = evaluate_model(model, var, mag)

    return heatmap

def save_policy(model: Any, path: str, theta: Optional[float] = None, max_r: Optional[float] = None, extra_info: Optional[Dict] = None) -> None:
    #Save the checkpoints and the model to the specified path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else model,
        "theta": theta,
        "max_r": max_r,
        "extra_info": extra_info or {},
    }

    torch.save(checkpoint, path)
    print(f"Saved policy to {path}")

def load_policy(path: str, model: Any, device: torch.device) -> Dict:
    #load the model
    checkpoint = torch.load(path, map_location=device)
    if model is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    return checkpoint
