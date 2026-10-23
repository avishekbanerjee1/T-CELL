import cv2
import numpy as np
import skfuzzy as fuzz

IMG_SIZE = (224, 224)

def load_rgb(path, size=IMG_SIZE):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image

def fcm_membership_map(rgb, clusters=3, sample_limit=12000, seed=42):
    """
    Return a heuristic tumor-like membership map plus all cluster maps.
    FCM is applied to sampled RGB pixels. The cluster with the highest mean
    red-channel intensity is selected as a simple heuristic only; this is not
    a validated biological tumor detector.
    """
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float64) / 255.0
    rng = np.random.default_rng(seed)
    if len(pixels) > sample_limit:
        idx = rng.choice(len(pixels), sample_limit, replace=False)
        sampled = pixels[idx].T
    else:
        sampled = pixels.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        sampled, c=clusters, m=2.0, error=0.005, maxiter=100, seed=seed
    )
    # Assign all pixels to the learned cluster centers using normalized distances.
    # Membership calculation follows the standard fuzzy c-means distance formula.
    dist = np.linalg.norm(pixels[:, None, :] - cntr[None, :, :], axis=2) + 1e-8
    power = 2.0 / (2.0 - 1.0)
    inv = dist ** (-power)
    memberships = inv / np.sum(inv, axis=1, keepdims=True)
    maps = memberships.T.reshape(clusters, h, w).astype(np.float32)
    # Heuristic only: redder average cluster is used as an illustrative ROI map.
    red_means = [cntr[k, 0] for k in range(clusters)]
    tumor_idx = int(np.argmax(red_means))
    return maps[tumor_idx], maps

def make_model_input(rgb, membership):
    rgb_f = rgb.astype(np.float32) / 255.0
    # 4-channel input: normalized RGB plus the selected FCM membership map.
    return np.concatenate([rgb_f, membership[..., None]], axis=-1).astype(np.float32)

def save_fcm_overlay(rgb, membership, out_path):
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(rgb)
    plt.imshow(membership, cmap="jet", alpha=0.38, vmin=0, vmax=1)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close()
