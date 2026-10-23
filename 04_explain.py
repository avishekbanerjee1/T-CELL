import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fcm_utils import load_rgb, fcm_membership_map, make_model_input, save_fcm_overlay

def find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

def gradcam(model, x, conv_layer_name):
    grad_model = tf.keras.Model(model.inputs,
                                [model.get_layer(conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(x, training=False)
        score = pred[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    heat = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat, axis=(1, 2), keepdims=True) + 1e-8)
    return heat[0].numpy(), float(pred[0, 0].numpy())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--image", required=True)
    p.add_argument("--out", default="runs/explanations")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(args.model)
    rgb = load_rgb(args.image); membership, all_maps = fcm_membership_map(rgb)
    x = make_model_input(rgb, membership)[None, ...]
    heat, prob = gradcam(model, x, find_last_conv(model))
    heat = tf.image.resize(heat[..., None], (224,224)).numpy().squeeze()
    save_fcm_overlay(rgb, membership, out/"fcm_overlay.png")
    plt.figure(figsize=(6,5)); plt.imshow(rgb); plt.imshow(heat, cmap="jet", alpha=.4)
    plt.axis("off"); plt.tight_layout(); plt.savefig(out/"gradcam_overlay.png", dpi=160); plt.close()
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(rgb); plt.title("Input image"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(membership, cmap="viridis"); plt.title("Heuristic FCM membership"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(heat, cmap="jet"); plt.title("Grad-CAM"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out/"explanation_panel.png", dpi=160); plt.close()
    label = "malignant" if prob >= .5 else "benign"
    (out/"prediction.txt").write_text(
        f"Predicted class: {label}\nMalignant probability: {prob:.6f}\n"
        "Research demonstration only; not a clinical diagnosis.\n")
    print((out/"prediction.txt").read_text())
    print(f"Saved visualizations to {out.resolve()}")

if __name__ == "__main__":
    main()
