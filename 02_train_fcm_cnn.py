import argparse, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from fcm_utils import load_rgb, fcm_membership_map, make_model_input

def residual_block(x, filters, stride=1, dropout=0.0):
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y); y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if stride != 1 or int(x.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([shortcut, y]); y = layers.Activation("relu")(y)
    if dropout:
        y = layers.Dropout(dropout)(y)
    return y

def build_model():
    inp = layers.Input((224, 224, 4), name="rgb_plus_fcm")
    x = layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    for filters, stride in [(32,1),(64,2),(128,2),(256,2)]:
        x = residual_block(x, filters, stride=stride, dropout=0.05)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid", name="malignant_probability")(x)
    model = Model(inp, out, name="FCM_CNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model

class FCMSequence(tf.keras.utils.Sequence):
    def __init__(self, csv_path, batch_size=8, shuffle=False, augment=False, max_images=None):
        self.df = pd.read_csv(csv_path)
        if max_images:
            self.df = self.df.iloc[:max_images].copy()
        self.batch_size, self.shuffle, self.augment = batch_size, shuffle, augment
        self.indices = np.arange(len(self.df)); self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)
    def __getitem__(self, idx):
        ids = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        xs, ys = [], []
        for i in ids:
            row = self.df.iloc[i]
            rgb = load_rgb(row.path)
            if self.augment:
                if random.random() < .5: rgb = np.fliplr(rgb).copy()
                if random.random() < .5: rgb = np.flipud(rgb).copy()
                rgb = np.rot90(rgb, random.randrange(4)).copy()
            membership, _ = fcm_membership_map(rgb)
            xs.append(make_model_input(rgb, membership))
            ys.append(float(row.label))
        return np.stack(xs), np.array(ys, dtype=np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True); p.add_argument("--out", default="runs/fcm_cnn")
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-images", type=int, default=None, help="Optional small smoke-test cap per split")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train = FCMSequence(Path(args.split)/"train.csv", args.batch_size, True, True, args.max_images)
    val = FCMSequence(Path(args.split)/"val.csv", args.batch_size, False, False, args.max_images)
    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out/"best_model.keras"), monitor="val_auc",
                                           mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=4,
                                             min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(str(out/"training_log.csv"))
    ]
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)
    model.save(out/"final_model.keras")
    print(f"Saved model and logs to {out.resolve()}")

if __name__ == "__main__":
    main()
