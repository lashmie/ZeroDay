# train.py
import os

# CPU + NUMA safe limits
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf
from reaper_model import build_reaper
from preprocessing import load_data

print("▶ Loading data...")
X, y = load_data("data/flows.csv")

print("▶ Building model...")
model = build_reaper(input_shape=(X.shape[1], X.shape[2]))

# 🔥 Use tf.data for stability
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(10000).batch(16).prefetch(2)

print("▶ Starting training...")

BATCH_SIZE = 16
total_batches = len(X) // BATCH_SIZE
val_batches = int(0.2 * total_batches)

train_ds = dataset.skip(val_batches)
val_ds = dataset.take(val_batches)

model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)

model.save("model/reaper_model.h5")
print("▶ Training finished")
