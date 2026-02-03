import numpy as np
from tensorflow.keras.models import load_model, Model
from preprocessing import load_data

# -------------------------------
# 1. Load trained REAPER model
# -------------------------------
print("▶ Loading REAPER model...")
reaper = load_model("model/reaper_model.h5")

# -------------------------------
# 2. Create embedding extractor
# -------------------------------
embedding_model = Model(
    inputs=reaper.input,
    outputs=reaper.get_layer("reaper_embedding").output
)

print("▶ Embedding model ready")

# -------------------------------
# 3. Load data (same as training)
# -------------------------------
print("▶ Loading flow sequences...")
X, y = load_data("data/flows.csv")

# -------------------------------
# 4. Run REAPER inference
# -------------------------------
print("▶ Running REAPER inference...")
p_mal = reaper.predict(X, batch_size=32).flatten()

# -------------------------------
# 5. Extract embeddings
# -------------------------------
print("▶ Extracting embeddings...")
embeddings = embedding_model.predict(X, batch_size=32)

print("Embeddings shape:", embeddings.shape)  # (N, 32)

# -------------------------------
# 6. Filter suspicious traffic
# -------------------------------
THRESHOLD = 0.5

suspicious_idx = p_mal >= THRESHOLD

suspicious_embeddings = embeddings[suspicious_idx]
suspicious_labels = y[suspicious_idx]

print("▶ Suspicious samples:", suspicious_embeddings.shape[0])

# -------------------------------
# 7. Save output for Paper-2
# -------------------------------
np.save("data/reaper_embeddings.npy", suspicious_embeddings)
np.save("data/reaper_labels.npy", suspicious_labels)

print("✅ Paper-1 output saved!")
print("   → data/reaper_embeddings.npy")
print("   → data/reaper_labels.npy")
