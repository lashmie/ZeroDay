import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN = 10
MAX_SEQUENCES = 100000

scaler = MinMaxScaler()

def load_data(csv_path):
    print("▶ Reading CSV...")
    
    # Load with dtype optimization
    df = pd.read_csv(csv_path, nrows=200000, dtype={
        'src_port': np.int32,
        'dst_port': np.int32,
        'protocol': np.int32,
        'duration': np.float32,
        'fwd_pkts': np.int32,
        'bwd_pkts': np.int32,
        'fwd_bytes': np.int32,
        'bwd_bytes': np.int32
    })
    
    print(f"▶ Memory usage before cleaning: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    df.dropna(inplace=True)

    print("▶ Rows loaded:", len(df))

    if "label" not in df.columns:
        df["label"] = ((df["fwd_pkts"] + df["bwd_pkts"]) > 20).astype(np.int8)

    numeric_cols = [
        'src_port', 'dst_port', 'protocol',
        'duration', 'fwd_pkts', 'bwd_pkts',
        'fwd_bytes', 'bwd_bytes'
    ]
    
    # Only scale numeric columns that exist
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Scale in chunks to save memory
    for col in existing_numeric_cols:
        df[col] = scaler.fit_transform(df[[col]].values).astype(np.float32)
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Create sequences efficiently using numpy arrays
    X_seq = []
    y_seq = []
    
    # Convert to numpy arrays for faster slicing
    numeric_data = df[existing_numeric_cols].values.astype(np.float32)
    labels = df["label"].values.astype(np.int8)
    
    # Calculate total sequences
    total_sequences = min(len(df) - SEQ_LEN + 1, MAX_SEQUENCES)
    
    print(f"▶ Creating {total_sequences} sequences...")
    
    for i in range(total_sequences):
        X_seq.append(numeric_data[i:i + SEQ_LEN])
        y_seq.append(labels[i:i + SEQ_LEN].max())
        
        # Progress indicator
        if i % 10000 == 0 and i > 0:
            print(f"  Created {i}/{total_sequences} sequences")
    
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int8)
    
    print("▶ Sequence shapes:", X_seq.shape, y_seq.shape)
    print(f"▶ Sequence memory usage: {X_seq.nbytes / 1024**2:.2f} MB")
    print("▶ Label distribution:", np.unique(y_seq, return_counts=True))

    return X_seq, y_seq