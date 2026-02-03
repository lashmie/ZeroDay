# scripts/pcap_to_flow.py
from nfstream import NFStreamer
import pandas as pd
import os

pcap_dir = "../../REAPER_Project/pcap"
output_csv = "../data/flows.csv"

all_flows = []

for pcap_file in os.listdir(pcap_dir):
    if not pcap_file.endswith(".pcap"):
        continue

    print("Processing:", pcap_file)

    streamer = NFStreamer(
        source=os.path.join(pcap_dir, pcap_file),
        statistical_analysis=True,
        splt_analysis=False,
        n_dissections=20
    )

    for flow in streamer:
        all_flows.append({
            "timestamp": flow.bidirectional_first_seen_ms,
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "duration": flow.bidirectional_duration_ms,
            "fwd_pkts": flow.src2dst_packets,
            "bwd_pkts": flow.dst2src_packets,
            "fwd_bytes": flow.src2dst_bytes,
            "bwd_bytes": flow.dst2src_bytes,
            "pcap_file": pcap_file
        })

df = pd.DataFrame(all_flows)
df.sort_values("timestamp", inplace=True)
df.to_csv(output_csv, index=False)

print("Flow CSV generated:", output_csv)
