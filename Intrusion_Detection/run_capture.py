"""Capture live traffic, compute UNSW-NB15-style features, and save to CSV."""
import os

from packet_capture.capture import capture_packets
from packet_capture.feature_calculator import features_from_packets


if __name__ == "__main__":
    output_csv = os.path.join(os.path.dirname(__file__), "captured_features.csv")

    # Step 1: pick the interface that actually sees traffic on your machine.
    iface = "Wi-Fi"  # e.g., "Wi-Fi" or a specific adapter name

    # Step 2: use a broad filter; set to None to capture everything or "tcp" for TCP only.
    bpf_filter = None

    # Step 3: capture long enough while generating traffic (browse/download/ping/stream).
    capture_seconds = 60

    packets = capture_packets(
        iface=iface,
        duration=capture_seconds,
        bpf_filter=bpf_filter,
    )
    print(f"Captured {len(packets)} packets; saving to {output_csv}")

    # Convert to feature frame and persist.
    df = features_from_packets(packets)
    df.to_csv(output_csv, index=False)
    print(f"DataFrame shape: {df.shape}")
