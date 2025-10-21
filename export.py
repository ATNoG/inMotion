import pandas as pd
import argparse
import ast
import os
from typing import List, Dict, Any, Optional


def extract_rssi_and_mac_pairs(
    data_dict: Dict[str, List[Dict[str, Any]]],
) -> List[tuple[float, str]]:
    """
    Extract all RSSI-MAC pairs from data dict, handling both phy0-ap0 and phy1-ap0.
    Returns a list of (rssi, mac) tuples for all valid measurements.
    """
    pairs = []

    # Check both phys
    for phy_key in ["phy0-ap0", "phy1-ap0"]: # TODO ADD MORE
        if phy_key in data_dict and len(data_dict[phy_key]) > 0:
            for station in data_dict[phy_key]:
                rssi = station.get("rssi", "N/A")
                mac = station.get("mac", "")
                # Only use valid RSSI values
                if rssi != "N/A" and isinstance(rssi, (int, float)):
                    pairs.append((float(rssi), mac))

    return pairs


def export_to_csv(
    input_file: str, output_file: str, window_size: int = 10, label: str = "AB"
) -> None:
    """
    Export wavecom data to CSV with sliding window approach.
    Each row contains window_size RSSI readings plus a label.

    Args:
            input_file: Path to input data file
            output_file: Path to output CSV file
            window_size: Number of time points per sample (default: 10)
    """
    # Read all lines and extract data
    data = []
    dict_strs = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            dict_strs.append(line.split(maxsplit=2)[2].strip())

    # Parse each dict string
    for dict_str in dict_strs:
        try:
            data_dict = ast.literal_eval(dict_str)
            data.append(data_dict)
        except Exception as e:
            print(f"Error parsing dict: {e}")
            continue

    # Extract all RSSI-MAC pairs from all data points
    all_pairs = []
    for data_dict in data:
        pairs = extract_rssi_and_mac_pairs(data_dict)
        all_pairs.extend(pairs)

    # Group pairs by MAC address
    mac_groups = {}
    for rssi, mac in all_pairs:
        if mac not in mac_groups:
            mac_groups[mac] = []
        mac_groups[mac].append(rssi)

    # Create records with non-overlapping windows for each MAC
    records = []

    for mac, rssi_values in mac_groups.items():
        # Process data in chunks of window_size for this MAC
        for i in range(0, len(rssi_values), window_size):
            # Get the next window_size values
            rssi_window = rssi_values[i : i + window_size]

            # If we have exactly window_size values, use them
            if len(rssi_window) == window_size:
                # Create record: [mac, rssi1, rssi2, ..., rssi10, label]
                record = [mac] + rssi_window + [label]
                records.append(record)
            # If we have fewer than window_size values, pad with None
            elif len(rssi_window) > 0:
                # Pad with None if needed
                padded_rssi = rssi_window + [None] * (window_size - len(rssi_window))
                record = [mac] + padded_rssi + [label]
                records.append(record)

    # Create DataFrame with proper headers
    headers = ["mac"] + [str(i) for i in range(1, window_size + 1)] + ["label"]
    df = pd.DataFrame(records, columns=headers)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Exported {len(records)} records to {output_file}")
    print(f"RSSI values extracted: {len(rssi_values)}")
    print(f"Label: {label}")


if __name__ == "__main__":
    # should receive the file flag and output flag
    parser = argparse.ArgumentParser(description="Export wavecom data to CSV")
    parser.add_argument("--label", type=str, required=True, help="Label for the data")
    parser.add_argument(
        "--input", type=str, required=True, help="Input wavecom data file"
    )
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    args = parser.parse_args()
    export_to_csv(args.input, args.output, 10, args.label)
