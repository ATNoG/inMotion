import pandas as pd
import argparse
import ast
from typing import Any


def extract_rssi_and_mac_pairs(
    data_dict: dict[str, list[dict[str, Any]]],
) -> list[tuple[float, str]]:

    pairs = []

    mac_to_rssis: dict[str, list[float]] = {}

    for phy_key, stations in data_dict.items():
        if not str(phy_key).startswith("phy"):
            continue
        if not stations:
            continue
        for station in stations:
            rssi = station.get("rssi", "N/A")
            mac = station.get("mac", "")
            if mac and rssi != "N/A" and isinstance(rssi, (int, float)):
                mac_to_rssis.setdefault(mac, []).append(float(rssi))

    for mac, rssi_list in mac_to_rssis.items():
        if rssi_list:
            mean_rssi = sum(rssi_list) / len(rssi_list)
            pairs.append((mean_rssi, mac))


    return pairs


def export_to_csv(
    input_file: str, output_file: str, window_size: int = 10, label: str = "AB"
) -> None:

    data = []
    dict_strs = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            dict_strs.append(line.split(maxsplit=2)[2].strip())

    for dict_str in dict_strs:
        try:
            data_dict = ast.literal_eval(dict_str)
            data.append(data_dict)
        except Exception as e:
            print(f"Error parsing dict: {e}")
            continue

    all_pairs = []
    for data_dict in data:
        pairs = extract_rssi_and_mac_pairs(data_dict)
        all_pairs.extend(pairs)

    mac_groups = {}
    for rssi, mac in all_pairs:
        if mac not in mac_groups:
            mac_groups[mac] = []
        mac_groups[mac].append(rssi)

    records = []

    for mac, rssi_values in mac_groups.items():
        for i in range(0, len(rssi_values), window_size):
            # Get the next window_size values
            rssi_window = rssi_values[i : i + window_size]

            if len(rssi_window) == window_size:
                record = [mac] + rssi_window + [label]
                records.append(record)
            elif len(rssi_window) > 0:
                padded_rssi = rssi_window + [None] * (window_size - len(rssi_window))
                record = [mac] + padded_rssi + [label]
                records.append(record)

    headers = ["mac"] + [str(i) for i in range(1, window_size + 1)] + ["label"]
    df = pd.DataFrame(records, columns=headers)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Exported {len(records)} records to {output_file}")
    print(f"RSSI values extracted: {len(rssi_values)}")
    print(f"Label: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export wavecom data to CSV")
    parser.add_argument(
        "--label", "-l", type=str, required=True, help="Label for the data"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input wavecom data file"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output CSV file"
    )
    args = parser.parse_args()
    export_to_csv(args.input, args.output, 10, args.label)
