import pandas as pd
import argparse
import ast
import os
from typing import List, Dict, Any, Optional

def extract_rssi(data_dict: Dict[str, List[Dict[str, Any]]]) -> Optional[float]:
	"""
	Extract RSSI value from data dict, handling both phy0-ap0 and phy1-ap0.
	Prioritizes authorized connections and handles 'N/A' values.
	If both phys have valid RSSI, returns the mean.
	"""
	rssi_values = []
	
	# Check both phys
	for phy_key in ['phy0-ap0', 'phy1-ap0']:
		if phy_key in data_dict and len(data_dict[phy_key]) > 0:
			for station in data_dict[phy_key]:
				rssi = station.get('rssi', 'N/A')
				# Only use valid RSSI values
				if rssi != 'N/A' and isinstance(rssi, (int, float)):
					rssi_values.append(float(rssi))
	
	if len(rssi_values) == 0:
		return None
	elif len(rssi_values) == 1:
		return rssi_values[0]
	else:
		# Return mean if multiple valid values
		return sum(rssi_values) / len(rssi_values)



def export_to_csv(input_file: str, output_file: str, window_size: int = 10, label: str = 'AB') -> None:
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
	with open(input_file, "r") as f:
		content = f.read()
	
	# Split by dict boundaries - find all {..} occurrences
	# This handles both newline-separated and concatenated formats
	dict_strs = []
	current_pos = 0
	
	while current_pos < len(content):
		# Find next opening brace
		start = content.find('{', current_pos)
		if start == -1:
			break
		
		# Find matching closing brace
		brace_count = 0
		i = start
		while i < len(content):
			if content[i] == '{':
				brace_count += 1
			elif content[i] == '}':
				brace_count -= 1
				if brace_count == 0:
					# Found complete dict
					dict_str = content[start:i+1]
					dict_strs.append(dict_str)
					current_pos = i + 1
					break
			i += 1
		else:
			# No matching brace found
			break
	
	# Parse each dict string
	for dict_str in dict_strs:
		try:
			data_dict = ast.literal_eval(dict_str)
			data.append(data_dict)
		except Exception as e:
			print(f"Error parsing dict: {e}")
			continue
	
	# Extract RSSI values from all data points
	rssi_values = []
	for data_dict in data:
		rssi = extract_rssi(data_dict)
		if rssi is not None:
			rssi_values.append(rssi)
	# Create records with non-overlapping windows
	records = []
	
	# Process data in chunks of window_size
	for i in range(0, len(rssi_values), window_size):
		# Get the next window_size values
		window = rssi_values[i:i + window_size]
		
		# If we have exactly window_size values, use them
		if len(window) == window_size:
			window.append(label)
			records.append(window)
		# If we have fewer than window_size values, pad with None
		elif len(window) > 0:
			# Pad with None if needed
			window = window + [None] * (window_size - len(window))
			window.append(label)
			records.append(window)
	
	# Create DataFrame with proper headers
	headers = [str(i) for i in range(1, window_size + 1)] + ['label']
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
	parser.add_argument("--input", type=str, required=True, help="Input wavecom data file")
	parser.add_argument("--output", type=str, required=True, help="Output CSV file")
	args = parser.parse_args()
	export_to_csv(args.input, args.output, 10, args.label)