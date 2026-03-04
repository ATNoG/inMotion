#!/usr/bin/env python3
"""
Merge multiple CSV files into a single dataset.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def extract_concurrent_noise_path(filename: str, label: str) -> str | None:
	"""
	Extract the concurrent noise path from filename and row label.
	
	For noise files like 'noiseAA_BB3.csv', if label is 'AA', concurrent is 'BB'.
	If label is 'BB', concurrent is 'AA'.
	
	Pattern: noise{PATH1}_{PATH2}{NUMBER}.csv
	
	Args:
		filename: Name of the CSV file
		label: The label/path of the current row
		
	Returns:
		The concurrent noise path or None if not applicable
	"""
	# Match pattern: noise{PATH1}_{PATH2}{NUMBER}.csv (e.g., noiseAA_BB3.csv)
	# Also handles files without number suffix (e.g., noiseAA_BB.csv)
	# The paths are alphabetic (AA, AB, BA, BB), number is iteration suffix
	pattern = r'^noise([A-Za-z]+)_([A-Za-z]+)\d*\.csv$'
	match = re.match(pattern, filename)
	
	if not match:
		return None
	
	path1, path2 = match.group(1), match.group(2)
	
	# Return the other path based on the current label
	if label == path1:
		return path2
	elif label == path2:
		return path1
	else:
		return None

def merge_csv_files(input_dir: str, output_file: str, pattern: str = "*.csv") -> None:
	"""
	Merge multiple CSV files into a single dataset.
	
	Args:
		input_dir: Directory containing CSV files
		output_file: Path to output merged CSV file
		pattern: File pattern to match (default: *.csv)
	"""
	input_path = Path(input_dir)
	
	if not input_path.exists():
		print(f"Error: Input directory {input_dir} does not exist")
		return
	
	# Find all matching CSV files
	csv_files = list(input_path.glob(pattern))
	
	if not csv_files:
		print(f"No CSV files matching pattern '{pattern}' found in {input_dir}")
		return
	
	print(f"Found {len(csv_files)} CSV files to merge")
	print("-" * 60)
	
	# Read and concatenate all CSVs
	dfs = []
	for csv_file in csv_files:
		print(f"Reading: {csv_file.name}")
		try:
			df = pd.read_csv(csv_file)
			
			# Normalize column names: some files use 'noise', others use 'noise_label'
			if 'noise_label' in df.columns and 'noise' not in df.columns:
				df = df.rename(columns={'noise_label': 'noise'})
			
			# Add concurrent_noise_path column for noise files
			# Only add if noise column exists and is True for at least some rows
			if 'noise' in df.columns:
				noise_values = df['noise'].astype(str).str.lower()
				has_noise = (noise_values == 'true').any()
				
				if has_noise:
					# Extract concurrent noise path based on label and filename
					df['concurrent_noise_path'] = df['label'].apply(
						lambda lbl: extract_concurrent_noise_path(csv_file.name, str(lbl))
					)
				else:
					df['concurrent_noise_path'] = None
			else:
				df['concurrent_noise_path'] = None
			
			dfs.append(df)
			print(f"  Rows: {len(df)}, Label distribution: {df['label'].value_counts().to_dict()}")
			if 'concurrent_noise_path' in df.columns and df['concurrent_noise_path'].notna().any():
				concurrent_counts = df['concurrent_noise_path'].value_counts().to_dict()
				print(f"  Concurrent noise paths: {concurrent_counts}")
		except Exception as e:
			print(f"  Error reading {csv_file.name}: {e}")
	
	if not dfs:
		print("No data to merge")
		return
	
	# Concatenate all dataframes
	merged_df = pd.concat(dfs, ignore_index=True)
	
	# Save merged CSV
	merged_df.to_csv(output_file, index=False)
	
	print("\n" + "=" * 60)
	print(f"Merged {len(dfs)} files into {output_file}")
	print(f"Total rows: {len(merged_df)}")
	print(f"Label distribution:")
	print(merged_df['label'].value_counts().to_dict())
	print(f"  0 (failed handovers): {(merged_df['label'] == '0').sum()}")
	print(f"  1 (successful handovers): {(merged_df['label'] == '1').sum()}")
	
	# Report on concurrent noise paths
	if 'concurrent_noise_path' in merged_df.columns:
		concurrent_counts = merged_df['concurrent_noise_path'].value_counts()
		if not concurrent_counts.empty:
			print(f"\nConcurrent noise path distribution:")
			for path, count in concurrent_counts.items():
				if pd.notna(path):
					print(f"  {path}: {count} samples")
			print(f"  None (no concurrent noise): {merged_df['concurrent_noise_path'].isna().sum()} samples")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Merge multiple CSV files into a single dataset",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Merge all CSV files in data/ directory
  python merge_csv.py --input data/ --output dataset.csv
  
  # Merge only fail files
  python merge_csv.py --input data/ --output fails.csv --pattern "*_fail*.csv"
		"""
	)
	
	parser.add_argument("--input", "-i", type=str, required=True,
						help="Input directory containing CSV files")
	parser.add_argument("--output", "-o", type=str, required=True,
						help="Output merged CSV file")
	parser.add_argument("--pattern", "-p", type=str, default="*.csv",
						help="File pattern to match (default: *.csv)")
	
	args = parser.parse_args()
	merge_csv_files(args.input, args.output, args.pattern)

