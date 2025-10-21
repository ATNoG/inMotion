#!/usr/bin/env python3
"""
Merge multiple CSV files into a single dataset.
"""

import pandas as pd
import argparse
from pathlib import Path

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
			dfs.append(df)
			print(f"  Rows: {len(df)}, Label distribution: {df['label'].value_counts().to_dict()}")
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

