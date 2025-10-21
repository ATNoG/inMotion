# 1: 1761055449.663832 {'phy0-ap0': [], 'phy1-ap0': [{'mac': 'e6:53:5c:2a:e8:e2', 'rssi': -51, 'rx_bytes': 259437, 'tx_bytes': 86091, 'connected_time': 0, 'inactive_time': 0, 'auth': True, 'assoc': True, 'authorized': True}]}
# 36: 1760454981.805809 {'phy0-ap0': [{'mac': '76:64:61:f6:e6:14', 'rssi': -53, 'rx_bytes': 50914, 'tx_bytes': 23489, 'connected_time': 0, 'inactive_time': 0, 'auth': True, 'assoc': True, 'authorized': True}], 'phy1-ap0': [{'mac': '76:64:61:f6:e6:14', 'rssi': 'N/A', 'rx_bytes': 0, 'tx_bytes': 0, 'connected_time': 0, 'inactive_time': 0, 'auth': True, 'assoc': False, 'authorized': False}]}
# 37: 1760454994.640037 {'phy0-ap0': [{'mac': '76:64:61:f6:e6:14', 'rssi': -31, 'rx_bytes': 70540, 'tx_bytes': 31391, 'connected_time': 0, 'inactive_time': 0, 'auth': True, 'assoc': True, 'authorized': True}], 'phy1-ap0': []}
# 34: 1760454863.122356 {'phy0-ap0': [{'mac': '76:64:61:f6:e6:14', 'rssi': 'N/A', 'rx_bytes': 0, 'tx_bytes': 0, 'connected_time': 0, 'inactive_time': 0, 'auth': True, 'assoc': False, 'authorized': False}], 'phy1-ap0': [{'mac': '76:64:61:f6:e6:14', 'rssi': -64, 'rx_bytes': 40423, 'tx_bytes': 29715, 'connected_time': 0, 'inactive_time': 0, 'auth': True, 'assoc': True, 'authorized': True}]}
import pandas as pd
import json
import argparse
import ast

def export_to_csv(input_file: str, output_file: str) -> None:
	data = [] # list of dicts
	with open(input_file, "r") as f:
		for line in f:
			data.append(line.split(maxsplit=2)[2].strip())
   

	records = []

	for line in data:
		data_dict = ast.literal_eval(line.replace("'", "\""))
		if data_dict['phy0-ap0'] and data_dict['phy0-ap0'][0]['rssi'] != 'N/A':
			records.append(data_dict['phy0-ap0'][0]['rssi'])
		elif data_dict['phy1-ap0'] and data_dict['phy1-ap0'][0]['rssi'] != 'N/A':
			records.append(data_dict['phy1-ap0'][0]['rssi'])
		else:
			records.append('PIXA')

	# Create DataFrame and export to CSV
	df = pd.DataFrame(records)
	df.to_csv(output_file, index=False, header=None)
 
if __name__ == "__main__":
    # should receive the file flag and output flag
	parser = argparse.ArgumentParser(description="Export wavecom data to CSV")
	
	parser.add_argument("--i", type=str, required=True, help="Input wavecom data file")
	parser.add_argument("--o", type=str, required=True, help="Output CSV file")
	args = parser.parse_args()
	export_to_csv(args.i, args.o)
