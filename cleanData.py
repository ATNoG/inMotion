def clean_wavecom_file(input_file: str, output_file: str) -> None:
	with open(input_file, "r") as f:
		lines = f.readlines()

	cleaned_lines = []
	for i in range(0, 450, 15):
		cleaned_lines.extend(lines[i+3:i+13])

	for i in range(450, len(lines)):
		cleaned_lines.append(lines[i])

	with open(output_file, "w") as f:
		f.writelines(cleaned_lines)
  
if __name__ == "__main__":
	clean_wavecom_file("wavecom_files/routeAtoB.txt", "wavecom_files/cleanRouteAtoB.txt")