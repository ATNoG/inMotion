from datetime import datetime
from typing import Any
import ast
import matplotlib.pyplot as plt

def main():
    # data is ts /s dict
    data = []
    with open("data/fileBtoAtoC2.txt", "r") as f:
        for line in f:
            data.append(line.strip())

    # por simplificacao a interface será sempre a mesma phy1-ap0 e o mac sempre o mesmo e6:53:5c:2a:e8:e2
    # 1759847189.87959 {'phy0-ap0': [], 'phy1-ap0': [{'mac': 'e6:53:5c:2a:e8:e2', 'rssi': -52, 'rx_bytes': 55311, 'tx_bytes': 26139, 'connected_time': 0, 'inactive_time': 0, 'auth': True, 'assoc': True, 'authorized': True}]}

    splitted_data: tuple[float, dict[str, list[dict[str, Any]]]] = [(float(line.split(maxsplit=1)[0]), ast.literal_eval(line.split(maxsplit=1)[1])) for line in data]

    timestamps = [datetime.fromtimestamp(item[0]) for item in splitted_data]  # converte timestamp para hora
    rssi_values = [item[1]['phy1-ap0'][0]['rssi'] for item in splitted_data] # this is logaritmic scale please do it

    plt.plot(timestamps, rssi_values)
    plt.grid()
    plt.title("RSSI/Time")
    plt.xlabel("Time")
    plt.ylabel("RSSI")
    plt.legend( ["RSSI"], loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()
