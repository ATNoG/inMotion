#!/usr/bin/env python3
"""
OpenWrt WiFi Client Scanner
This script connects to an OpenWrt router via SSH and uses system calls (ubus alias)
to retrieve MAC addresses and RSSI values of connected WiFi clients.
"""

import json
import subprocess
import sys
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
import argparse


class OpenWrtWiFiScanner:
    def __init__(self, router_ip: str, username: str = "root", password: Optional[str] = None, 
                 key_file: Optional[str] = None, port: int = 22, mode: str = "auto"):
        """
        Initialize the WiFi scanner for OpenWrt router.
        
        Args:
            router_ip: IP address of the OpenWrt router
            username: SSH username (default: root)
            password: SSH password (optional if using key file)
            key_file: Path to SSH private key file (optional if using password)
            port: SSH port (default: 22)
            mode: Command mode - "ubus", "system", or "auto" (default: auto)
        """
        self.router_ip = router_ip
        self.username = username
        self.password = password
        self.key_file = key_file
        self.port = port
        
        # Auto-detect mode based on username if mode is "auto"
        if mode == "auto":
            if username == "admin":
                self.mode = "system"
            else:  # root or other users default to ubus
                self.mode = "ubus"
        else:
            self.mode = mode
        # Cache for interface discovery to avoid repeated calls
        self._cached_interfaces = None
        self._interface_cache_time = 0
        self._interface_cache_timeout = 30  # Cache interfaces for 30 seconds
        
        # Cache for method discovery - which method works for each interface
        self._interface_methods = {}  # interface -> method_name mapping
        self._method_cache_timeout = 300  # Cache methods for 5 minutes
        
    def _execute_ssh_command(self, command: str) -> str:
        """
        Execute a command on the OpenWrt router via SSH.
        
        Args:
            command: Command to execute
            
        Returns:
            Command output as string
            
        Raises:
            subprocess.CalledProcessError: If SSH command fails
        """
        ssh_cmd = [
            "ssh", 
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            "-p", str(self.port)
        ]
        
        if self.key_file:
            ssh_cmd.extend(["-i", self.key_file])
        elif self.password:
            # Use sshpass if password is provided
            ssh_cmd = ["sshpass", "-p", self.password] + ssh_cmd
            
        ssh_cmd.append(f"{self.username}@{self.router_ip}")
        ssh_cmd.append(command)
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True, timeout=15)
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            print(f"SSH command timed out: {command}")
            raise
        except subprocess.CalledProcessError as e:
            print(f"SSH command failed: {e}")
            print(f"Error output: {e.stderr}")
            raise
    
    def _ubus_call(self, path: str, method: str, params: Dict = None) -> Dict:
        """
        Execute a call on the OpenWrt router using the specified mode (ubus or system).
        
        Args:
            path: ubus path
            method: ubus method
            params: Optional parameters for the call
            
        Returns:
            JSON response from the call
        """
        if params:
            params_str = json.dumps(params)
            command = f"{self.mode} call {path} {method} '{params_str}'"
        else:
            command = f"{self.mode} call {path} {method}"
        
        try:
            output = self._execute_ssh_command(command)
            try:
                return json.loads(output) if output else {}
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from {self.mode} call: {e}")
                print(f"Raw output: {output}")
                return {}
        except subprocess.CalledProcessError as e:
            print(f"{self.mode} call failed for {path} {method}: {e}")
            return {}
    
    def get_ubus_services(self) -> List[str]:
        """
        Get list of available services for debugging using the specified mode.
        
        Returns:
            List of service names
        """
        if self.mode == "system":
            command = "system show"  # system uses 'show' instead of 'list'
        else:
            command = f"{self.mode} list"
        
        try:
            output = self._execute_ssh_command(command)
            return output.split('\n') if output else []
        except Exception as e:
            print(f"Error getting {self.mode} services: {e}")
            return []
    
    def get_wireless_interfaces(self) -> List[str]:
        """
        Get list of wireless interfaces on the router.
        Uses caching to avoid repeated expensive calls.
        
        Returns:
            List of actual interface names (like phy0-ap0, phy1-ap0)
        """
        # Check if we have cached interfaces that are still valid
        current_time = time.time()
        if (self._cached_interfaces is not None and 
            current_time - self._interface_cache_time < self._interface_cache_timeout):
            return self._cached_interfaces
        
        try:
            # Get wireless status
            result = self._ubus_call("network.wireless", "status")
            
            interfaces = []
            for radio_name, radio_data in result.items():
                if radio_data.get("up", False):
                    # Extract actual interface names from the interfaces array
                    if "interfaces" in radio_data:
                        for iface in radio_data["interfaces"]:
                            if "ifname" in iface:
                                interfaces.append(iface["ifname"])
                            
            # If no specific interface names found, try to find from hostapd services
            if not interfaces:
                print("No interface names found in network.wireless, trying hostapd services...")
                services = self.get_ubus_services()
                hostapd_interfaces = [s for s in services if s.startswith('hostapd.') and s != 'hostapd']
                if hostapd_interfaces:
                    # Extract interface names from hostapd services
                    interfaces = [s.replace('hostapd.', '') for s in hostapd_interfaces]
                    print(f"Found hostapd interfaces: {interfaces}")
            
            # Cache the result
            self._cached_interfaces = interfaces
            self._interface_cache_time = current_time
            return interfaces
            
        except Exception as e:
            print(f"Error getting wireless interfaces: {e}")
            # Return cached interfaces if available, even if expired
            return self._cached_interfaces if self._cached_interfaces is not None else []
    
    def clear_interface_cache(self):
        """Clear the cached interface list to force refresh on next call."""
        self._cached_interfaces = None
        self._interface_cache_time = 0
    
    def clear_method_cache(self):
        """Clear the cached method mappings to force re-discovery."""
        self._interface_methods = {}
    
    def _get_clients_hostapd(self, interface: str) -> List[Dict]:
        """Get clients using hostapd method."""
        clients_data = self._ubus_call("hostapd." + interface, "get_clients")
        clients = []
        # Check if the call succeeded (has "clients" key, even if empty)
        if "clients" in clients_data:
            for mac, client_info in clients_data["clients"].items():
                client = {
                    "mac": mac,
                    "rssi": client_info.get("signal", "N/A"),
                    "rx_bytes": client_info.get("bytes", {}).get("rx", 0),
                    "tx_bytes": client_info.get("bytes", {}).get("tx", 0),
                    "connected_time": client_info.get("connected_time", 0),
                    "inactive_time": client_info.get("inactive_time", 0),
                    "auth": client_info.get("auth", False),
                    "assoc": client_info.get("assoc", False),
                    "authorized": client_info.get("authorized", False)
                }
                clients.append(client)
        return clients
    
    def _get_clients_iwinfo(self, interface: str) -> List[Dict]:
        """Get clients using iwinfo method."""
        iwinfo_data = self._ubus_call("iwinfo", "assoclist", {"device": interface})
        clients = []
        if "results" in iwinfo_data:
            for result in iwinfo_data["results"]:
                client = {
                    "mac": result.get("mac", ""),
                    "rssi": result.get("signal", "N/A"),
                    "rx_bytes": result.get("rx_bytes", 0),
                    "tx_bytes": result.get("tx_bytes", 0),
                    "connected_time": result.get("connected_time", 0),
                    "inactive_time": result.get("inactive_time", 0),
                    "auth": True,
                    "assoc": True,
                    "authorized": True
                }
                clients.append(client)
        return clients
    
    def get_hostapd_clients(self, interface: str, debug: bool = False) -> Dict:
        """
        Get connected clients for a specific hostapd interface.
        
        Args:
            interface: Wireless interface name
            
        Returns:
            Dictionary containing client information
        """
        try:
            return self._ubus_call("hostapd." + interface, "get_clients")
        except Exception as e:
            if debug:
                print(f"Error getting clients for interface {interface} via hostapd: {e}")
            return {}
    
    def get_iwinfo_assoclist(self, interface: str, debug: bool = False) -> Dict:
        """
        Get associated clients using iwinfo (alternative method).
        
        Args:
            interface: Wireless interface name
            
        Returns:
            Dictionary containing client information
        """
        try:
            return self._ubus_call("iwinfo", "assoclist", {"device": interface})
        except Exception as e:
            if debug:
                print(f"Error getting clients for interface {interface} via iwinfo: {e}")
            return {}
    
    def get_clients_from_proc(self, interface: str, debug: bool = False) -> List[Dict]:
        """
        Get clients by parsing /proc/net/arp and wireless stats (fallback method).
        
        Args:
            interface: Wireless interface name
            
        Returns:
            List of client dictionaries
        """
        try:
            # Get ARP table
            arp_output = self._execute_ssh_command("cat /proc/net/arp")
            
            # Get wireless station dump if available
            try:
                iw_output = self._execute_ssh_command(f"iw dev {interface} station dump")
            except:
                iw_output = ""
            
            clients = []
            
            # Parse ARP entries
            for line in arp_output.split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6 and parts[5] == interface:
                        ip_addr = parts[0]
                        mac_addr = parts[3]
                        
                        # Try to get RSSI from iw station dump
                        rssi = "N/A"
                        if mac_addr.lower() in iw_output.lower():
                            # Extract signal info from iw output
                            for iw_line in iw_output.split('\n'):
                                if mac_addr.lower() in iw_line.lower():
                                    # Look for signal line in the next few lines
                                    iw_lines = iw_output.split('\n')
                                    idx = iw_lines.index(iw_line)
                                    for i in range(idx, min(idx + 10, len(iw_lines))):
                                        if 'signal:' in iw_lines[i]:
                                            rssi = iw_lines[i].split('signal:')[1].strip().split()[0]
                                            break
                        
                        client = {
                            "mac": mac_addr,
                            "rssi": rssi,
                            "ip": ip_addr,
                            "rx_bytes": 0,
                            "tx_bytes": 0,
                            "connected_time": 0,
                            "inactive_time": 0,
                            "auth": True,
                            "assoc": True,
                            "authorized": True
                        }
                        clients.append(client)
            
            return clients
            
        except Exception as e:
            if debug:
                print(f"Error getting clients from proc for interface {interface}: {e}")
            return []
    
    def get_all_wifi_clients_optimized(self, debug: bool = False) -> Dict[str, List[Dict]]:
        """
        Optimized version that batches multiple commands into single SSH call.
        
        Returns:
            Dictionary with interface names as keys and client lists as values
        """
        all_clients = {}
        
        # Get wireless interfaces (uses caching)
        interfaces = self.get_wireless_interfaces()
        
        if not interfaces:
            if debug:
                print("No wireless interfaces found")
            return all_clients
        
        # If we have cached methods for all interfaces, use batch mode
        all_cached = all(interface in self._interface_methods for interface in interfaces)
        
        if all_cached and all(self._interface_methods[iface] == "hostapd" for iface in interfaces):
            if debug:
                print("Using optimized batch hostapd calls")
            
            # Create a single command that gets all interfaces at once
            commands = []
            for interface in interfaces:
                commands.append(f"{self.mode} call hostapd.{interface} get_clients")
            
            # Execute all commands in one SSH call separated by echo markers
            batch_command = " && ".join([f"echo '===START_{i}===' && {cmd} && echo '===END_{i}==='" 
                                       for i, cmd in enumerate(commands)])
            
            try:
                output = self._execute_ssh_command(batch_command)
                
                # Parse the batched output
                sections = output.split("===START_")
                for i, section in enumerate(sections[1:]):  # Skip first empty section
                    end_marker = f"===END_{i}==="
                    if end_marker in section:
                        json_part = section.split(end_marker)[0].strip()
                        # Remove any remaining markers that might be in the JSON
                        json_part = json_part.replace(f"{i}===", "").strip()
                        try:
                            data = json.loads(json_part)
                            interface = interfaces[i]
                            clients = []
                            if "clients" in data:
                                for mac, client_info in data["clients"].items():
                                    client = {
                                        "mac": mac,
                                        "rssi": client_info.get("signal", "N/A"),
                                        "rx_bytes": client_info.get("bytes", {}).get("rx", 0),
                                        "tx_bytes": client_info.get("bytes", {}).get("tx", 0),
                                        "connected_time": client_info.get("connected_time", 0),
                                        "inactive_time": client_info.get("inactive_time", 0),
                                        "auth": client_info.get("auth", False),
                                        "assoc": client_info.get("assoc", False),
                                        "authorized": client_info.get("authorized", False)
                                    }
                                    clients.append(client)
                            all_clients[interface] = clients
                        except json.JSONDecodeError as e:
                            if debug:
                                print(f"Failed to parse JSON for interface {interfaces[i]}: {e}")
                                print(f"Raw JSON part: '{json_part}'")
                            all_clients[interfaces[i]] = []
                
                return all_clients
                
            except Exception as e:
                if debug:
                    print(f"Batch command failed: {e}, falling back to individual calls")
                # Fall back to individual calls
        
        # Fall back to the original method
        return self.get_all_wifi_clients_original(debug)
    
    def get_all_wifi_clients_original(self, debug: bool = False) -> Dict[str, List[Dict]]:
        """
        Get all WiFi clients connected to all access points.
        Uses cached method discovery for maximum performance.
        
        Returns:
            Dictionary with interface names as keys and client lists as values
        """
        all_clients = {}
        
        # Get wireless interfaces (uses caching)
        interfaces = self.get_wireless_interfaces()
        
        if not interfaces:
            if debug:
                print("No wireless interfaces found")
            return all_clients
        
        for interface in interfaces:
            if debug:
                print(f"Scanning interface: {interface}")
            
            clients = []
            
            # Check if we know which method works for this interface
            if interface in self._interface_methods:
                method = self._interface_methods[interface]
                if debug:
                    print(f"  Using cached method '{method}' for {interface}")
                
                try:
                    if method == "hostapd":
                        clients = self._get_clients_hostapd(interface)
                    elif method == "iwinfo":
                        clients = self._get_clients_iwinfo(interface)
                    elif method == "proc":
                        clients = self.get_clients_from_proc(interface, debug)
                    
                    # If we get here, the cached method worked
                    all_clients[interface] = clients
                    continue
                    
                except Exception as e:
                    if debug:
                        print(f"  Cached method '{method}' error: {e}")
                    # Remove failed method from cache
                    del self._interface_methods[interface]
            
            # Method discovery phase - try methods in order of reliability/speed
            methods_to_try = [
                ("hostapd", self._get_clients_hostapd),
                ("iwinfo", self._get_clients_iwinfo),
            ]
            
            # Only try proc method in debug mode (it's slow)
            if debug:
                methods_to_try.append(("proc", self.get_clients_from_proc))
            
            method_found = False
            for method_name, method_func in methods_to_try:
                if debug:
                    print(f"  Trying {method_name} method for {interface}")
                
                try:
                    if method_name == "proc":
                        clients = method_func(interface, debug)
                    else:
                        clients = method_func(interface)
                    
                    # For hostapd and iwinfo, success means no exception was thrown
                    # (empty client list is still a successful response)
                    if method_name in ["hostapd", "iwinfo"]:
                        # Check if we got a proper response structure
                        method_data = None
                        if method_name == "hostapd":
                            method_data = self._ubus_call("hostapd." + interface, "get_clients")
                            if "clients" in method_data:
                                method_found = True
                        elif method_name == "iwinfo":
                            method_data = self._ubus_call("iwinfo", "assoclist", {"device": interface})
                            if "results" in method_data:
                                method_found = True
                        
                        if method_found:
                            # Cache this working method
                            self._interface_methods[interface] = method_name
                            if debug:
                                print(f"  Success! Cached {method_name} method for {interface}")
                            break
                    else:
                        # For proc method, only consider it successful if it returns data or in debug mode
                        if clients or method_name == "proc":
                            self._interface_methods[interface] = method_name
                            if debug:
                                print(f"  Success! Cached {method_name} method for {interface}")
                            method_found = True
                            break
                        
                except Exception as e:
                    if debug:
                        print(f"  {method_name} method failed for {interface}: {e}")
                    continue
            
            if not method_found:
                if debug:
                    print(f"  No working method found for {interface}")
                # Cache that no method works to avoid future attempts
                self._interface_methods[interface] = "none"
            
            all_clients[interface] = clients
            
        return all_clients
    
    def get_all_wifi_clients(self, debug: bool = False) -> Dict[str, List[Dict]]:
        """
        Get all WiFi clients - automatically chooses optimized or fallback method.
        """
        return self.get_all_wifi_clients_optimized(debug)
    
    def get_wireless_scan(self, interface: str) -> List[Dict]:
        """
        Perform a wireless scan to detect nearby devices.
        
        Args:
            interface: Wireless interface to scan with
            
        Returns:
            List of detected wireless devices
        """
        try:
            # Trigger scan
            self._ubus_call("iwinfo", "scan", {"device": interface})
            
            # Get scan results
            scan_result = self._ubus_call("iwinfo", "scanlist", {"device": interface})
            
            devices = []
            if "results" in scan_result:
                for device in scan_result["results"]:
                    device_info = {
                        "bssid": device.get("bssid", ""),
                        "ssid": device.get("ssid", ""),
                        "signal": device.get("signal", "N/A"),
                        "quality": device.get("quality", "N/A"),
                        "encryption": device.get("encryption", {})
                    }
                    devices.append(device_info)
                    
            return devices
        except Exception as e:
            print(f"Error performing wireless scan on {interface}: {e}")
            return []
    
    def display_connected_clients(self, all_clients: Dict[str, List[Dict]],
                                  clear_screen: bool = False, show_timestamp: bool = True):
        """
        Display connected clients in a formatted way.
        
        Args:
            all_clients: Dictionary of clients per interface
            clear_screen: Whether to clear the screen before displaying
            show_timestamp: Whether to show timestamp in header
        """
        if clear_screen:
            os.system('clear' if os.name == 'posix' else 'cls')
        
        header = "CONNECTED WIFI CLIENTS"
        if show_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header += f" - Last Update: {timestamp}"
        
        print("\n" + "="*80)
        print(header)
        print("="*80)
        
        total_clients = 0
        
        for interface, clients in all_clients.items():
            print(f"\nInterface: {interface}")
            print("-" * 50)
            
            if not clients:
                print("No clients connected")
                continue
                
            print(f"{'MAC Address':<18} {'RSSI':<6} {'RX Bytes':<12} {'TX Bytes':<12} {'Status'}")
            print("-" * 70)
            
            for client in clients:
                status_flags = []
                if client["auth"]: status_flags.append("AUTH")
                if client["assoc"]: status_flags.append("ASSOC")
                if client["authorized"]: status_flags.append("AUTHORIZED")
                status = ",".join(status_flags) if status_flags else "UNKNOWN"
                
                print(f"{client['mac']:<18} {client['rssi']:<6} "
                      f"{client['rx_bytes']:<12} {client['tx_bytes']:<12} {status}")
                
            total_clients += len(clients)
            
        print(f"\nTotal connected clients: {total_clients}")
    
    def monitor_clients(self, refresh_interval: float = 5.0, debug: bool = False, file:str= None):
        """
        Continuously monitor WiFi clients with auto-refresh.
        
        Args:
            refresh_interval: Time in seconds between refreshes
            debug: Whether to show debug messages
        """
        print(f"Starting WiFi client monitoring (refresh every {refresh_interval}s)")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                start_time = time.time()
                
                # Get current clients
                all_clients = self.get_all_wifi_clients(debug)
                with open(file, "a") as f:
                    f.write(f"{datetime.now().timestamp()} {str(all_clients)}\n")

                
                scan_time = time.time() - start_time
                
                # Display with clear screen and timestamp
                self.display_connected_clients(all_clients, clear_screen=True, show_timestamp=True)
                
                if debug:
                    print(f"\nScan #{iteration} completed in {scan_time:.2f} seconds")
                    print(f"Cached methods: {self._interface_methods}")
                
                print(f"\nRefreshing in {refresh_interval} seconds... (Press Ctrl+C to stop)")
                
                # Wait for the specified interval
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Scan WiFi clients on OpenWrt router",
        epilog="Examples:\n"
               "  %(prog)s 192.168.1.1 -u root -p password     # Auto-detects ubus for root user\n"
               "  %(prog)s 192.168.1.1 -u admin -p password    # Auto-detects system for admin user\n"
               "  %(prog)s 192.168.1.1 -u admin -p password --mode system  # Explicit system mode\n"
               "  %(prog)s 192.168.1.1 -u root -p password --port 2222 -i 5  # Root with custom port",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("router_ip", help="IP address of the OpenWrt router")
    parser.add_argument("-u", "--username", default="root", help="SSH username (default: root)")
    parser.add_argument("-p", "--password", help="SSH password")
    parser.add_argument("-k", "--key-file", help="SSH private key file")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("-f", "--file", type=str)
    parser.add_argument("--mode", choices=["ubus", "system", "auto"], default="auto",
                       help="Command mode: 'ubus' for standard ubus commands, 'system' for system alias, or 'auto' to detect based on username (default: auto)")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear method cache to force re-discovery of working methods")
    parser.add_argument("-s", "--scan", action="store_true",
                       help="Also perform wireless scan for nearby devices (single scan mode only)")
    parser.add_argument("-d", "--debug", action="store_true",
                       help="Enable debug mode to show available services and interfaces")
    parser.add_argument("-i", "--interval", type=float, metavar="SECONDS",
                       help="Enable auto-refresh mode with specified interval in seconds (min: 1)")


    args = parser.parse_args()
    
    if not args.password and not args.key_file:
        print("Error: Either password (-p) or key file (-k) must be provided")
        sys.exit(1)
    
    try:
        # Initialize scanner
        scanner = OpenWrtWiFiScanner(
            router_ip=args.router_ip,
            username=args.username,
            password=args.password,
            key_file=args.key_file,
            port=args.port,
            mode=args.mode
        )
        
        # Clear caches if requested
        if args.clear_cache:
            scanner.clear_method_cache()
            scanner.clear_interface_cache()
            print("Caches cleared - will re-discover interfaces and methods")
        
        # Get connected clients
        print(f"Connecting to OpenWrt router at {args.router_ip}...")
        print(f"Using command mode: {scanner.mode} (auto-detected from username: {args.username})" if args.mode == "auto" else f"Using command mode: {scanner.mode}")
        
        # Debug mode - show available services and interfaces
        if args.debug:
            print("\n" + "="*80)
            print("DEBUG INFORMATION")
            print("="*80)
            
            print(f"\nUsing command mode: {args.mode}")
            print(f"\nAvailable {args.mode} services:")
            services = scanner.get_ubus_services()
            for service in services[:20]:  # Limit output
                print(f"  {service}")
            if len(services) > 20:
                print(f"  ... and {len(services) - 20} more services")
            
            print(f"\nTotal {args.mode} services found: {len(services)}")
            
            # Show wireless-related services
            wireless_services = [s for s in services if 'wireless' in s.lower() or 'hostapd' in s.lower() or 'iwinfo' in s.lower()]
            if wireless_services:
                print(f"\nWireless-related services:")
                for service in wireless_services:
                    print(f"  {service}")
        
        # Check if auto-refresh mode is enabled
        if args.interval:
            # if args.interval < 1:
            #     print("Error: Refresh interval must be at least 1 second")
            #     sys.exit(1)
            
            if args.scan:
                print("Warning: Wireless scan (-s) is not supported in auto-refresh mode")
            
            # Start monitoring mode
            scanner.monitor_clients(refresh_interval=args.interval, debug=args.debug, file=args.file)
        else:
            # Single scan mode
            all_clients = scanner.get_all_wifi_clients(debug=args.debug)
            
            # Display results
            scanner.display_connected_clients(all_clients)
        
            # Optionally perform wireless scan (only in single scan mode)
            if args.scan:
                print("\n" + "="*80)
                print("WIRELESS SCAN RESULTS")
                print("="*80)
                
                interfaces = scanner.get_wireless_interfaces()
                for interface in interfaces:
                    print(f"\nScanning with interface: {interface}")
                    scan_results = scanner.get_wireless_scan(interface)
                    
                    if scan_results:
                        print(f"{'BSSID':<18} {'SSID':<20} {'Signal':<8} {'Quality'}")
                        print("-" * 60)
                        for device in scan_results:
                            print(f"{device['bssid']:<18} {device['ssid']:<20} "
                                  f"{device['signal']:<8} {device['quality']}")
                    else:
                        print("No devices found in scan")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()