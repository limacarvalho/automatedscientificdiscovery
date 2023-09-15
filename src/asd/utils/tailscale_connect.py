import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List

import psutil

# Constants
TAILSCALE_IPS_STARTWITH = "100.64.0."
ASD_CONTAINER_FOLDER = "/opt/asd"

# Configure logging
logging.basicConfig(level=logging.INFO)


def get_network_interfaces() -> Dict[str, str]:
    """
    Get a dictionary of network interfaces along with their IPv4 addresses.

    Returns:
        dict: Dictionary with interface names as keys and their IPv4 addresses as values.
    """
    network_interfaces = {}

    # Loop through all network interfaces and their addresses
    for interface_name, addrs in psutil.net_if_addrs().items():
        # Filter addresses for IPv4 (AF_INET family code is 2)
        interface_ipv4 = [addr.address for addr in addrs if addr.family == 2]

        # If IPv4 address is found for the interface, add it to the result dictionary
        if interface_ipv4:
            network_interfaces[interface_name] = interface_ipv4[0]

    return network_interfaces


def get_tailscale_ipv4(network_interfaces: Dict[str, str]) -> Optional[str]:
    """
    Get the IPv4 address for the tailscale interface if available.

    Args:
        network_interfaces (dict): Dictionary with network interface names and their IPv4 addresses.

    Returns:
        Optional[str]: IPv4 address for the tailscale interface or None if not found.
    """
    return network_interfaces.get("tailscale0", None)


def start_tailscale_connect_process() -> bool:
    """
    Start the Tailscale connection process.

    This function searches for existing Tailscale-related processes. If Tailscale is not running,
    it starts the Tailscale connection process by executing a shell script.

    Returns:
        bool: True if the Tailscale process was started successfully, False otherwise.
    """
    is_tailscale_running = []

    # Search for all tailscale-related system processes
    for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            process_info = process.info
            process_name = process_info["name"]
            cmdline = process_info["cmdline"]
            if "tailscaled" in process_name or any("tailscaled" in cmd for cmd in cmdline):
                logging.info(
                    f"+++ Tailscale is already running: PID {process_info['pid']}, Name {process_name} +++"
                )
                is_tailscale_running.append(True)
            else:
                is_tailscale_running.append(False)
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ) as error:
            logging.warning(
                f"!!! Unable to find any tailscale-related processes: {error} !!!"
            )

    if not any(is_tailscale_running):
        asd_container_folder = Path(ASD_CONTAINER_FOLDER)
        _tailscale_connect_shell_script = asd_container_folder.rglob(
            "tailscale_connect_service.sh"
        )
        tailscale_connect_shell_script = [
            shell_script_file for shell_script_file in _tailscale_connect_shell_script
        ]
        if not tailscale_connect_shell_script:
            logging.error("Shell script not found!")
            return False

        # Change the permission of the shell script to make it executable
        chmod_shell_script = subprocess.run(
            ["chmod", "+x", str(tailscale_connect_shell_script[0])],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Start the shell script to initiate the Tailscale connection
        start_shell_script = subprocess.run(
            ["bash", "-c", str(tailscale_connect_shell_script[0])],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if chmod_shell_script.returncode == 0 and start_shell_script.returncode == 0:
            # Check the status of Tailscale after starting the process
            tailscale_status = subprocess.run(
                ["tailscale", "status"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logging.info(
                f"+++ Tailscale process was started successfully +++\n{tailscale_status.stdout}"
            )
            return True
        else:
            logging.error(
                f"!!! ERROR: Unable to start Tailscale process !!!\n{chmod_shell_script.stdout}\n{start_shell_script.stderr}"
            )
            return False
    return False


if __name__ == "__main__":
    # Start the tailscale connection process
    start_tailscale_connect_process()

    # Fetch network interfaces and their IPv4 addresses
    interfaces = get_network_interfaces()

    # Extract tailscale interface's IPv4 address if available
    tailscale_ipv4 = get_tailscale_ipv4(interfaces)

    if tailscale_ipv4:
        print(tailscale_ipv4)
    else:
        print("tailscale0 interface not found or does not have an IPv4 address.")
