import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import psutil
from utils.os_utils import file_operations
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()

# Constants
TAILSCALE_IPS_STARTWITH = "100.64.0."
ASD_CONTAINER_FOLDER = "/opt/asd"

# AWS Deployment Tracking file
home_dir: Path = Path.home()
asd_deployment_tracking_file: Path = home_dir / ".asd_container_uuid"


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


def start_tailscale_connect_process(tailscale_state_screen_name: str = "tailscale_state") -> bool:
    """
    Start the Tailscale connection process.

    This function searches for existing Tailscale-related processes. If Tailscale is not running,
    it starts the Tailscale connection process by executing a shell script.

    If any tailscale processes are running but unable to connect to the control server, the function
    will attempt to terminate any existing Tailscale sessions running in a 'screen' context and start
    the connection process from scratch.

    The function relies on the global variables `tailscale_connect_shell_script`
    and `asd_aws_deployment_counter` to construct the command for starting Tailscale.

    Parameters:
        - tailscale_state_screen_name (str): linux command 'screen' name defined in tailscale_connect_service.sh˛

    Returns:
        - bool: True if the Tailscale service started successfully (indicated by a return code of 0), False otherwise.
    """
    # List (boolean) to keep track of tailscale system-level processes
    is_tailscale_running = []

    # Define Tailscale bash script location
    asd_container_folder = Path(ASD_CONTAINER_FOLDER)
    _tailscale_connect_shell_script = asd_container_folder.rglob("tailscale_connect_service.sh")
    tailscale_connect_shell_script = [shell_script_file for shell_script_file in _tailscale_connect_shell_script]
    if not tailscale_connect_shell_script:
        logging.error("Shell script not found!")
        return False

    # Change the permission of the shell script to make it executable
    subprocess.run(
        ["chmod", "+x", str(tailscale_connect_shell_script[0])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Read and parse the deployment tracking file
    asd_deployment_tracking_dict = json.loads(file_operations(file_path=asd_deployment_tracking_file, mode="read"))
    asd_aws_deployment_counter = asd_deployment_tracking_dict["asd_deployment_number"]

    # Search for all tailscale-related system processes
    for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            process_info = process.info
            process_name = process_info["name"]
            cmdline = process_info["cmdline"]
            if "tailscaled" in process_name or any("tailscaled" in cmd for cmd in cmdline):
                logging.info(f"+++ Tailscale is already running: PID {process_info['pid']}, Name {process_name} +++")
                is_tailscale_running.append(True)
            else:
                is_tailscale_running.append(False)
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ) as error:
            logging.warning(f"!!! Unable to find any tailscale-related processes: {error} !!!")

    # if/else logic to handle edge cases where tailscale is running but control server (headscale) is unreachable
    if any(is_tailscale_running):
        # Check if tailscale is running and retrieve tailscale healthcheck information
        tailscale_status = subprocess.run(
            ["/usr/bin/tailscale", "status", "--json"], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

        try:
            tailscale_healthcheck: list = json.loads(tailscale_status.stdout)["Health"]
        except json.JSONDecodeError as error_msg:
            tailscale_healthcheck: list = ["not available"]
            logging.error(f"!!! Tailscale is not available !!!\nERROR: {error_msg}")

        # Deals with the healthcheck information obtained by running 'tailscale status --json', searches for failed keywords in the 'Health' JSON key
        if tailscale_healthcheck:
            healthcheck_keyword_errors: set = {"not", "error", "failed"}
            healthcheck_nok = [
                error_keyword in healthcheck_msg
                for healthcheck_msg in tailscale_healthcheck
                for error_keyword in healthcheck_keyword_errors
            ]
            tailscale_failed_healthchecks: bool = any(healthcheck_nok)
        else:
            tailscale_failed_healthchecks: bool = False

        tailscale_status_ok: bool = tailscale_status.returncode == 0 and tailscale_healthcheck is None
        tailscale_status_nok: bool = tailscale_status.returncode != 0 or tailscale_failed_healthchecks

        if tailscale_status_ok:
            logging.info("+++ tailscale is healthy and connected to control server (headscale) +++")
            logging.debug(f"\n### tailscale_status_ok LOG_MSG: {json.loads(tailscale_status.stdout)} ###")
            logging.info(f"+++ local tailscale IPv4: {json.loads(tailscale_status.stdout)['TailscaleIPs']} +++")
            return True
        elif tailscale_status_nok:
            logging.info("+++ terminating any existing instance of tailscale 'screen' command...")
            subprocess.run(
                ["screen", "-XS", tailscale_state_screen_name, "quit"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(2)
            # Start the shell script to initiate the Tailscale connection
            logging.info("+++ Starting tailscale...")
            start_tailscale_code = subprocess.run(
                [
                    "bash",
                    "-c",
                    str(tailscale_connect_shell_script[0]),
                    f"headscale-asd-{asd_aws_deployment_counter}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(2)
            if start_tailscale_code.returncode == 0:
                logging.info(
                    f"+++ tailscale was successfully started. exit code: {start_tailscale_code.returncode} +++"
                )
                return True
            else:
                logging.error(
                    f"!!! tailscale failed to start. exit code: {start_tailscale_code.returncode}. start_tailscale_code stderr: {start_tailscale_code.stderr} !!!"
                )
                return False
        else:
            logging.error(
                f"+++ Unable to determine tailscale status +++\ntailscale stdout: {tailscale_status.stdout}\ntailscale stderr: {tailscale_status.stderr}"
            )
            return False
    elif not any(is_tailscale_running):
        # Start the shell script to initiate the Tailscale connection
        logging.info("+++ Tailscale was not running... starting tailscale...")
        start_tailscale_code = subprocess.run(
            [
                "bash",
                "-c",
                str(tailscale_connect_shell_script[0]),
                f"headscale-asd-{asd_aws_deployment_counter}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(2)
        if start_tailscale_code.returncode == 0:
            logging.info(f"+++ tailscale was successfully started. exit code: {start_tailscale_code.returncode} +++")
            return True
        else:
            logging.error(
                f"!!! tailscale failed to start. exit code: {start_tailscale_code.returncode}. start_tailscale_code stderr: {start_tailscale_code.stderr} !!!"
            )
            return False


def bootstrap_tailscale() -> bool:
    """
    End-to-end bootstrap of the tailscale connection process

    This function invokes other functions like 'start_tailscale_connect_process()',
    'get_network_interfaces()', 'get_tailscale_ipv4()' and logs the tailscale Ipv4 assigned to
    the local interface

    Returns:
        bool: True if the Tailscale connection and Ipv4 assignment were successful, False otherwise.
    """

    # Start the tailscale connection process
    start_tailscale_connect_process()

    # Fetch network interfaces and their IPv4 addresses
    interfaces = get_network_interfaces()

    # Extract tailscale interface's IPv4 address if available
    tailscale_ipv4 = get_tailscale_ipv4(interfaces)

    if tailscale_ipv4:
        logging.info(f"+++ Tailscale IPv4: {str(tailscale_ipv4)} +++")
        return True
    else:
        logging.info("!!! tailscale0 interface not found or does not have an IPv4 address !!!")
        return False


if __name__ == "__main__":
    bootstrap_tailscale()
