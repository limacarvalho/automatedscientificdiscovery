import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

# Constants
MAX_RETRIES = 3  # Maximum number of retries for subprocess calls

# Configure logging
logging.basicConfig(level=logging.INFO)


def delete_dir(folder_path: str, retries: int = MAX_RETRIES) -> Tuple[bool, str]:
    """
    Forces deletion of a directory using the 'rm -rf' command.

    Parameters:
    - folder_path (str): The path of the directory to be deleted.
    - retries (int): The number of times the deletion operation should be retried in case of failure.

    Returns:
    - Tuple[bool, str]: A tuple containing a boolean indicating the success status and a string message with the result.

    Example:
    - delete_dir("/root/testdirectory")
    """
    for attempt in range(retries):
        try:
            # Using subprocess to delete the folder
            result = subprocess.run(
                ["rm", "-rf", folder_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check if the process was successful
            if result.returncode == 0:
                logging.info(f"+++ Successfully deleted directory: '{folder_path}' +++")
                return True, f"+++ Successfully deleted directory: '{folder_path}' +++"

            # If there was an error, log it and retry
            logging.warning(
                f"!!! Attempt {attempt + 1}: Error deleting '{folder_path}': {result.stderr}"
            )

        except Exception as error:
            # Handle exceptions related to the subprocess call
            logging.error(
                f"!!! Exception while trying to delete '{folder_path}': {error}"
            )

    # If reached here, deletion was not successful after all retries
    return (
        False,
        f"!!! Failed to delete directory: '{folder_path}' after {retries} attempts.",
    )


def write_aws_credentials_to_file(
    aws_folder: str,
    aws_credentials_file: str,
    aws_config_file: str,
    account_id_file: str,
    account_id: str,
    aws_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str = "",
) -> None:
    """
    Writes AWS credentials and configurations to respective files.

    Parameters:
    - aws_folder (str): Filesystem Path to folder (normally at /root/.aws)
    - aws_credentials_file (str): Filesystem Path to AWS Credentials file
    - aws_config_file (str): Filesystem Path to AWS Config file
    - account_id_file (str): Filesystem Path to a custom file containing the AWS Account Id
    - account_id (str): AWS Account Id
    - aws_region (str): AWS region identifier (eg. us-east-1)
    - aws_access_key_id (str): AWS access key ID.
    - aws_secret_access_key (str): AWS secret access key.
    - aws_session_token (str, optional): AWS session token. Defaults to an empty string.

    Returns:
    - None
    """
    try:
        # Create the AWS directory if it doesn't exist
        aws_folder.mkdir(parents=True, exist_ok=True)

        # Write credentials to the credentials file
        with aws_credentials_file.open("w") as file_obj:
            file_obj.write("[default]\n")
            file_obj.write(f"aws_access_key_id={aws_access_key_id}\n")
            file_obj.write(f"aws_secret_access_key={aws_secret_access_key}\n")
            if aws_session_token:
                file_obj.write(f"aws_session_token={aws_session_token}\n")

        # Write default region and output format to the config file
        with aws_config_file.open("w") as file_obj:
            file_obj.write("[default]\n")
            file_obj.write(f"region = {aws_region}\n")
            file_obj.write(f"output = json\n")

        if account_id:
            # Write AWS AccountId to the 'account_id' file
            with account_id_file.open("w") as file_obj:
                file_obj.write(account_id)
        else:
            # Using subprocess retrieve AWS Account Id by running 'aws sts get-caller-identity'
            account_id_command = subprocess.run(
                ["aws", "sts", "get-caller-identity", "--output", "json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if account_id_command.returncode == 0:
                account_id = json.loads(account_id_command.stdout)["Account"]
                # Write AWS AccountId to the 'account_id' file
                with account_id_file.open("w") as file_obj:
                    file_obj.write(account_id)
            else:
                logging.error(
                    f"!!! Error: Unable to retrieve AWS Account Id: {account_id_command.stderr} !!!"
                )

        logging.info("+++ AWS credentials and configurations written successfully +++")

    except Exception as error:
        logging.error(f"!!! Error writing AWS credentials/configurations: {error} !!!")


def read_aws_credentials_from_file(aws_folder: str) -> Dict[str, Optional[str]]:
    """
    Reads the AWS credentials file and creates a Python Dictionary containing the AWS Access Key, Secret Access Key, and Session Token (if any).

    Parameters:
    - aws_folder (str): Filesystem Path to folder (typically at /root/.aws)

    Returns:
    - dict: Dictionary containing AWS Access Key, Secret Access Key, and Session Token.
    """

    credentials_file = os.path.join(aws_folder, "credentials")

    # Ensure the credentials file exists
    if not os.path.exists(credentials_file):
        raise FileNotFoundError(
            f"!!! The AWS credentials file was not found at {credentials_file} !!!"
        )

    aws_credentials = {}
    try:
        with open(credentials_file, "r") as file_obj:
            for line in file_obj:
                line = line.strip()
                # Parsing the file for AWS credentials
                if "aws_access_key_id" in line:
                    aws_credentials["aws_access_key_id"] = line.split("=")[1].strip()
                elif "aws_secret_access_key" in line:
                    aws_credentials["aws_secret_access_key"] = line.split("=")[
                        1
                    ].strip()
                elif "aws_session_token" in line:
                    aws_credentials["aws_session_token"] = line.split("=")[1].strip()
                elif "aws_session_token" not in line:
                    aws_credentials["aws_session_token"] = ""
    except Exception as error:
        raise Exception(f"!!! Error reading the AWS credentials file: {error} !!!")
    else:
        return aws_credentials
