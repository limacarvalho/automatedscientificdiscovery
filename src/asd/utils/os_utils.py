import re
import json
import uuid
import base64
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Constants
MAX_RETRIES = 3  # Maximum number of retries for subprocess calls

# Configure logging
logging.basicConfig(level=logging.INFO)


def delete_dir(folder_path: str, retries: int = MAX_RETRIES) -> Tuple[bool, str]:
    """
    Forces deletion of a directory using the 'rm -rf' command.

    Args:
        folder_path (str): Path of the directory to be deleted.
        retries (int, optional): Number of times the deletion operation should be retried in case of failure. Defaults to MAX_RETRIES.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating the success status and a string message with the result.

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
    aws_folder: Union[str, Path],
    aws_credentials_file: Union[str, Path],
    aws_config_file: Union[str, Path],
    account_id_file: Union[str, Path],
    account_id: str,
    aws_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = "",
) -> None:
    """
    Writes AWS credentials and configurations to respective files.

    Args:
        aws_folder (Union[str, Path]): Directory path (typically at /root/.aws).
        aws_credentials_file (Union[str, Path]): Path to AWS Credentials file.
        aws_config_file (Union[str, Path]): Path to AWS Config file.
        account_id_file (Union[str, Path]): Path to a custom file containing the AWS Account Id.
        account_id (str): AWS Account Id.
        aws_region (str): AWS region identifier (e.g., us-east-1).
        aws_access_key_id (str): AWS access key ID.
        aws_secret_access_key (str): AWS secret access key.
        aws_session_token (Optional[str], optional): AWS session token. Defaults to an empty string.

    Raises:
        Exception: If any error occurs during the file write operation.
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


def read_aws_credentials_from_file(aws_folder: Union[str, Path]) -> Dict[str, Optional[str]]:
    """
    Reads the AWS credentials file and returns a dictionary containing the AWS Access Key, Secret Access Key, and Session Token (if any).

    Args:
        aws_folder (Union[str, Path]): Directory path (typically at /root/.aws).

    Returns:
        Dict[str, Optional[str]]: Dictionary containing AWS Access Key, Secret Access Key, and Session Token.

    Raises:
        FileNotFoundError: If the AWS credentials file does not exist.
        Exception: If any error occurs during the file read operation.
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


def encode_file_to_base64(filepath: Union[Path, str], retries: int = 3) -> Optional[str]:
    """
    Convert file contents to a base64 encoded string.

    Args:
    - filepath (Union[Path, str]): The path of the file to be encoded.
    - retries (int, optional): Number of retries in case of a failure. Defaults to 3.

    Returns:
    - Optional[str]: The base64 encoded string of the file's content. Returns None if unsuccessful.
    """

    for attempt in range(retries):
        try:
            # Ensure the file path is a Path object
            if not isinstance(filepath, Path):
                filepath = Path(filepath)
            
            # Open the file in binary mode and read its contents
            with filepath.open('rb') as file:
                file_content = file.read()

            # Convert the file content to a base64 encoded string
            encoded_string = base64.b64encode(file_content).decode('utf-8')
            return encoded_string

        except Exception as error:
            # Log the exception for debugging purposes
            # Depending on your context, you might replace the print statement with proper logging
            logging.error(f"!!! Error while encoding the file: {error}. Attempt {attempt + 1} of {retries} !!!")

    # If the function reaches this point, all retries have been exhausted.
    logging.error(f"!!! Failed to encode the file after {retries} attempts !!!")
    return None


def generate_uuid():
    """
    Generate a universally unique identifier (UUID).

    Returns:
    str: A UUID string.
    """
    return str(uuid.uuid4())


def file_operations(file_path: str, mode: str, content: str = None) -> str:
    """
    Perform operations on a file - read, append, or write.

    Args:
    file_path (str): Path of the file to operate on.
    mode (str): Operation mode - 'read', 'append', or 'write'.
    content (str, optional): Content to write or append to the file. Required for 'append' and 'write' modes.

    Returns:
    str: Content of the file for 'read' mode, otherwise confirmation message.
    """

    if mode not in ['read', 'append', 'write']:
        raise ValueError("!!! Mode must be 'read', 'append', or 'write' !!!")

    try:
        if mode == 'read':
            with open(file_path, 'r') as file:
                return file.read()

        if mode in ['append', 'write']:
            if content is None:
                raise ValueError("!!! Content must be provided for 'append' or 'write' mode !!!")
            
            write_mode = 'a' if mode == 'append' else 'w'
            with open(file_path, write_mode) as file:
                file.write(content)
                return f"+++ Content successfully added/updated to the file +++"

    except IOError as error:
        return f"!!! An error occurred: {error} !!!"


def is_valid_uuid(uuid_to_test: str) -> bool:
    """
    Check if a given string is a valid UUID.

    Args:
    uuid_to_test (str): The string to test.

    Returns:
    bool: True if the string is a valid UUID, False otherwise.
    """
    regex = (
        r'^[0-9a-fA-F]{8}-'    # 8 hex digits
        r'[0-9a-fA-F]{4}-'     # 4 hex digits
        r'4[0-9a-fA-F]{3}-'    # '4' and 3 hex digits for version 4
        r'[89aAbB][0-9a-fA-F]{3}-'  # One of '8', '9', 'a', 'b', 'A', 'B', and 3 hex digits
        r'[0-9a-fA-F]{12}$'    # 12 hex digits
    )
    return bool(re.match(regex, uuid_to_test))