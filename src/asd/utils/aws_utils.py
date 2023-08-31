import json
import logging
from pathlib import Path
from time import sleep
from typing import Dict, Optional

import boto3
from utils.os_utils import read_aws_credentials_from_file

# Set up logging
logging.basicConfig(level=logging.ERROR)


class AWSInfrastructure:
    """
    Manages AWS-related operations, primarily focusing on Step Functions.

    Attributes:
        AWS_REGION (str): AWS region where the resources reside.
        ASD_STATE_MACHINE_NAME (str): Name of the target Step Functions state machine.
        ASD_STATE_MACHINE_ARN (str): ARN of the target Step Functions state machine.
    """

    # Constants and Boto3 clients (Class attributes)
    AWS_REGION: tuple = ("us-east-1",)
    ASD_STATE_MACHINE_NAME: tuple = ("asd-test",)  # 'asd-main'
    # ASD_AWS_RAY_ASG_NAME: tuple = ("asd_asg",)

    # Fetch Linux Home directory and AWS folders
    home_dir = Path.home()
    aws_folder = Path(f"{home_dir}/.aws")
    aws_accountid_file = Path(f"{aws_folder}/account_id")

    # Read AWS Credentials File
    aws_credentials = read_aws_credentials_from_file(aws_folder=aws_folder)

    # Initialize boto3 clients
    step_functions = boto3.client(
        "stepfunctions",
        region_name=AWS_REGION[0],
        aws_access_key_id=aws_credentials["aws_access_key_id"],
        aws_secret_access_key=aws_credentials["aws_secret_access_key"],
        aws_session_token=aws_credentials["aws_session_token"],
    )

    try:
        with aws_accountid_file.open("r") as file_obj:
            aws_account_id = file_obj.readline().strip()
    except FileNotFoundError:
        aws_account_id = ""

    ASD_STATE_MACHINE_ARN = f"arn:aws:states:{AWS_REGION[0]}:{aws_account_id}:stateMachine:{ASD_STATE_MACHINE_NAME[0]}"

    def __init__(self):
        cloudformation_headscale_main_status: str
        stepfunctions_state_machine_status: str
        lightsail_container_status: str

    @classmethod
    def start_statemachine(
        cls, state_machine_arn: str, execution_name: str, input_data: str
    ) -> Dict[str, str]:
        """
        Starts a Step Functions state machine execution.

        Parameters:
        - state_machine_arn (str): ARN of the state machine to be executed.
        - execution_name (str): Unique name for the execution.
        - input_data (str): JSON input for the state machine.

        Returns:
        - dict: Response from the start_execution call.
        """
        try:
            start_execution = cls.step_functions.start_execution(
                stateMachineArn=state_machine_arn, name=execution_name, input=input_data
            )
            return start_execution
        except Exception as error:
            logging.error(f"!!! Error starting state machine: {error} !!!")
            exit(1)

    @classmethod
    def get_data_from_statemachine(cls, execution_arn: str) -> Dict[str, str]:
        """
        Retrieves data from a running Step Functions execution.

        Parameters:
        - execution_arn (str): ARN of the running execution.

        Returns:
        - dict: Data retrieved from the execution.
        """
        try:
            execution_data = cls.step_functions.describe_execution(
                executionArn=execution_arn
            )
            return execution_data
        except Exception as error:
            logging.error(f"!!! Error fetching data from state machine: {error} !!!")
            exit(1)
