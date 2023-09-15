import json
import logging
from pathlib import Path
from time import sleep
from typing import Dict, Optional, Union

import boto3
from utils.os_utils import read_aws_credentials_from_file

# Set up logging
logging.basicConfig(level=logging.ERROR)


class AWSInfrastructure:
    """
    Python Class that manages AWS Cloud resources needed for the ASD Project when using a Remote Multiprocessing Cluster
    The class implements different methods to bootstrap the initial resources needed like the AWS StepFunctions StateMachine, 
    a method to check the current status of the EC2 AutoScaling Group and other Cloud Resources, a method that scales out the
    existing EC2 AutoScaling Group nodes, a method that scales in the existing EC2 AutoScaling Group nodes and another method that
    'purge' the existing resources in the AWS Cloud by removing all related resources from the AWS Account/Region.

    The Class expects the AWS SDK for Python (boto3) available in the global-scope.

    Runs on:
    Python 3.8.10+

    Example:
    aws_env = AWSInfrastructure()
    aws_env.check_status()
    # or
    aws_env.set_up_infrastructure()

    Attributes:
        AWS_REGION (str): AWS region where the resources reside.
        ASD_STATE_MACHINE_NAME (str): Name of the target Step Functions state machine.

    +------------------------------+------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    |           Methods            |                 Expected Parameters                  |                                                              +info                                                               |
    +------------------------------+------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    | check_status()               | n/a                                                  | Runs by default when an instance of class is created.                                                                            |
    |                              |                                                      | Checks the status of any ASD-related AWS resources by triggering the execution of a StateMachine.                                |
    |                              |                                                      |                                                                                                                                  |
    | set_up_infrastructure()      | n/a                                                  | Creates the AWS StepFunctions StateMachine and associated resources,                                                             |
    |                              |                                                      | that handle the deployment and operation of AWS Cloud resources needed by the ASD components (eg. Ray EC2 AutoScaling Cluster).  |
    |                              |                                                      |                                                                                                                                  |
    | start_statemachine()         | execution_name (str): Unique name for the execution. | Starts the execution of the ASD AWS StepFunctions.                                                                               |
    |                              | input_data (str): JSON input for the state machine.  |                                                                                                                                  |
    |                              |                                                      |                                                                                                                                  |
    | get_data_from_statemachine() | execution_arn (str): ARN of the running execution.   | Retrieves data from a running Step Functions execution.                                                                          |
    +------------------------------+------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+

    """

    def __init__(self) -> None:
        """Initialize AWSInfrastructure class with AWS details and boto3 clients."""
        # Constants and Boto3 clients
        self.AWS_REGION: str = "us-east-1"
        self.ASD_STATE_MACHINE_NAME: str = "asd-main" 

        # Fetch Linux Home directory and AWS folders
        self.home_dir: Path = Path.home()
        self.aws_folder: Path = self.home_dir / ".aws"
        self.aws_accountid_file: Path = self.aws_folder / "account_id"

        # Read AWS Credentials File
        self.aws_credentials: Dict[str, Optional[str]] = read_aws_credentials_from_file(self.aws_folder)

        # Initialize boto3 clients
        self.step_functions = boto3.client(
            "stepfunctions",
            region_name=self.AWS_REGION,
            **self.aws_credentials,
        )

        self.cloudformation = boto3.client(
            "cloudformation",
            region_name=self.AWS_REGION,
            **self.aws_credentials,
        )

        # Retrieve AWS Account Id
        try:
            with self.aws_accountid_file.open("r") as file_obj:
                self.aws_account_id: str = file_obj.readline().strip()
        except FileNotFoundError:
            self.aws_account_id = ""

        # Build the AWS StateMachine ARN string
        self.asd_state_machine_arn: str = f"arn:aws:states:{self.AWS_REGION}:{self.aws_account_id}:stateMachine:{self.ASD_STATE_MACHINE_NAME}"

        self.aws_env_status: bool = self.check_status()

    def check_status(self) -> bool:
        """
        Check the status of the AWS Environment, more specifically if the AWS StepFunctions StateMachine that
        handles the deployment and operationality of the ASD components is available or not.

        :return: True if the StateMachine 'asd-main' exists, False otherwise.
        """
        try:
            check_state_machine = self.step_functions.describe_state_machine(
                stateMachineArn=self.asd_state_machine_arn
            )
            return check_state_machine["status"] == "ACTIVE"
        except Exception as error:
            logging.error(f"!!! Error getting state machine info: {error} !!!")
            return False

    def start_statemachine(self, execution_name: str, input_data: str) -> Dict[str, str]:
        """
        Start a Step Functions state machine execution.

        Parameters:
        - execution_name (str): Unique name for the execution.
        - input_data (str): JSON input for the state machine.

        Returns:
        - dict: Response from the start_execution call.
        """
        try:
            start_execution = self.step_functions.start_execution(
                stateMachineArn=self.asd_state_machine_arn, name=execution_name, input=input_data
            )
            return start_execution
        except Exception as error:
            logging.error(f"!!! Error starting state machine: {error} !!!")
            exit(1)

    def get_data_from_statemachine(self, execution_arn: str) -> Dict[str, str]:
        """
        Retrieve data from a running Step Functions execution.

        Parameters:
        - execution_arn (str): ARN of the running execution.

        Returns:
        - dict: Data retrieved from the execution.
        """
        try:
            execution_data = self.step_functions.describe_execution(
                executionArn=execution_arn
            )
            return execution_data
        except Exception as error:
            logging.error(f"!!! Error fetching data from state machine: {error} !!!")
            exit(1)

    def set_up_infrastructure(self) -> bool:
        """
        Start the deployment of an AWS CloudFormation Stack that automatically deploys and configures
        a StepFunctions StateMachine and associated resources needed to operate the Cloud environment and
        components related to the Multiprocessing Remote Cluster for ASD. 

        Parameters:
        - none

        Returns:
        :return: True if the CloudFormation deployment is started successfully, False otherwise.
        """
        stepfunctions_statemachine_cf_file: Union[Path, None] = next(Path("/opt/asd").rglob("asd_stepfunctions_statemachine.yml"), None)
        if not stepfunctions_statemachine_cf_file:
            logging.error("The CF File for the StateMachine was not found.")
            return False

        try:
            with stepfunctions_statemachine_cf_file.open("r") as file_obj:
                cf_template = file_obj.read()
        except Exception as error:
            logging.error(f"!!! Error reading the CF file: {error} !!!")
            return False

        try:
            self.cloudformation.create_stack(
                StackName="statemachine-asd",
                TemplateBody=cf_template,
                Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM", "CAPABILITY_AUTO_EXPAND"],
                OnFailure="DELETE",
            )
        except Exception as error:
            logging.error(f"!!! Error creating the CloudFormation stack: {error} !!!")
            return False

        logging.info("+++ ASD AWS StepFunctions StateMachine creation started successfully +++")
        self.aws_env_status = True
        return True
