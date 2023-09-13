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

    Important Instance Attributes:
        AWS_REGION (str): AWS region where the resources reside.
        ASD_STATE_MACHINE_NAME (str): Name of the target Step Functions state machine.
    """

    def __init__(self):

        # Constants and Boto3 clients
        self.AWS_REGION: tuple = ("us-east-1",)
        self.ASD_STATE_MACHINE_NAME: tuple = ("asd-main",)  # 'asd-main'
        # ASD_AWS_RAY_ASG_NAME: tuple = ("asd_asg",)

        # Fetch Linux Home directory and AWS folders
        self.home_dir = Path.home()
        self.aws_folder = Path(f"{self.home_dir}/.aws")
        self.aws_accountid_file = Path(f"{self.aws_folder}/account_id")

        # Read AWS Credentials File
        self.aws_credentials = read_aws_credentials_from_file(aws_folder=self.aws_folder)

        # Initialize boto3 clients
        self.step_functions = boto3.client(
            "stepfunctions",
            region_name=self.AWS_REGION[0],
            aws_access_key_id=self.aws_credentials["aws_access_key_id"],
            aws_secret_access_key=self.aws_credentials["aws_secret_access_key"],
            aws_session_token=self.aws_credentials["aws_session_token"],
        )

        self.cloudformation = boto3.client(
            "cloudformation",
            region_name=self.AWS_REGION[0],
            aws_access_key_id=self.aws_credentials["aws_access_key_id"],
            aws_secret_access_key=self.aws_credentials["aws_secret_access_key"],
            aws_session_token=self.aws_credentials["aws_session_token"],
        )        

        # Retrieve AWS Account Id
        try:
            with self.aws_accountid_file.open("r") as file_obj:
                self.aws_account_id = file_obj.readline().strip()
        except FileNotFoundError:
            self.aws_account_id = ""

        # Build the AWS StateMachine ARN string
        self.asd_state_machine_arn = f"arn:aws:states:{self.AWS_REGION[0]}:{self.aws_account_id}:stateMachine:{self. ASD_STATE_MACHINE_NAME[0]}"

        #! WIP
        self.aws_env_status: bool = self.check_status()
        self.stepfunctions_state_machine_status: str
        self.cloudformation_headscale_main_status: str
        self.lightsail_container_status: str


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
            if check_state_machine['status'] == 'ACTIVE':
                return True
            else:
                return False
        except Exception as error:
            logging.error(f"!!! Error getting state machine info/details: {error} !!!")
            return False


    def start_statemachine(
        self, execution_name: str, input_data: str
    ) -> Dict[str, str]:
        """
        Starts a Step Functions state machine execution.

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
        Retrieves data from a running Step Functions execution.

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


    def set_up_infrastructure(self):
        """
        Starts the deployment of an AWS CloudFormation Stack that automatically deploys and configures
        a StepFunctions StateMachine and associated resources needed to operate the Cloud environment and
        components related to the Multiprocessing Remote Cluster for ASD. 

        Parameters:
        - none

        Returns:
        :return: True if the CloudFormation deployment is started successfully, False otherwise.
        """
        stepfunctions_statemachine_cf_file = Path('/opt/asd').rglob("asd_stepfunctions_statemachine.yml")
        stepfunctions_statemachine_cf_file = [file_found for file_found in stepfunctions_statemachine_cf_file]
        try:
            with open(stepfunctions_statemachine_cf_file[-1], 'r') as file_obj:
                cf_template = file_obj.read()
        except FileNotFoundError as error:
            logging.error(f"!!! The CF File '{stepfunctions_statemachine_cf_file[-1]}' for the StateMachine was not found. ERROR: {error}")
            return False
        else:
            create_cf_stack_statemachine_asd = self.cloudformation.create_stack(
                StackName='statemachine-asd',
                TemplateBody=cf_template,
                Capabilities=[
                    'CAPABILITY_IAM',
                    'CAPABILITY_NAMED_IAM',
                    'CAPABILITY_AUTO_EXPAND'
                ],
                OnFailure='DELETE'
            )
            logging.info(f"+++ The ASD AWS StepFunctions StateMachine creation was started +++\n{create_cf_stack_statemachine_asd}")
            self.aws_env_status = True
            return True
