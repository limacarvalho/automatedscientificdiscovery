import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from os import chmod
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import boto3
import botocore
import pytz
from botocore.config import Config
from utils.os_utils import (generate_uuid, is_valid_uuid,
                            read_aws_credentials_from_file)
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()

# Sets the boto3 Standard retry mode
config = Config(retries={"max_attempts": 10, "mode": "standard"})

SUPPORTED_INSTANCE_TYPES = [
    "t3.xlarge",
    "m5a.xlarge",
    "m5a.2xlarge",
    "m5a.4xlarge",
    "c6a.2xlarge",
    "c6a.4xlarge",
    "c6a.8xlarge",
    "c6a.12xlarge",
    "c6a.16xlarge",
]


@dataclass
class RemoteClusterSizing:
    """
    Basic Python Dataclass that contains attributes needed to configure AWS resources like
    EC2 instances and AutoScaling Groups.
    This is mainly used when the user selects, in the Streamlit page (Multiprocessing Cluster), the EC2 Instance Type and Number of Maximum Running
    instances.
    """

    instance_type: str = "t3.xlarge"
    max_number_running_instances: int = 1


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
    aws_env.bootstrap_aws_resources()

    Attributes:
        AWS_REGION (str): AWS region where the resources reside.
        ASD_STATE_MACHINE_NAME (str): Name of the target Step Functions state machine.

    +------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    |           Methods            |                 Expected Parameters                    |                                                              +info                                                               |
    +------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    | check_status()               | n/a                                                    | Runs by default when an instance of class is created.                                                                            |
    |                              |                                                        | Checks the status of any ASD-related AWS resources by triggering the execution of a StateMachine.                                |
    |                              |                                                        |                                                                                                                                  |
    | bootstrap_aws_resources()    | n/a                                                    | Creates the AWS StepFunctions StateMachine and associated resources,                                                             |
    |                              |                                                        | that handle the deployment and operation of AWS Cloud resources needed by the ASD components (eg. Ray EC2 AutoScaling Cluster).  |
    |                              |                                                        |                                                                                                                                  |
    | start_statemachine()         | execution_name (str): Unique name for the execution.   | Starts the execution of the ASD AWS StepFunctions.                                                                               |
    |                              | input_data (str): JSON input for the state machine.    |                                                                                                                                  |
    |                              |                                                        |                                                                                                                                  |
    | get_data_from_statemachine() | execution_arn (str): ARN of the running execution.     | Retrieves data from a running Step Functions execution.                                                                          |
    |                              |                                                        |                                                                                                                                  |
    | handle_state_machine_exec()  | statemachine_action (str): 1 Word description of exec. | Handles the start and execution check of a StateMachine. Calls 'start_statemachine' and 'get_data_from_statemachine' functions.  |
    |                              | input_payload (dict): Full payload to StateMachine.    |                                                                                                                                  |
    |                              | synchronous_invocation (bool): True or False.          |                                                                                                                                  |
    |                              |                                                        |                                                                                                                                  |
    | put_ssm_parameter()          | param_name (str):  Name of the SSM Parameter           | Creates or updates an SSM Parameter used by the deployment to track the number of ASD Deployments in the AWS account             |
    |                              | param_value (str): Value of the SSM Parameter          |                                                                                                                                  |
    |                              | param_desc (str): Description                          |                                                                                                                                  |
    |                              |                                                        |                                                                                                                                  |                                                                                                                     |
    +------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+

    """

    def __init__(self) -> None:
        """Initialize AWSInfrastructure class with AWS details and boto3 clients."""

        # Constants
        self.AWS_REGION: str = "us-east-1"
        self.ASD_NUM_DEPLOYMENTS_SSM_PARAM = "asd-num-of-deployments"
        self.ASD_NUM_DEPLOYMENTS_SSM_PARAM_INIT_VALUE = "1"
        self.ASD_NUM_DEPLOYMENTS_PARAM_PATTERN = (
            r"^([1-9]|[1-9]\d|1\d\d|2\d\d|3\d\d|4\d\d|5\d\d|6\d\d|7\d\d|8\d\d|9\d\d)$"
        )

        # Set paths for AWS configuration and account ID
        self.home_dir: Path = Path.home()
        self.aws_folder: Path = self.home_dir / ".aws"
        self.aws_accountid_file: Path = self.aws_folder / "account_id"

        # Read AWS Account ID from file, if available
        try:
            with self.aws_accountid_file.open("r") as file_obj:
                self.aws_account_id: str = file_obj.readline().strip()
        except FileNotFoundError:
            self.aws_account_id = ""
            raise FileNotFoundError

        # Read AWS credentials from the configuration file
        self.aws_credentials: Dict[str, Optional[str]] = read_aws_credentials_from_file(self.aws_folder)

        # Initialize boto3 clients
        self.step_functions = boto3.client(
            "stepfunctions", region_name=self.AWS_REGION, config=config, **self.aws_credentials
        )

        self.cloudformation = boto3.client(
            "cloudformation", region_name=self.AWS_REGION, config=config, **self.aws_credentials
        )

        self.ssm = boto3.client("ssm", region_name=self.AWS_REGION, config=config, **self.aws_credentials)

        # AWS Deployment Tracking and UUID
        self.asd_aws_deployment_counter: int = int()
        self.asd_container_uuid: str = str()
        self.uuid_not_valid_msg: str = str()
        self.asd_deployment_tracking_file: Path = self.home_dir / ".asd_container_uuid"

        # Invoke the aws_deployment_tracking() method
        self.aws_deployment_tracking: tuple = self.asd_deployment_identification()

        # Build the AWS StepFunctions StateMachine name based on the number of existing deployments
        self.asd_state_machine_name: str = f"asd-main-{self.aws_deployment_tracking[-1]}"
        self.asd_state_machine_arn: str = (
            f"arn:aws:states:{self.AWS_REGION}:{self.aws_account_id}:stateMachine:{self.asd_state_machine_name}"
        )
        self.asd_state_machine_stack_name = f"statemachine-asd-{self.aws_deployment_tracking[-1]}"

        # Invoke the aws_env_status() and save result as part of instance attribute
        self.aws_env_status: bool = self.check_status()

    def asd_deployment_identification(self) -> Tuple[str]:
        """
        Identify and track ASD deployments using a UUID and a Deployment counter integer.

        :return: a Tuple containing both UUID and Deployment Counter
        """

        try:
            # Read and parse the deployment tracking file
            with self.asd_deployment_tracking_file.open("r") as file_obj:
                file_content_dict = json.loads(file_obj.read())
                self.asd_container_uuid: str = file_content_dict["asd_container_uuid"]
                self.asd_aws_deployment_counter = file_content_dict["asd_deployment_number"]

                # Validate UUID; if invalid, log error and raise ValueError
                if not is_valid_uuid(self.asd_container_uuid):
                    self.uuid_not_valid_msg = f"!!! Invalid UUID format in '{self.asd_deployment_tracking_file}' file. AWS components have to be re-deployed !!!"
                    logging.info(self.uuid_not_valid_msg)
                    raise ValueError(self.uuid_not_valid_msg)

        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            # Handle file not found, invalid UUID, or JSON parsing errors
            logging.info(f"+++ New ASD Container UUID generated and saved to '{self.asd_deployment_tracking_file}' +++")
            self.asd_container_uuid: str = generate_uuid()

            # Check and update the AWS SSM parameter for the number of deployments
            try:
                # Attempt to get the current value of the SSM parameter
                ssm_get_parameter = self.ssm.get_parameter(
                    Name=self.ASD_NUM_DEPLOYMENTS_SSM_PARAM, WithDecryption=False
                )
            except self.ssm.exceptions.ParameterNotFound:
                # If the parameter is not found, create it with the initial value
                self.put_ssm_parameter(
                    param_name=self.ASD_NUM_DEPLOYMENTS_SSM_PARAM,
                    param_value=self.ASD_NUM_DEPLOYMENTS_SSM_PARAM_INIT_VALUE,
                )
                self.asd_aws_deployment_counter: int = self.ASD_NUM_DEPLOYMENTS_SSM_PARAM_INIT_VALUE
                logging.info(
                    f"+++ AWS SSM Parameter '{self.ASD_NUM_DEPLOYMENTS_SSM_PARAM}' was created successfully +++"
                )
            else:
                # If the parameter exists, increment and update its value
                existing_ssm_param_value: int = int(ssm_get_parameter["Parameter"]["Value"])
                self.asd_aws_deployment_counter = existing_ssm_param_value + 1

                # Ensure the updated value matches the specified pattern before updating
                if re.match(self.ASD_NUM_DEPLOYMENTS_PARAM_PATTERN, str(existing_ssm_param_value)) and re.match(
                    self.ASD_NUM_DEPLOYMENTS_PARAM_PATTERN, str(self.asd_aws_deployment_counter)
                ):
                    self.put_ssm_parameter(
                        param_name=self.ASD_NUM_DEPLOYMENTS_SSM_PARAM, param_value=str(self.asd_aws_deployment_counter)
                    )
                    logging.info(
                        f"+++ AWS SSM Parameter '{self.ASD_NUM_DEPLOYMENTS_SSM_PARAM}' was updated successfully +++"
                    )
                else:
                    # Log an error if the updated value does not match the pattern
                    logging.error(
                        "!!! SSM Parameter value does not match existing regex rule 'values from 1 to 999' !!!"
                    )
            finally:
                # Writes ASD Container UUID and AWS Deployment Counter information to .asd_container_uuid file
                with self.asd_deployment_tracking_file.open("w") as file_obj:
                    file_content_dict = {
                        "asd_container_uuid": self.asd_container_uuid,
                        "asd_deployment_number": self.asd_aws_deployment_counter,
                    }
                    file_obj.write(json.dumps(file_content_dict, indent=4, default=str))
                # Sets the file to be read-only for all users
                chmod(self.asd_deployment_tracking_file, 0o444)
        finally:
            # Returns a tuple containing ASD UUID and Deployment Counter Integer value
            return (self.asd_container_uuid, self.asd_aws_deployment_counter)

    def check_status(self) -> bool:
        """
        Check the status of the AWS Environment, more specifically if the AWS StepFunctions StateMachine that
        handles the deployment and operationality of the ASD components is available or not.

        :return: True if the StateMachine 'asd-main' exists, False otherwise.
        """
        try:
            check_state_machine = self.step_functions.describe_state_machine(stateMachineArn=self.asd_state_machine_arn)
            return check_state_machine["status"] == "ACTIVE"
        except Exception as error:
            logging.warning(
                f"!!! Unable to retrieve state machine info for resource '{self.asd_state_machine_arn}': {error} !!!"
            )
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
            execution_data = self.step_functions.describe_execution(executionArn=execution_arn)
            return execution_data
        except Exception as error:
            logging.error(f"!!! Error fetching data from state machine: {error} !!!")
            exit(1)

    def handle_state_machine_exec(
        self, statemachine_action: str = "CheckStatus", input_payload: dict = {}, synchronous_invocation: bool = True
    ) -> Union[str, bool]:
        """
        Handle the Start and Status Check of an AWS StepFunctions StateMachine execution
        Calls 'start_statemachine' and 'get_data_from_statemachine' functions.

        Parameters:
        - statemachine_action (str): Optional. One word describing the execution of the StateMachine
        - input_payload (dict): Optional. Python dictionary contaning the full payload to be passed to the StateMachine

        Returns:
        - str | bool: A JSON string containing the output of the execution or a boolean if the execution fails
        """

        # Get Date/Time UTC
        iso_datetime_utc = datetime.now(pytz.timezone("UTC")).replace(microsecond=0).strftime("%d-%m-%Y-%H%M%S")

        # Defaul maximum synchronous invocation timeout value
        max_timeout_synchronous_invocation: int = 60

        if input_payload:
            input_data = json.dumps(input_payload)
            asd_state_machine_exec_name = f"{input_payload['Action']}-{iso_datetime_utc}"
        else:
            input_data = json.dumps(
                {"Action": statemachine_action, "AsdDeploymentCount": self.aws_deployment_tracking[-1]}
            )
            asd_state_machine_exec_name = f"{statemachine_action}-{iso_datetime_utc}"

        # Start the state machine
        execution_response = self.start_statemachine(
            execution_name=asd_state_machine_exec_name,
            input_data=input_data,
        )

        # Handles synchronous and asynchronous invocations
        if synchronous_invocation:
            # Poll the state machine until it completes or times out
            timeout_sleep = 1
            while self.get_data_from_statemachine(execution_response["executionArn"])["status"] == "RUNNING":
                sleep(timeout_sleep)
                timeout_sleep += 1
                if timeout_sleep >= max_timeout_synchronous_invocation:
                    logging.error(
                        f"!!! ERROR: StateMachine took more than {timeout_sleep} seconds to execute !!! Aborting !!!"
                    )
                    return False

            # Extract relevant data from the state machine's output
            return json.loads(self.get_data_from_statemachine(execution_response["executionArn"])["output"])
        else:
            # Extract the execution AWS ARN from the state machine's execution object
            return execution_response["executionArn"]

    def bootstrap_aws_resources(self) -> bool:
        """
        Start the deployment of an AWS CloudFormation Stack that automatically deploys and configures
        a StepFunctions StateMachine and associated resources needed to operate the Cloud environment and
        components related to the Multiprocessing Remote Cluster for ASD.

        Parameters:
        - none

        Returns:
        :return: True if the CloudFormation deployment is started successfully, False otherwise.
        """
        stepfunctions_statemachine_cf_file: Union[Path, None] = next(
            Path("/opt/asd").rglob("asd_stepfunctions_statemachine.yml"), None
        )
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
                StackName=self.asd_state_machine_stack_name,
                TemplateBody=cf_template,
                Parameters=[
                    {
                        "ParameterKey": "StateMachineName",
                        "ParameterValue": self.asd_state_machine_name,
                    },
                    {
                        "ParameterKey": "AsdDeploymentCount",
                        "ParameterValue": str(self.asd_aws_deployment_counter),
                    },
                    {
                        "ParameterKey": "AsdContainerUuid",
                        "ParameterValue": self.asd_container_uuid,
                    },
                ],
                Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM", "CAPABILITY_AUTO_EXPAND"],
                OnFailure="DELETE",
                Tags=[
                    {"Key": "CreatedBy", "Value": "ASD"},
                    {"Key": "AsdContainerUuid", "Value": self.asd_container_uuid},
                ],
            )
        except Exception as error:
            logging.error(f"!!! Error creating the CloudFormation stack: {error} !!!")
            return False

        logging.info("+++ ASD AWS StepFunctions StateMachine creation started successfully +++")
        self.aws_env_status = True
        return True

    def put_ssm_parameter(
        self,
        param_name: str,
        param_value: str,
        param_desc: str = "AWS SSM Parameter string containing an integer number from 1 to 999, that keeps track of a particular ASD deployment number",
    ) -> bool:
        """
        Creates or Updates (PUT) an SSM Parameter needed in order to track the number of ASD deployments in a particular AWS Account.

        Parameters:
        - param_name (str): Name of the SSM Parameter
        - param_value (str): Value to assign to the SSM Parameter
        - param_desc (str): Optional. Description of the SSM Parameter

        Returns:
        - str | bool: A JSON string containing the output of the execution or a boolean if the execution fails
        """

        # Creates or Updates the SSM Parameter that keeps track of the ASD Deployments
        try:
            ssm_put_parameter = self.ssm.put_parameter(
                Name=param_name,
                Description=param_desc,
                Value=param_value,
                Type="String",
                Overwrite=True,
                Tier="Standard",
                DataType="text",
            )

            # Sleeps for 2 seconds to give enough time for the resource creation (eventual consistency) and then tags it
            sleep(2)
            self.ssm.add_tags_to_resource(
                ResourceType="Parameter",
                ResourceId=param_name,
                Tags=[{"Key": "CreatedBy", "Value": "ASD"}],
            )
        except botocore.exceptions.ClientError as error:
            logging.error(
                f"!!! ERROR while creating/updating SSM parameter '{param_name}' with value '{param_value}' !!! ERROR_MSG: {error}"
            )
            return False
        else:
            # If condition to evaluate if SSM parameter is to be created for the first time or to be updated
            if ssm_put_parameter["Version"] == "1":
                logging.info(
                    f"+++ SSM Parameter '{param_name}' was successfully created with value '{param_value}' +++"
                )
            else:
                logging.info(
                    f"+++ SSM Parameter '{param_name}' was successfully updated with value '{param_value}' +++"
                )
            return True


class AWSGetdata:
    """
    Python Class containing only StaticMethods used to retrieve data from AWS resources
    Including, but not limited to, EC2 instances and AutoScalingGroups, CloudFormation, StepFunctions, Lightsail and VPC resources.

    The Class expects the AWS SDK for Python (boto3) available in the global-scope.

    Runs on:
    Python 3.8.10+

    Example:
    AWSGetdata.get_vpc_data()

    Attributes:
        n/a

    +------------------------------+------------------------------------------------------+----------------------------------------------------------------------------------------------------+
    |           Methods            |                 Expected Parameters                  |                                                +info                                               |
    +------------------------------+------------------------------------------------------+----------------------------------------------------------------------------------------------------+
    | get_vpc_data()               | n/a                                                  | Retrieves data from AWS containing VPC information                                                 |
    |                              |                                                      | Returns a list of custom dictionaries containing basic VPC information like ID, CIDR, Subnets, etc |
    | get_aws_service_quota()      | asd_custom_service_quota_dict: dict                  | Returns a dictionary containing the AWS Service Quotas Information for a specified AWS Service     |
    | get_autoscaling_group_data() | autoscaling_group_name: str                          | Returns a dictionary containing the properties/settings of an AWS EC2 AutoScalingGroup             |
    +------------------------------+------------------------------------------------------+----------------------------------------------------------------------------------------------------+

    """

    @staticmethod
    def get_vpc_data() -> List[Dict]:
        """
        Retrieves information about available VPCs on AWS.

        Parameters:
        - none

        Returns:
        :return: List of custom dictionaries containing information like VPC Id, VPC CIDR Block, List of Subnets, etc
        """

        aws_region: str = "us-east-1"
        # Fetch Linux Home directory and AWS folders
        home_dir: Path = Path.home()
        aws_folder: Path = home_dir / ".aws"

        # Read AWS Credentials File
        aws_credentials: Dict[str, Optional[str]] = read_aws_credentials_from_file(aws_folder)

        # Initialize boto3 clients
        ec2 = boto3.client("ec2", region_name=aws_region, config=config, **aws_credentials)

        # Get the list of AWS VPCs in a specific account/region
        try:
            list_vpcs = ec2.describe_vpcs(MaxResults=100)
        except Exception as error:
            logging.error(f"!!! Error retrieving the AWS VPC information: {error} !!!")

        # Compact list of VPC results
        compact_vpc_info: list = []

        # For each VPC available, retrieves all Subnets (Name and Subnet Id)
        for vpc in list_vpcs["Vpcs"]:
            if vpc["State"] == "available":
                # Lists available subnets per VPC
                try:
                    list_subnets = ec2.describe_subnets(
                        Filters=[
                            {
                                "Name": "vpc-id",
                                "Values": [
                                    vpc["VpcId"],
                                ],
                            },
                        ],
                        MaxResults=100,
                    )
                except Exception as error:
                    logging.error(f"!!! Error retrieving the AWS VPC Subnet(s) information: {error} !!!")

                # Builds a list of strings containing Subnet Name and ID
                subnets_list = []
                for subnet in list_subnets["Subnets"]:
                    subnet_name = None
                    try:
                        for tag in subnet["Tags"]:
                            if tag["Key"] == "Name":
                                subnet_name = tag["Value"]
                            else:
                                continue
                    except KeyError as no_tags_error:
                        logging.warning(
                            f"!!! No Tags found for AWS VPC subnets !!!\nERROR: {no_tags_error}\nSubnets: {list_subnets}"
                        )
                    subnets_list.append(f"SubnetId: {subnet['SubnetId']} | Name: {subnet_name}")

                # Builds custom dictionary containing VPC and Subnet information (list)
                vpc_dict = {
                    "VpcId": vpc["VpcId"],
                    "CidrBlock": vpc["CidrBlock"],
                    "IsDefault": vpc["IsDefault"],
                    "Subnets": subnets_list,
                }
                try:
                    if vpc["Tags"]:
                        for tag in vpc["Tags"]:
                            if tag["Key"] == "Name":
                                vpc_dict["VpcName"] = tag["Value"]
                except KeyError:
                    vpc_dict["VpcName"] = "None"
                    logging.warning(f"+++ VPC {vpc['VpcId']} does not have a name +++")
                compact_vpc_info.append(vpc_dict)

        return compact_vpc_info

    @staticmethod
    def get_autoscaling_group_data(autoscaling_group_name: str) -> Dict:
        """
        Retrieves information about a particular AWS EC2 AutoScalingGroup.

        Parameters:
        - autoscaling_group_name: str

        Returns:
        :return: boto3 response dictionary containing the autoscaling group properties and settings
        """

        aws_region: str = "us-east-1"
        # Fetch Linux Home directory and AWS folders
        home_dir: Path = Path.home()
        aws_folder: Path = home_dir / ".aws"

        # Read AWS Credentials File
        aws_credentials: Dict[str, Optional[str]] = read_aws_credentials_from_file(aws_folder)

        # Initialize boto3 clients
        asg = boto3.client("autoscaling", region_name=aws_region, config=config, **aws_credentials)

        # Retrieves the EC2 AutoScalingGroup information based on the ASG name provided
        try:
            asg_info = asg.describe_auto_scaling_groups(
                AutoScalingGroupNames=[
                    autoscaling_group_name,
                ]
            )
        except Exception as error:
            logging.error(f"!!! Error retrieving the AWS EC2 AutoScalingGroup information: {error} !!!")
            return False
        else:
            return asg_info

    @staticmethod
    def get_aws_service_quota(asd_custom_service_quota_dict: dict) -> dict:
        """
        Retrieves the current AWS Service Quota available in the specified account/region, for a specific AWS Service.
        More info at: https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html

        Parameters:
        - asd_custom_service_quota_dict (dict): Custom Python dictionary containing the AWS ServiceCode and QuotaCode:
        eg. {'ServiceCode': 'ec2', 'QuotaCode': 'L-1216C47A'}

        Returns:
        - dict: Response from the service_quotas.get_service_quota API call.

        """

        aws_region: str = "us-east-1"
        # Fetch Linux Home directory and AWS folders
        home_dir: Path = Path.home()
        aws_folder: Path = home_dir / ".aws"

        # Read AWS Credentials File
        aws_credentials: Dict[str, Optional[str]] = read_aws_credentials_from_file(aws_folder)

        # Initialize boto3 clients
        service_quotas = boto3.client("service-quotas", region_name=aws_region, config=config, **aws_credentials)

        try:
            get_current_service_quota = service_quotas.get_service_quota(
                ServiceCode=asd_custom_service_quota_dict["ServiceCode"],
                QuotaCode=asd_custom_service_quota_dict["QuotaCode"],
            )
        except Exception as error:
            logging.error(f"!!! Error retrieving the AWS Service Quota information: {error} !!!")
            return False

        return get_current_service_quota
