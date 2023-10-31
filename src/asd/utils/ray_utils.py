import json
import logging
import os
import shutil
import signal
import subprocess

from time import sleep
from typing import Optional

import boto3
import psutil

from utils.aws_utils import AWSInfrastructure

# Constants
DIR_PATH = "/tmp/ray"

# Configure logging
logging.basicConfig(level=logging.INFO)


class RayCluster:
    """
    Python Class for Ray Local and Remote Cluster administration operations.
    Ray CLI commands are invoked as part of the execution in order to leverage the existing Ray CLI functionality and to isolate Ray from the Python context
    execution.
    The class implements different methods to check the current status of the Ray Cluster, to start a new cluster, to stop an existing one, or to purge
    any related item like Ray system processes, temporary files or remote AWS Cloud resources.

    The Class expects the Ray Python module and AWS SDK for Python (boto3) available in the global-scope, as well as the Ray CLI commands needed to interact with the different Ray
    components.

    Runs on:
    Python 3.8.10+

    Example:
    ray_cluster = RayCluster(mode='local')
    # or
    ray_cluster = RayCluster(mode='remote')

    ray_cluster.start()
    # or
    ray_cluster.stop()

    +----------------+---------------------+-------------------------------------------------------------------------------------------------------------+
    |    Methods     | Expected Parameters |                                                   +info                                                     |
    +----------------+---------------------+-------------------------------------------------------------------------------------------------------------+
    | check_status() |  n/a                | Runs by default when an instance of class is created. Check the status of any existing                      |
    |                |                     | Ray Cluster (local or remote)                                                                               |
    |                |                     |                                                                                                             |
    | start()        |  n/a                | Starts a Ray Cluster if not available already (Starts local and remote clusters on AWS)                     |
    | stop()         |  n/a                | Stops a Ray Cluster if available (Stops local cluster and/or remote cluster on AWS by terminating EC2s)     |
    | purge()        |  n/a                | Stops a Ray Cluster (Local and/or Remote), kills any local OS process related to Ray and deletes existing   |
    |                |                     | temporary Ray files                                                                                         |
    +----------------+---------------------+-------------------------------------------------------------------------------------------------------------+
    """

    def __init__(self, mode: str) -> None:
        """Initialize the RayCluster class."""
        self.mode = mode
        if self.mode == "local":
            self.cluster_status: bool = self.check_status()
            self.local_ray_status_stdout: str
        elif self.mode == "remote":
            self.ASD_AWS_RAY_ASG_NAME = "asd-nodes"
            self.aws_env = AWSInfrastructure()
            self.aws_statemachine: bool = False
            self.cluster_status: bool = self.check_status()
            self.remote_ray_status_stdout: str
        else:
            logging.error(f"!!! Invalid RayCluster mode: '{self.mode}' !!!")


    def check_status(self) -> bool:
        """
        Checks the status of the Ray cluster running locally or on AWS EC2s.

        :return: True if the Ray cluster is running, False otherwise.
        """
        if self.mode == "local":
            try:
                get_local_ray_status = subprocess.run(
                    ["ray", "status"], stdout=subprocess.PIPE, text=True
                )
                ray_status = get_local_ray_status.returncode
                self.local_ray_status_stdout = get_local_ray_status.stdout
            except subprocess.CalledProcessError:
                logging.error("!!! Unable to execute ray status command !!!")
                return False
            except ConnectionError:
                logging.debug("+++ Ray cluster is NOT running +++")
            if ray_status == 0:
                logging.debug("+++ Ray cluster is already running +++")
                return True
            elif ray_status == 1:
                logging.debug("+++ Ray cluster is NOT running +++")
                return False
            else:
                logging.warning(
                    "!!! Unable to determine the status of the Ray cluster !!!"
                )
                return False

        elif self.mode == "remote":
            # Evaluates if AWS StepFunctions StateMachine exists
            if self.aws_env.aws_env_status:
                self.aws_statemachine = True

                # Extract relevant data from the state machine's output
                describe_asd_asg = self.aws_env.handle_state_machine_exec(statemachine_action='CheckStatus')

                if self.ASD_AWS_RAY_ASG_NAME in describe_asd_asg and describe_asd_asg[self.ASD_AWS_RAY_ASG_NAME]['AutoScalingGroups']:
                    asd_asg_total_instances = describe_asd_asg[self.ASD_AWS_RAY_ASG_NAME][
                        "AutoScalingGroups"
                    ][0]["MaxSize"]
                    asd_asg_running_instances = len(
                        describe_asd_asg[self.ASD_AWS_RAY_ASG_NAME]["AutoScalingGroups"][0][
                            "Instances"
                        ]
                    )
                    asd_asg_msg_1 = f"+++ ASD Autoscaling Group currently has a total of {asd_asg_total_instances} instances/VMs, {asd_asg_running_instances} currently running | "
                    logging.info(asd_asg_msg_1)
                    # Commented-out code to calculate the available number of vCPUs in the ASD ASG. This, however, calculates the number of running instances vCPUs and not the 
                    # total ASD vCPU capacity. This example also applies for the 'asd_asg_available_mem_inmb'.
                    # asd_asg_available_vcpus = int(
                    #     describe_asd_asg["instance_types"]["InstanceTypes"][0]["VCpuInfo"][
                    #         "DefaultVCpus"
                    #     ]
                    # ) * int(
                    #     len(
                    #         describe_asd_asg[self.ASD_AWS_RAY_ASG_NAME]["AutoScalingGroups"][0][
                    #             "Instances"
                    #         ]
                    #     )
                    # )
                    asd_asg_max_available_vcpus = int(
                        describe_asd_asg["instance_types"]["InstanceTypes"][0]["VCpuInfo"][
                            "DefaultVCpus"
                        ]
                    ) * int(
                        describe_asd_asg[self.ASD_AWS_RAY_ASG_NAME]["AutoScalingGroups"][0][
                                "MaxSize"
                        ]
                    )
                    asd_asg_max_available_mem_inmb = float(
                        describe_asd_asg["instance_types"]["InstanceTypes"][0][
                            "MemoryInfo"
                        ]["SizeInMiB"]
                        / 1024
                    ) * int(
                        describe_asd_asg[self.ASD_AWS_RAY_ASG_NAME]["AutoScalingGroups"][0][
                            "MaxSize"
                        ]
                    )

                    asd_asg_msg_2 = (
                        f"Max available Cluster vCPUs: {asd_asg_max_available_vcpus} | "
                    )
                    asd_asg_msg_3 = f"Max available Cluster Memory in GB: {asd_asg_max_available_mem_inmb} +++"
                    logging.info(asd_asg_msg_2)
                    logging.info(asd_asg_msg_3)

                    self.remote_ray_status_stdout = (
                        f"{asd_asg_msg_1}\n{asd_asg_msg_2}\n{asd_asg_msg_3}\n"
                    )
                    return True
                elif "Error" in describe_asd_asg:
                    no_available_nodes_msg = (
                        f"+++ The Cluster exists but there are NO running compute nodes on AWS. Choose 'Start Cluster' +++"
                    )
                    logging.info(no_available_nodes_msg)
                    self.remote_ray_status_stdout = no_available_nodes_msg
                    return False

                elif not describe_asd_asg[self.ASD_AWS_RAY_ASG_NAME]['AutoScalingGroups']:
                    no_cluster_msg = (
                        f"+++ No Cluster is available/deployed on AWS. Choose 'Create Cluster' +++"
                    )
                    logging.info(no_cluster_msg)
                    self.remote_ray_status_stdout = no_cluster_msg
                    return False
                else:
                    logging.error(
                        f"!!! Could not retrieve response from StepFunctions StateMachine. ERROR: {describe_asd_asg}"
                    )
                    return False
            else:
                no_aws_resources_msg = "!!! No AWS resoures yet deployed for ASD !!!"
                logging.error(no_aws_resources_msg)
                self.remote_ray_status_stdout = no_aws_resources_msg
                return False


    def update_status(self):
        """
        Forces an update on the instance attribute 'cluster_status' by running the check_status() function.
    
        :return: None
        """
        self.cluster_status: bool = self.check_status()


    def start(self) -> bool:
        """
        Starts the Ray cluster locally or remotely on AWS Cloud (EC2 Instances).

        :return: True if the Ray cluster is successfully started, False otherwise.
        """
        if self.mode == "local":
            logging.debug(
                f"++++ Ray Cluster status is set to: {self.cluster_status} +++"
            )
            if not self.cluster_status:
                try:
                    start_cluster = subprocess.run(
                        ["ray", "start", "--head"], stdout=None, text=True
                    )
                    ray_start_result = start_cluster.returncode
                except subprocess.CalledProcessError:
                    logging.error("!!! Unable to execute ray start command !!!")
                    return False

                if ray_start_result == 0:
                    self.cluster_status = True
                    return True
                else:
                    logging.warning("!!! Unable to start up the Ray cluster !!!")
                    self.cluster_status = False
                    return False
            else:
                logging.warning("!!! Ray Cluster already seems to be running !!!")
                return False
        elif self.mode == "remote":
            # Start and extract relevant data from the state machine's output
            describe_asd_asg = self.aws_env.handle_state_machine_exec(statemachine_action='StartCluster')


    def stop(self) -> bool:
        """
        Stops the Ray cluster running locally or remotely (by shutting down EC2 AutoScaling nodes)

        :return: True if the Ray cluster is successfully stopped, False otherwise.
        """
        logging.debug(f"++++ Ray Cluster status is set to: {self.cluster_status} +++")
        if self.cluster_status:
            try:
                stop_cluster = subprocess.run(["ray", "stop"], stdout=None, text=True)
                ray_stop_result = stop_cluster.returncode
            except subprocess.CalledProcessError:
                logging.error("!!! Unable to execute ray stop command !!!")
                return False

            if ray_stop_result == 0:
                logging.debug("+++ Ray cluster is now down +++")
                self.cluster_status = False
                return True
            else:
                logging.warning("!!! Unable to shut down the Ray cluster !!!")
                self.cluster_status = True
                return False
        else:
            logging.warning("!!! Ray Cluster does not seem to be running !!!")
            return False


    def purge(self) -> bool:
        """
        Stops the Ray cluster running in 'local' mode or remotely on AWS. Purges all related resources and processes.
        For the remote mode, this function deletes any AWS infrastructure that relates to ASD and ray (EC2 AutoScaling 
        Group, IAM Roles, Lightsail Container, Security Groups, etc)

        :return: False, indicating that the Ray cluster is not running.
        """
        self.stop()
        # Delete the directory if it exists
        if os.path.exists(DIR_PATH) and os.path.isdir(DIR_PATH):
            shutil.rmtree(DIR_PATH, ignore_errors=True)
            logging.info(f"+++ Deleted directory: {DIR_PATH} +++")

        # Search for all ray-related system processes
        for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            try:
                process_info = process.info
                process_name = process_info["name"]
                cmdline = process_info["cmdline"]
                if "ray" in process_name or any("ray" in cmd for cmd in cmdline):
                    logging.debug(
                        f"+++ Killing process: PID {process_info['pid']}, Name {process_name} +++"
                    )
                    os.kill(process_info["pid"], signal.SIGTERM)
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ) as error:
                logging.warning(
                    f"!!! Unable to find any ray-related processes: {error} !!!"
                )
        self.cluster_status = False
        return False
