import io
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from utils.os_utils import delete_dir, write_aws_credentials_to_file
from utils.ray_utils import RayCluster

# Set streamlit layout
st.set_page_config(
    page_title="ASD - Multiprocessing Cluster",
    page_icon="https://www.ipp.mpg.de/assets/touch-icon-32x32-a66937bcebc4e8894ebff1f41a366c7c7220fd97a38869ee0f2db65a9f59b6c1.png",
    layout="wide",
    initial_sidebar_state="expanded"
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

# Set streamlit session state
if "execution" not in st.session_state and "machine" not in st.session_state:
    st.session_state["execution"] = "Local Execution/Cluster"
    st.session_state["machine"] = "Small Instance (2 vCPUs / 8 GB RAM)"


# Create function for sidebar selection
def handle_execution(new_execution):
    st.session_state.execution = new_execution


# Create a function for progress bar
def progress_bar(msg, sleep_time=0.03):
    progress_bar_init = st.progress(0, text=msg)
    for percent_complete in range(100):
        time.sleep(sleep_time)
        progress_bar_init.progress(percent_complete + 1, text=msg)


# Set user selection and button on the sidebar
execution_type = st.sidebar.radio(
    "Execution options:", ["Local Execution/Cluster", "Remote Execution/Cluster"]
)
execution_change = st.sidebar.button(
    "Confirm", on_click=handle_execution, args=[execution_type]
)

# Implement if statements based on sidebar selection
if st.session_state["execution"] == "Remote Execution/Cluster":
    st.title("Configure Remote Execution/Cluster")
    """
    ---
    ### **Remote Ray Cluster Management**

    **What is a Remote Ray Cluster?**  
    [Ray](https://docs.ray.io/en/latest/index.html) is an open-source framework that simplifies building distributed applications. 
    **ASD** relies on Ray to run workloads locally or remotely on AWS Servers. You have the option to run a remote Ray cluster on the AWS (Amazon Web Services) Cloud, known as a "Remote Ray Cluster". 
    This remote setup is ideal for production runs, compute-intensive workloads, and medium to large-scale executions, depending on the chosen remote resources like EC2 instances sizes, etc.

    **Associated running costs on AWS**
    TODO

    #### **Available Operations**:

    1. **Check Status**:  
    Review the current status of the remote cluster to determine if it's operational or not.

    2. **Create Cluster**:  
    Initialize and spin up a remote Ray cluster. If the cluster is already running, this action might have no effect.
    To create a remote Cluster, some aspects like EC2 instance sizes need to be provided:
        * **S-size**: Gracefully terminate the remote Ray cluster.  

    3. **Modify Cluster**:  
    Manage the state of an existing remote Ray cluster. Under this option, you'll find further choices:

        * **Stop**: Gracefully terminate the remote Ray cluster.  
        * **Restart**: Perform a sequential stop and start operation to refresh the remote cluster.  
        * **Purge**: Force stop the remote cluster and clean up any related resources.
    
    ---

    #### Credentials to access AWS:
    """
    st.caption(
        body="Make sure you have an AWS account, an IAM User and the respective 'access key' and 'secret access key'",
        unsafe_allow_html=False,
        help="Make sure the user(s) has permissions to 'Create', 'List', 'Modify' and 'Delete' resources on the following AWS services: EC2, VPC, StepFunctions, CloudFormation, Lightsail.",
    )
    st.caption(
        body="For more information refer to the AWS official documentation",
        unsafe_allow_html=False,
        help="[How to create an AWS Account](https://docs.aws.amazon.com/accounts/latest/reference/manage-acct-creating.html) | [How to create an IAM User on AWS](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console) | [How to create access keys on AWS](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey)",
    )
    # Variables' Section
    home_dir = Path.home()
    aws_folder = Path(f"{home_dir}/.aws")
    aws_credentials_file = Path(f"{aws_folder}/credentials")
    aws_config_file = Path(f"{aws_folder}/config")
    account_id_file = Path(f"{aws_folder}/account_id")
    aws_region = "us-east-1"
    # Set account_id to False since this is going to be retrieved automatically as part of the function
    account_id = False
    valid_aws_credentials = False

    # AWS Credentials specific
    if aws_credentials_file.exists():
        with open(aws_credentials_file, "r") as file_obj:
            file_obj_content = file_obj.readlines()
            file_obj_content = " ".join(file_obj_content)
            st.text_area(label="Current available credentials:", value=file_obj_content)
        st.button(
            "Delete AWS Credentials",
            type="primary",
            on_click=delete_dir,
            kwargs={"folder_path": str(aws_folder)},
        )
        checking_aws_credentials = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--output", "yaml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        progress_bar(msg="Checking if AWS Credentials are valid...")
        if checking_aws_credentials.returncode == 0:
            st.text_area(
                label="+++ AWS Credentials are valid +++",
                label_visibility="visible",
                value=checking_aws_credentials.stdout,
                disabled=True,
            )
            valid_aws_credentials = True
        elif checking_aws_credentials.returncode != 0:
            st.text_area(
                label=":warning: AWS Credentials are NOT valid :warning:",
                label_visibility="visible",
                value=f"WARNING: Please choose to 'Delete AWS Credentials' and provide new ones.\nError output: {checking_aws_credentials.stderr}",
                disabled=True,
            )
    else:
        aws_access_key_id = st.text_input(
            "AWS Access Key ID:", placeholder="Enter an AWS access key"
        )
        st.markdown("")
        aws_secret_access_key = st.text_input(
            "AWS Secret Access Key:", placeholder="Enter an AWS secret access key"
        )
        st.markdown("")
        aws_session_token = st.text_input(
            "AWS Session Token (If needed):", placeholder="Enter an AWS Session Token)"
        )
        st.markdown("")
        aws_region_msg = st.text_input(
            "AWS Region",
            placeholder="(Currently not supported) (By default, using us-east-1 / N. Virginia)",
            disabled=True,
        )
        st.markdown("")
        if aws_access_key_id and aws_secret_access_key:
            st.button(
                "Submit Credentials",
                type="primary",
                on_click=write_aws_credentials_to_file,
                kwargs={
                    "aws_folder": aws_folder,
                    "aws_credentials_file": aws_credentials_file,
                    "aws_config_file": aws_config_file,
                    "account_id_file": account_id_file,
                    "account_id": account_id,
                    "aws_region": aws_region,
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                    "aws_session_token": aws_session_token,
                },
            )
    if valid_aws_credentials:
        """
        ---

        #### Choose an action from the dropdown to manage your Remote Ray Cluster.
        """
        remote_cluster_options_choices = [
            "Check Status",
            "Create Cluster",
            "Modify Cluster",
        ]
        remote_cluster_options_select = st.selectbox(
            "Available actions:",
            remote_cluster_options_choices,
            help="Choose between creating a remote cluster or modify the properties of an existing one.",
        )

        ray_cluster_remote = RayCluster(mode="remote")

        # Streamlit 'Magic' for ray status output
        ray_status_msg = f"""
        ```text
        {ray_cluster_remote.remote_ray_status_stdout}
        """

        if "Check Status" in remote_cluster_options_select:
            progress_bar(msg="Checking Cluster Status...")
            if ray_cluster_remote.cluster_status:
                st.write("+++ Cluster is available in remote mode +++")
                st.write(ray_status_msg)
            else:
                st.write("+++ Cluster is not yet started +++")
        elif "Create Cluster" in remote_cluster_options_select:
            progress_bar(msg="Creating Remote Cluster...", sleep_time=0.08)
            if ray_cluster_remote.cluster_status:
                st.write("+++ Cluster was already available in remote mode +++")
                st.write(ray_status_msg)
            else:
                ray_cluster_remote.start()
                time.sleep(3)
                ray_cluster_remote.check_status()
                st.write(ray_status_msg)
        elif "Modify Cluster" in remote_cluster_options_select:
            progress_bar(msg="Checking Cluster Status...")
            if ray_cluster_remote.cluster_status:
                st.write("+++ Cluster is available in remote mode +++")
                st.write(ray_status_msg)
            else:
                st.write("+++ Cluster is not yet started +++")
            st.divider()
            st.caption("Choose to 'Stop', 'Restart' or 'Purge' the remote Cluster:")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Stop", use_container_width=True):
                    if ray_cluster_remote.cluster_status:
                        ray_cluster_remote.stop()
                        st.write("+++ Remote cluster has been stopped +++")
                    else:
                        st.write("+++ Remote cluster was already stopped +++")
            with col2:
                if st.button("Restart", use_container_width=True):
                    ray_cluster_remote.stop()
                    time.sleep(5)
                    ray_cluster_remote.start()
                    time.sleep(3)
                    ray_cluster_remote.check_status()
                    st.write(ray_status_msg)
                    st.write("+++ Remote cluster has been restarted +++")
            with col3:
                if st.button("Purge", use_container_width=True):
                    ray_cluster_remote.purge()
                    st.write(ray_status_msg)
                    st.write("+++ Remote cluster has been forced stopped/purged +++")

        # Old section from Gerrit
        # def handle_machine(new_machine):
        #     st.session_state.machine = new_machine
        #     # integrate variable with connection to cluster

        # machine_type = st.radio("Server options:", ["Small Instance (2 vCPUs / 8 GB RAM)", "Medium Instance (4 vCPUs / 16 GB RAM)", "Large Instance (8 vCPUs / 32 GB RAM)", "GPU Instance (8 vCPUs / 32 GB RAM and 1xGPU)"])
        # st.markdown("***")
        # cluster_name = st.text_input('Cluster name:', placeholder="Enter a name")
        # st.markdown("")
        # machine_change = st.button("Set Cluster", on_click=handle_machine, args= [machine_type])

elif st.session_state["execution"] == "Local Execution/Cluster":
    st.title("Configure Local Execution/Cluster")
    """
    ---
    ### **Local Ray Cluster Management**

    **What is a Local Ray Cluster?**  
    [Ray](https://docs.ray.io/en/latest/index.html) is an open-source framework that simplifies building distributed applications. 
    **ASD** relies on Ray to run workloads locally or remotely on AWS Servers. You have the option to run a Ray cluster locally on your machine, known as a "Local Ray Cluster". 
    This local setup is ideal for development, debugging, and small-scale execution, depending on the available local RAM and CPU units.

    #### **Available Operations**:

    1. **Check Status**:  
    Review the current status of the local cluster to determine if it's operational or not.

    2. **Create Cluster**:  
    Initialize and spin up a local Ray cluster. If the cluster is already running, this action might have no effect.

    3. **Modify Cluster**:  
    Manage the state of an existing local Ray cluster. Under this option, you'll find further choices:

        * **Stop**: Gracefully terminate the local Ray cluster.  
        * **Restart**: Perform a sequential stop and start operation to refresh the local cluster.  
        * **Purge**: Force stop the local cluster and clean up any related resources.

    #### Choose an action from the dropdown to manage your Local Ray Cluster.
    """
    local_cluster_options_choices = [
        "",
        "Check Status",
        "Create Cluster",
        "Modify Cluster",
    ]
    local_cluster_options_select = st.selectbox(
        "Available actions:",
        local_cluster_options_choices,
        help="Choose between creating a local cluster or modify the properties of an existing one.",
    )

    ray_cluster = RayCluster(mode="local")
    # Streamlit 'Magic' for ray status output
    ray_status_msg = f"""
    ```text
    {ray_cluster.local_ray_status_stdout}
    """

    if "Check Status" in local_cluster_options_select:
        progress_bar(msg="Checking Cluster Status...")
        if ray_cluster.cluster_status:
            st.write("+++ Cluster is available in local mode +++")
            st.write(ray_status_msg)
        else:
            st.write("+++ Cluster is not yet started +++")
    elif "Create Cluster" in local_cluster_options_select:
        progress_bar(msg="Creating Local Cluster...", sleep_time=0.08)
        if ray_cluster.cluster_status:
            st.write("+++ Cluster was already available in local mode +++")
            st.write(ray_status_msg)
        else:
            ray_cluster.start()
            time.sleep(3)
            ray_cluster.check_status()
            st.write(ray_status_msg)
    elif "Modify Cluster" in local_cluster_options_select:
        progress_bar(msg="Checking Cluster Status...")
        if ray_cluster.cluster_status:
            st.write("+++ Cluster is available in local mode +++")
            st.write(ray_status_msg)
        else:
            st.write("+++ Cluster is not yet started +++")
        st.divider()
        st.caption("Choose to 'Stop', 'Restart' or 'Purge' the local Cluster:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Stop", use_container_width=True):
                if ray_cluster.cluster_status:
                    ray_cluster.stop()
                    st.write("+++ Local cluster has been stopped +++")
                else:
                    st.write("+++ Local cluster was already stopped +++")
        with col2:
            if st.button("Restart", use_container_width=True):
                ray_cluster.stop()
                time.sleep(5)
                ray_cluster.start()
                time.sleep(3)
                ray_cluster.check_status()
                st.write(ray_status_msg)
                st.write("+++ Local cluster has been restarted +++")
        with col3:
            if st.button("Purge", use_container_width=True):
                ray_cluster.purge()
                st.write(ray_status_msg)
                st.write("+++ Local cluster has been forced stopped/purged +++")

        # from utils.tailscale_connect import get_network_interfaces, tailscale_ips_startwith
        # interfaces = get_network_interfaces()
        # for key, value in interfaces.items():
        #     if tailscale_ips_startwith in value:
        #         tailscale_int = interfaces[key]
        #         st.write(f'+++ Tailscale Interface found with ipv4: {tailscale_int} +++')

        # text_input = st.text_input(
        #     "Enter AWS Credentials here:",
        #     label_visibility="visible",
        #     key="str",
        #     type="default"
        #     )
        # if text_input:
        #     aws = AwsCredentials(text_input)

        # ###### Part 1, Relevance function of ml-algorithm ######
        # st.markdown("***")
        # st.subheader("Relevance")
        # st.markdown("""
        # Discover the relevance of your dataset.
        # """)
        # if "button_relevance_start_discovery" not in st.session_state:
        #     st.session_state["button_relevance_start_discovery"] = False

        # def relevance_discovery_click():
        #     st.session_state["button_relevance_start_discovery"] = True

        # if st.button("Start discovery", on_click=relevance_discovery_click) or st.session_state["button_relevance_start_discovery"]:
        #     ###### Implement tbe code of Relevance ######
        #     @st.cache(allow_output_mutation=True)
        #     def relevance_discovery():
        #         return_relevance = relevance(relevance_df, relevance_column, relevance_target, relevance_options)
        #         return return_relevance

        #     return_relevance = relevance_discovery()
        #     st.markdown("""
        #     Discovery finished.
        #     """)

        #     #Visualize the output (the return values) of the relevance function
        #     st.markdown("""
        #     Output of the relevance part:
        #     """)
        #     st.write(return_relevance)
        # else:
        #     st.markdown("""
        #     You have not chosen this task.
        #     """)

############################################################

# st.write(st.session_state)
