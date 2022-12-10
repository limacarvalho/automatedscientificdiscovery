# Import libraries
import streamlit as st
import pandas as pd
import io

# Set streamlit layout
st.set_page_config(layout="wide")

# Set streamlit session state
if "execution" not in st.session_state and "machine" not in st.session_state:
    st.session_state["execution"] = "Local machine"
    st.session_state["machine"] = "Small Instance (2 vCPUs / 8 GB RAM)"

# Create function for sidebar selection
def handle_execution(new_execution):
    st.session_state.execution = new_execution

# Set user selection and button on the sidebar
execution_type = st.sidebar.radio("Execution options:", ["Local machine", "Remote machine"])
execution_change = st.sidebar.button("Set execution", on_click=handle_execution, args= [execution_type])

# Implement if statements based on sidebar selection
if st.session_state["execution"] == "Remote machine":
    st.title("Create remote cluster")
    #st.write(st.session_state)
    
    def handle_machine(new_machine):
        st.session_state.machine = new_machine
        # integrate variable with connection to cluster

    machine_type = st.radio("Server options:", ["Small Instance (2 vCPUs / 8 GB RAM)", "Medium Instance (4 vCPUs / 16 GB RAM)", "Large Instance (8 vCPUs / 32 GB RAM)", "GPU Instance (8 vCPUs / 32 GB RAM and 1xGPU)"])
    st.markdown("***")
    cluster_name = st.text_input('Cluster name:', placeholder="Enter a name")
    st.write('Your cluster name is:', cluster_name)
    st.markdown("")
    access_key = st.text_input('Access key:', placeholder="Enter an access key")
    st.write('Your access key is:', access_key)
    st.markdown("")
    secret_access_key = st.text_input('Secrect access key:', placeholder="Enter a scecret access key")
    st.write('Your secret access key is:', secret_access_key)
    st.markdown("")
    machine_change = st.button("Set machine", on_click=handle_machine, args= [machine_type])
elif st.session_state["execution"] == "Local machine":
    st.title("Create local cluster")
    
    def handle_local(new_local):
        st.session_state.localOS = new_local
        # integrate variable with connection to local anaconda env

    local_type = st.radio("OS options:", ["macOS with Intel", "Linux"])
    local_change = st.button("Set OS", on_click=handle_local, args= [local_type])
    #st.write(st.session_state)

