import streamlit as st
import pandas as pd
import io

st.set_page_config(layout="wide")

if "execution" not in st.session_state and "machine" not in st.session_state:
    st.session_state["execution"] = "Local machine"
    st.session_state["machine"] = "Small Instance (2 vCPUs / 8 GB RAM)"

def handle_execution(new_execution):
    st.session_state.execution = new_execution


execution_type = st.sidebar.radio("Execution options:", ["Local machine", "Remote machine"])#, on_change=handle_machine, key="execution")
execution_change = st.sidebar.button("Set execution", on_click=handle_execution, args= [execution_type])

if st.session_state["execution"] == "Remote machine":
    st.title("Create remote cluster")
    #st.write(st.session_state)
    def handle_machine(new_machine):
        st.session_state.machine = new_machine

    machine_type = st.radio("Server options:", ["Small Instance (2 vCPUs / 8 GB RAM)", "Medium Instance (4 vCPUs / 16 GB RAM)", "Large Instance (8 vCPUs / 32 GB RAM)", "GPU Instance (8 vCPUs / 32 GB RAM and 1xGPU)"])#, on_change=handle_machine, key="machine_type")
    machine_change = st.button("Set machine", on_click=handle_machine, args= [machine_type])
    st.markdown("***")
    cluster_name = st.text_input('Cluster name:', placeholder="Enter a name")
    st.write('Your cluster name is:', cluster_name)
    st.markdown("")
    access_key = st.text_input('Access key:', placeholder="Enter an access key")
    st.write('Your access key is:', access_key)
    st.markdown("")
    secret_access_key = st.text_input('Secrect access key:', placeholder="Enter a scecret access key")
    st.write('Your secret access key is:', secret_access_key)
if st.session_state["execution"] == "Local machine":
    st.title("Create local cluster")
    #st.write(st.session_state)

