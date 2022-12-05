import streamlit as st
st.set_page_config(layout="wide")

df_input = st.session_state["df_input"]
st.title("Data normalization and machine learning model selection")
st.dataframe(df_input)

def handle_click_without():
    #if st.session_state.column_kind:
    #    st.session_state.type=st.session_state.column_kind
        st.empty()

st.markdown("***")
column_type = st.radio("Which kind of machine learning task do you prefer?", ["Complexity", "Predictability", "Auto-ML"], on_change=handle_click_without, key="column_kind")