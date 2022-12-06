import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

df_input = st.session_state["df_input"]
if "type" not in st.session_state:
    st.session_state["type"] = "Numerical"
column = st.session_state["type"]

if st.session_state["type"] == "Numerical":
    st.markdown("""
    You selected the following columns:
    """)
    st.dataframe(df_input[column])
    st.markdown("""
    These numerical visualizations are based on your column selection:
    """)        
    tab1, tab2, tab3 = st.tabs(["PCA", "Box plot", "Scatter plot"])

    with tab1:
        st.subheader("Principal component analysis")
        fig_num = px.scatter_matrix(df_input[column])
        fig_num.update_layout()
        st.plotly_chart(fig_num, use_container_width=True)
    with tab2:
        st.subheader("Box plot")
        fig_num = px.box(df_input[column])
        fig_num.update_layout()
        st.plotly_chart(fig_num, use_container_width=True)
    with tab3:
        st.subheader("Scatter plot")
        fig_num = px.scatter(df_input[column])
        fig_num.update_layout()
        st.plotly_chart(fig_num, use_container_width=True)

elif st.session_state["type"] == "Categorical":
    st.markdown("""
    You selected the following columns:
    """)
    st.dataframe(df_input[column])
    st.markdown("""
    These categorical visualizations are based on your column selection:
    """)        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scatter plot", "Box plot", "Pie chart", "Bar chart", "Sunburst chart"])
    with tab1:
        st.subheader("Scatter plot")
        fig_cat = px.scatter(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)
    with tab2:
        st.subheader("Box plot")
        fig_cat = px.box(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)            
    with tab3:
        st.subheader("One dimensional pie chart")
        fig_cat = px.pie(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)            
    with tab4:
        st.subheader("Two dimensional bar chart")
        fig_cat = px.bar(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)          
    with tab5:
        st.subheader("More dimensional sunburst chart")
        fig_cat = px.sunburst(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)           

elif st.session_state["type"] == "Categorical by selection":
    st.markdown("""
    You selected the following columns:
    """)
    st.dataframe(df_input[column])
    st.markdown("""
    These categorical (by user) visualizations are based on your column selection:
    """)        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scatter plot", "Box plot", "Pie chart", "Bar chart", "Sunburst chart"])
    with tab1:
        st.subheader("Scatter plot")
        fig_cat = px.scatter(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)
    with tab2:
        st.subheader("Box plot")
        fig_cat = px.box(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True) 
    with tab3:
        st.subheader("One dimensional pie chart")
        fig_cat = px.pie(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)            
    with tab4:
        st.subheader("Two dimensional bar chart")
        fig_cat = px.bar(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)          
    with tab5:
        st.subheader("Multidimensional sunburst chart")
        fig_cat = px.sunburst(df_input[column])
        fig_cat.update_layout()
        st.plotly_chart(fig_cat, use_container_width=True)