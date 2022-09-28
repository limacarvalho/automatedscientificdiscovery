# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import eplot

p = plt.rcParams

p["figure.figsize"] = 6, 2.5
p["figure.dpi"] = 200


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


#df = px.data.gapminder()
st.image("./media/logo1.png", use_column_width=True)
st.title('Automated-Scientific-Discovery Project')
st.sidebar.title('Plasma Data')
st.subheader('Reza Bakhtiari (adesso) & Jakob Svensson (MPI-IPP)')

# st.sidebar.image("./media/logo2.png", use_column_width=True)

st.write('***')

def write_markdown(title):
    return st.markdown(
            f"<p style='color: #F63366; "
            f"font-weight: bold; font-size: 24px;'> {title} </p>",
            unsafe_allow_html=True,
                        )

write_markdown("Data Upload")

uploaded_file = st.file_uploader("Choose file to upload")
# if uploaded_file is not None:
#         # Can be used wherever a "file-like" object is accepted:
#      df = pd.read_csv(uploaded_file)
#      st.write(df)

file_name = str(uploaded_file).split('.')[-1]
print(f"here is the filename {file_name}")

df = pd.read_csv(uploaded_file)
df.Pulse = df.Pulse.astype(object)
st.write(df)

write_markdown("Automated Data Inspection")

st.write('data shape: #rows->',df.shape[0],' & #features(columns)->',df.shape[1],  '. Below is a sample selection',df.describe(include='all'))
#st.write('Below is a DataFrame:', data_frame, 'Above is a dataframe.')
#st.dataframe(df.sample(5))


df_cols = df.columns
num_cols = df.select_dtypes(include = np.number).columns
cat_cols = df.select_dtypes(exclude = np.number).columns


# Allow use to choose
num_axis = st.sidebar.selectbox('select a numerical feature?', num_cols)
cat_axis = st.sidebar.selectbox('select a categorical feature?', cat_cols)

check = st.checkbox("mark here to see a data-profile report (Caution: for large data, it might take time!)")

if check:
    from pandas_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)



def plot_category(df_, col, top_n=None, figsize=None):
    """
    bar chart of a categorical feature in a dataframe (df_)
    top_n: the number of top categories for plot

    """
    figsize = (10,4)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    #plt.figure(dpi = 600)

    sns.countplot(x=col, 
                  data=df_, 
                  ax=ax, 
                  order=df_[col].value_counts().iloc[:top_n].index,
                    color=None,
                 )
    # sns.color_palette("hls")

    # bar label as percentage
    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/df_.shape[0]), (p.get_x()+0.01, p.get_height()+0.01),
                    weight="normal", horizontalalignment='left', fontsize=10, rotation=0)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.set_title(f"top {top_n if top_n is not None else df_[col].nunique()} categories amonge all {df[col].nunique()} categories of {col}", fontdict={
                 "size": "12", "color": "black", "style": "italic"})
    ax.set_xlabel(col, color="k", bbox=dict(facecolor='orange', edgecolor='orange', boxstyle='round,pad=0.1'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

#st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Different Viz Frameworks"}</h1>', unsafe_allow_html=True)
st.write('***')
st.markdown(f'<span style="background :orange;color:black;font-size:32px;font-family:sans-serif" > **Plasma Data Visualization** </span>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Matplotlib/seaborn", "Plotly"])


####plotly function
def plotly_category(df_, col, sub_cat = None, width = None, height = None):

    fig = px.histogram(df_, x=col, color = sub_cat , text_auto='.1f', histnorm='percent', color_discrete_sequence=['orange'], 
    width=width, height=height )
    
    
    fig.update_xaxes(categoryorder='total descending')
    
    fig.layout.title = f' {df_[col].nunique()} categories of {col}'
    fig.update_layout(
    #title="original data",
    #xaxis_title="X Axis Title",
    #yaxis_title="Y Axis Title",
    #legend_title="Legend Title",
    font=dict(
        #family="",
        size=16,
        #color=""
    ))
    fig.update_layout(bargap=0.2)
    return fig
####

with tab1:

    tab1.subheader("Matplotlib & Seaborn")

    fig, ax = plt.subplots(dpi=300)
    fig = plot_category(df, cat_axis, top_n=10, figsize=None)
    st.pyplot(fig)

    fig, ax = plt.subplots(dpi=300)
    fig = plot_category(df, cat_axis, top_n=2, figsize=None)
    st.pyplot(fig)

with tab2:

    tab2.subheader("Plotly")

    fig = px.histogram(df, x=num_axis,  
                    marginal='box',opacity= 0.7, 
                    color=cat_axis,barmode='overlay', 
                    width=1000, height=500,
                    title=f"main plot (numerical) is {num_axis} and marginal boxplot (categorical) is {cat_axis}" )


    fig.update_layout( font=dict(size=18) )
    fig.update_traces(visible="legendonly")

    st.plotly_chart(fig, use_container_width=False)

    fig1 = plotly_category(df, cat_axis, width=1000, height=600)
    fig1.update_layout( font=dict(size=18) )

    st.plotly_chart(fig1, use_container_width=False)



