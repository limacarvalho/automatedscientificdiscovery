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

#import eplot

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


# @st.cache()
# def load_data():
#     # df = pd.read_csv( 
#     #     'https://github.com/chris1610/pbpython/blob/master/data/cereal_data.csv?raw=True'
#     # )
#     df = px.data.gapminder()
#     return df


# # Read in the cereal data
# df = load_data()

df = px.data.gapminder()
st.image("./media/logo1.png", use_column_width=True)
st.title('Automated-Scientific-Discovery Project')
st.sidebar.title('Gap-Minder Data')
st.subheader('Reza Bakhtiari (adesso) & Jakob Svensson (MPI-IPP)')

# st.sidebar.image("./media/logo2.png", use_column_width=True)

st.write('***')

st.markdown('**Data inspection** ')
st.write('data shape: #rows->',df.shape[0],' & #features(columns)->',df.shape[1],  '. Below is a sample selection',df.sample(5))
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
st.markdown(f'<span style="background :orange;color:black;font-size:32px;font-family:sans-serif" > **Visualization Frameworks** </span>', unsafe_allow_html=True)
st.markdown(f'<span style="background :white;color:blue;font-size:20px;font-family:sans-serif" > **Comprehensiv Tools -> https://pyviz.org** </span>', unsafe_allow_html=True)




st.image("https://rougier.github.io/python-visualization-landscape/landscape-colors.png", use_column_width=True)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Plotly", "Matplotlib/seaborn", "Pyechart(eplot)","Altair", "Bokeh/HoloViz", "ViSad"])


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

    tab1.subheader("Plotly")

    fig = px.histogram(df, x=num_axis,  
                    marginal='box',opacity= 0.7, 
                    color=cat_axis,barmode='overlay', 
                    width=1000, height=500,
                    title=f"main plot (numerical) is {num_axis} and marginal boxplot (categorical) is {cat_axis}" )


    fig.update_layout( font=dict(size=18) )

    st.plotly_chart(fig, use_container_width=False)

    fig1 = plotly_category(df, cat_axis, width=1000, height=600)
    fig1.update_layout( font=dict(size=18) )

    st.plotly_chart(fig1, use_container_width=False)

with tab2:

    tab2.subheader("Matplotlib & Seaborn")

    fig, ax = plt.subplots(dpi=300)
    fig = plot_category(df, cat_axis, top_n=10, figsize=None)
    st.pyplot(fig)

    fig, ax = plt.subplots(dpi=300)
    fig = plot_category(df, cat_axis, top_n=2, figsize=None)
    st.pyplot(fig)


# from streamlit_echarts import st_pyecharts
# import streamlit.components.v1 as components
#eplot.set_config(return_type='CHART') # this is important to retrieve the Dict spec of your chart
with tab3:
    tab3.subheader("Pyechart")
    st.markdown("**Does not seem to be very stable, e.g https://morioh.com/p/8d76612fcafc**")
    # V= df['continent'].value_counts()
    # chart = V.eplot.bar()
    # st_pyecharts(chart) # with my little https://github.com/andfanilo/streamlit-echarts
    # components.html(chart.render_embed(), width=800, height=500) # using components.html, though you need to specify size of chart so it doesn't break

# tab2.write(data)

import altair as alt
from altair import *
with tab4:
    tab4.subheader("Altair")
    chart = alt.Chart(df).mark_circle().encode(
    x=num_axis, y=num_axis, color=cat_axis, tooltip=[num_axis])#, size='c', color='c', tooltip=['a', 'b', 'c'])

    chart.encode(Y('Miles_per_Gallon', scale=Scale(domain=[0, 10000])))
    chart.configure_view(height=100, width=400)
    
    st.altair_chart(chart, use_container_width=True)

with tab5:
    tab5.subheader("Bokeh-HoloViz")
    #import holoviews as hv
    #from holoviews import opts
    #hv.extension('bokeh')
    #import hvplot.pandas # noqa: adds hvplot method to pandas objects

    #hist = df.hvplot.hist('lifeExp', by='Continent', legend=False, alpha=0.5, responsive=True, min_height=300)
    #hist
    #df.hvplot.hist(y=num_axis, bin_range=(0, 10), bins=50)

    #istg = df.hist(dimension='lifeExp', groupby='Continent')#, bin_range=(9, 46), bins=40, adjoin=False)

    #histg.opts(opts.Histogram(alpha=0.9, width=600))

    from bokeh.plotting import figure
    #from bokeh.charts import Scatter
    #from bokeh.io import output_notebook, show
    
    p = figure(
        title='simple line example',
        x_axis_label=num_axis,
        y_axis_label=num_axis)
    #p = Scatter(df, x=num_axis, y=num_axis, color=cat_axis)
    p.line(df[num_axis], df[num_axis], legend_label=num_axis, line_width=4)
               #toolbar_location=None, tools="")
     
    st.bokeh_chart(p, use_container_width=True) 

with tab6:
     tab6.subheader("ViSad")
     st.markdown('1. http://visad.ssec.wisc.edu/ (main page)')
     st.markdown('2. Visad tutorial (last update 2003): https://www.ssec.wisc.edu/~billh/tutorial/index.html')
     st.markdown('3. https://www.ssec.wisc.edu/mcidas/software/v/')
     st.markdown('4. https://studylib.net/doc/12410382/utilizing-python-as-a-scripting-language-for-the-mcidas-v')