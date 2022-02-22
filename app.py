import dash
#import dash_core_components as dcc
from dash import dcc
#import dash_html_components as html
from dash import html
from dash.dependencies import Input, Output, State, ClientsideFunction
#import dash_core_components as dcc
from dash import dcc
import plotly.express as px
import plotly.graph_objects as go
import  plotly as py
import pandas as pd
import sys
import numpy as np
import os
import geopandas as gpd
import pandas as pd
#from dask import dataframe as dd
import plotly.express as px
import  plotly as py
from plotly.subplots import make_subplots
import os
from pyproj import Transformer
from shapely.geometry import Point
import math
import random
from datetime import datetime, date, timedelta, time
import queue
import threading
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

#import vaex

ZIPS = ['33510', '33511', '33527', '33534', '33547', '33548', '33549', '33556', '33558', '33559', '33563', '33565',
            '33566', '33567', '33569', '33570', '33572', '33573', '33578', '33579', '33584', '33592', '33594', '33596',
            '33598', '33602', '33603', '33604', '33605', '33606', '33607', '33609', '33610', '33611', '33612', '33613',
            '33614', '33615', '33616', '33617', '33618', '33619', '33624', '33625', '33626', '33629', '33634', '33635',
            '33637', '33647']
default_zipcode ="33510"

MAX_ROWS=10000000
path = os.path.join('..', 'EDEN-ABM-Simulator', 'SimulationEngine', 'output', '2021-12-29', 'run4')

#path = os.path.join('..', 'EDEN-ABM-Simulator-old', 'SimulationEngine', 'output', '2021-12-10', 'run12')
#path2 = os.path.join('..', 'EDEN-ABM-Simulator', 'SimulationEngine', 'output', '2021-12-21', 'run2')
print(path)
#print(path2)

SF1 = 22
SF2 = 5

#CHUNK_SIZE1=2000000
#CHUNK_SIZE2=2000000
#CHUNK_SIZE1=4000000
#CHUNK_SIZE2=4000000

# scatter
#SKIP_EVERY_NTH_1=100 # best at 2
#SKIP_EVERY_NTH_1=100 # best at 2
SAMPLING_PERCENT_1=0.2 # default 0.25

# heatmap
#SKIP_EVERY_NTH_2=10 # best at 2
#SKIP_EVERY_NTH_2=1 # best at 2
SAMPLING_PERCENT_2=0.25 # default 0.5

startdate = date(2020, 3, 1)
enddate = date(2021, 8, 31)
def load_scatter(zipcode):
    return load_scatter_mongodb(zipcode)
    #return load_scatter_read_parquet(zipcode)

def load_heatmap(zipcode):
    return load_heatmap_mongodb(zipcode)
    #return load_heatmap_read_parquet(zipcode)
'''
legend_map={
    "susceptible":"blue",
    "asymptomatic":"purple",
    "vaccinated":"olive",
    "boosted":"olive",
    "recovered":"green",
    "critical": "red",
    "dead": "black",
    "exposed": "orange",
    "mild": "red",
    "presymptomatic": "red",
    "severe":"red"}
'''
legend_map={
    "susceptible":"blue",
    "asymptomatic":"purple",
    "vaccinated":"olive",
    "boosted":"olive",
    "recovered":"green",
    "critical": "#F1948A",
    "dead": "black",
    "exposed": "orange",
    "mild": "#F5B7B1",
    "presymptomatic": "#F2D7D5",
    "severe":"#EC7063"}

def plot(min, mean, max):
    sub_groups = ['Cases', 'Admissions', 'Deaths']
    fig = make_subplots(rows=3, cols=1, subplot_titles=sub_groups
                        )
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['cases']/SF1,
                             name="cases", line=dict({'width': 2, 'color': 'red'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=min['date'], y=min['cases']/(SF1/2),
                             name="cases", fillcolor='rgba(255,153,204,0.5)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(255,153,204,1)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=max['date'], y=max['cases']/SF1,
                             name="cases", fillcolor='rgba(255,153,204,0.5)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(255,153,204,1)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vcases'],
                             name="vcases", line=dict({'width': 2, 'color': 'blue', 'dash': 'dot'})), row=1, col=1)

    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['admissions']/SF1,
                             name="admissions", line=dict({'width': 2, 'color': 'green'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=min['date'], y=min['admissions']/(SF1/2),
                             name="admissions", fillcolor='rgba(217, 255, 217,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(204,255,204,1)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=max['date'], y=max['admissions']/SF1,
                             name="admissions", fillcolor='rgba(217, 242, 217,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(204,255,204,1)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vadmissions'],
                             name="actual admissions", line=dict({'width': 2, 'color': 'blue', 'dash': 'dot'})), row=2, col=1)


    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['deaths'] / SF2,
                             name="deaths", line=dict({'width': 2, 'color': 'black'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=min['deaths'] / (SF2 / 2),
                             name="deaths", fillcolor='rgba(230, 230, 230,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(192,192,192,1)'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=max['deaths'] / (SF2 * 2),
                             name="deaths", fillcolor='rgba(230, 230, 230,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(192,192,192,1)'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vdeaths'],
                             name="actual deaths", line=dict({'width': 2, 'color': 'blue', 'dash': 'dot'})), row=3, col=1)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, autosize=True, width=800, height=900,
                      legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    #fig.show()
    return fig


def plot_age(df):
    sub_groups = ['Cases >65', 'Admissions >65', 'Deaths >65', 'Cases >18', 'Admissions >18', 'Deaths >18',
                  'Cases >1', 'Admissions >1', 'Deaths >1']
    fig = make_subplots(rows=3, cols=3, subplot_titles=sub_groups
                        )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_65']/SF1,
                             name="cases >65", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=1,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_65']/SF1,
                             name="admissions >65", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_65']/SF2,
                             name="deaths >65", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_18']/SF1,
                             name="cases >18", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_18']/SF1,
                             name="admissions >18", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_18']/SF2,
                             name="deaths >18", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=2,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_1']/SF1,
                             name="cases >1", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=3,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_1']/SF1,
                             name="admissions >1", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_1']/SF2,
                             name="deaths >1", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=3,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(showlegend=True, autosize=True, width=950, height=1100,
                      legend=dict(orientation="h", x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyAge-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    #fig.show()
    return fig


def plot_gender(df):
    sub_groups = ['Cases Male', 'Admissions Male', 'Deaths Male',
                  'Cases Female', 'Admissions Female', 'Deaths Female']
    fig = make_subplots(rows=3, cols=3, subplot_titles=sub_groups
                        )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_male']/SF1,
                             name="cases Male", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=1,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_male']/SF1,
                             name="admissions Male", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_male']/SF2,
                             name="deaths Male", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_female']/SF1,
                             name="cases Female", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_female']/SF1,
                             name="admissions Female", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_female']/SF2,
                             name="deaths Female", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=2,
                  col=3)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(showlegend=True, autosize=True, width=950, height=1000,
                      legend=dict(orientation="h", x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyAge-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    #fig.show()
    return fig


def plot_race(df):
    sub_groups = ['Cases (white)', 'Admissions (white)', 'Deaths (white)',
                  'Cases (black)', 'Admissions (black)', 'Deaths (black)',
                  'Cases (asian)', 'Admissions (asian)', 'Deaths (asian)',
                  'Cases (other)', 'Admissions (other)', 'Deaths (other)',
                  'Cases (two)', 'Admissions (two)', 'Deaths (two)'
                  ]
    fig = make_subplots(rows=5, cols=3, subplot_titles=sub_groups
                        )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_white']/SF1,
                             name="Cases (white)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=1,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_white']/SF1,
                             name="admissions (white)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_white']/SF2,
                             name="deaths (white)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=1,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_black']/SF1,
                             name="cases (black)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_black']/SF1,
                             name="admissions (black)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_black']/SF2,
                             name="deaths (black)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=2,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_asian']/SF1,
                             name="cases (asian)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=3,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_asian']/SF1,
                             name="admissions (asian)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_asian']/SF2,
                             name="deaths (asian)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=3,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_other']/SF1,
                             name="cases (other)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=4,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_other']/SF1,
                             name="admissions (other)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=4, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=4, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_other']/SF2,
                             name="deaths (other)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=4, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=4,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_two']/SF1,
                             name="cases (two)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=5,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_two']/SF1,
                             name="admissions (two)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=5, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=5, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_two']/SF2,
                             name="deaths (two)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=5, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=5,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True) #autorange=True
    fig.update_layout(showlegend=True, autosize=True, width=950, height=1700,
                      legend=dict(orientation="h", x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyRace-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    #fig.show()
    return fig


def plot_FPL(df):
    sub_groups = ['Cases (0-100)', 'Admissions (0-100)', 'Deaths (0-100)',
                  'Cases (100-150)', 'Admissions (100-150)', 'Deaths (100-150)',
                  'Cases (150-175)', 'Admissions (150-175)', 'Deaths (150-175)',
                  'Cases (175-200)', 'Admissions (175-200)', 'Deaths (175-200)',
                  'Cases (200-1800)', 'Admissions (200-1800)', 'Deaths (200-1800)'
                  ]
    fig = make_subplots(rows=5, cols=3, subplot_titles=sub_groups
                        )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_0-100']/SF1,
                             name="Cases (0-100)", line=dict({'width':2, 'color':'rgba(255,153,204,0.5)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash':'dot'})), row=1, col=1)
    #-------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_0-100']/SF1,
                             name="admissions (0-100)", line=dict({'width':2, 'color':'rgba(217, 255, 217,0.75)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})), row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_0-100']/SF2,
                             name="deaths (0-100)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_100-150']/SF1,
                             name="cases (100-150)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_100-150']/SF1,
                             name="admissions (100-150)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_100-150']/SF2,
                             name="deaths (100-150)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=2,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_150-175']/SF1,
                             name="cases (150-175)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=3,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_150-175']/SF1,
                             name="admissions (150-175)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_150-175']/SF2,
                             name="deaths (150-175)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=3,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_175-200']/SF1,
                             name="cases (175-200)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=4,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_175-200']/SF1,
                             name="admissions (175-200)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=4, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=4, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_175-200']/SF2,
                             name="deaths (175-200)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=4, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=4,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_200-1800']/SF1,
                             name="cases (200-1800)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=5,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_200-1800']/SF1,
                             name="admissions (200-1800)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=5, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=5, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_200-1800']/SF2,
                             name="deaths (200-1800)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=5, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=5,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
   # fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(showlegend=True, autosize=True, width=950, height=1700,
                      legend=dict(orientation="h",x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyFPL-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    #fig.show()
    return fig

def plot_old(min, mean, max):
    sub_groups = ['Daily Cases', 'Daily Admissions', 'Daily Deaths']
    fig = make_subplots(rows=3, cols=1, subplot_titles=sub_groups
                        )
    fig.add_trace(go.Scatter(mode='lines', x=min['date'], y=min['cases'],
                             name="min(cases)", fillcolor='rgba(255,255,255,0.5)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(245,153,204,1)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=max['date'], y=max['cases'],
                             name="max(cases)", fillcolor='rgba(235,153,204,0.5)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(245,153,204,1)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['cases'],
                             name="mean(cases)", line=dict({'width': 1, 'color': 'red'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=1, col=1)

    fig.add_trace(go.Scatter(mode='lines', x=min['date'], y=min['admissions'],
                             name="min(admissions)", fillcolor='rgba(255,255,255,0.5)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgb(0,205,0)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=max['date'], y=max['admissions'],
                             name="max(admissions)", fillcolor='rgba(217, 242, 217,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgb(0,205,0)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['admissions'],
                             name="mean(admissions)", line=dict({'width': 1, 'color': 'rgb(0,103,0)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})), row=2,  col=1)


    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['deaths'],
                             name="mean deaths", line=dict({'width': 1, 'color': 'black'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=3, col=1)
    # fig.update_traces(hoverinfo='text+name', mode='lines+markers')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, autosize=True, width=800, height=900,
                      legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    return fig

red_fillcolor='rgba(245,153,204,1)'
green_fillcolor='rgba(0,205,0,0.5)'
gray_fillcolor='rgba(120,120,120,0.5)'

def plot_age_old(df):
    sub_groups = ['Cases >65', 'Admissions >65', 'Deaths >65', 'Cases >18', 'Admissions >18', 'Deaths >18',
                  'Cases >1', 'Admissions >1', 'Deaths >1']
    fig = make_subplots(rows=3, cols=3, subplot_titles=sub_groups
                        )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_65'],
                             name="cases >65", line=dict({'width': 1.5, 'color': red_fillcolor})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=1,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_65'],
                             name="admissions >65", line=dict({'width': 1.5, 'color': green_fillcolor})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_65'],
                             name="deaths >65", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_18'],
                             name="cases >18", line=dict({'width': 1.5, 'color': red_fillcolor})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_18'],
                             name="admissions >18", line=dict({'width': 1.5, 'color': green_fillcolor})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_18'],
                             name="deaths >18", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=2,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_1'],
                             name="cases >1", line=dict({'width': 1.5, 'color': red_fillcolor})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=3,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_1'],
                             name="admissions >1", line=dict({'width': 1.5, 'color': green_fillcolor})), row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_1'],
                             name="deaths >1", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=3,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(showlegend=True, autosize=True, width=950, height=1000,
                      legend=dict(orientation="h", x=0, y=-0.2, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyAge-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    # fig.show()
    return fig

def plot_gender_old(df):
    sub_groups = ['Cases (Male)', 'Admissions (Male)', 'Deaths (Male)',
                  'Cases (Female)', 'Admissions (Female)', 'Deaths (Female)']
    fig = make_subplots(rows=3, cols=3, subplot_titles=sub_groups
                        )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_male'],
                             name="cases (Male)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=1,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_male'],
                             name="admissions (Male)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_male'],
                             name="deaths (Male)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_female'],
                             name="cases (Female)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_female'],
                             name="admissions (Female)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_female'],
                             name="deaths (Female)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})),
                  row=2,
                  col=3)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(showlegend=True, autosize=True, width=950, height=900,
                      legend=dict(orientation="h", x=0, y=0, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyAge-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    # fig.show()
    return fig

def plot_race_old(df):
    sub_groups = ['Cases (White)', 'Admissions (White)', 'Deaths (White)',
                  'Cases (Black/African American)', 'Admissions (Black/African American)', 'Deaths (Black/African American)',
                  'Cases (Asian)', 'Admissions (Asian)', 'Deaths (Asian)',
                  'Cases (Some other race)', 'Admissions (Some other race)', 'Deaths (Some other race)',
                  'Cases (Two or more races)', 'Admissions (Two or more races)', 'Deaths (Two or more races)'
                  ]
    fig = make_subplots(rows=5, cols=3, subplot_titles=sub_groups
                        )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_white'],
                             name="Cases (White)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=1,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_white'],
                             name="admissions (White)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_white'],
                             name="deaths (White)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=1,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_black'],
                             name="cases (Black/African American)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_black'],
                             name="admissions (Black/African American)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_black'],
                             name="deaths (Black/African American)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=2,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_asian'],
                             name="cases (Asian)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=3,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_asian'],
                             name="admissions (Asian)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_asian'],
                             name="deaths (Asian)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=3,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_other'],
                             name="cases (Some other race)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=4,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_other'],
                             name="admissions (Some other race)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=4, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=4, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_other'],
                             name="deaths (Some other race)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=4, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=4,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_two'],
                             name="cases (Two or more races)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=5,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_two'],
                             name="admissions (Two or more races)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=5, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=5, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_two'],
                             name="deaths (Two or more races)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=5, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=5,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True) #autorange=True
    fig.update_layout(showlegend=True, autosize=True, width=950, height=1700,
                      legend=dict(orientation="h", x=0, y=-0.1, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyRace-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    # fig.show()
    return fig

def plot_FPL_old(df):
    sub_groups = ['Cases (0-100)', 'Admissions (0-100)', 'Deaths (0-100)',
                  'Cases (100-150)', 'Admissions (100-150)', 'Deaths (100-150)',
                  'Cases (150-175)', 'Admissions (150-175)', 'Deaths (150-175)',
                  'Cases (175-200)', 'Admissions (175-200)', 'Deaths (175-200)',
                  'Cases (200-1800)', 'Admissions (200-1800)', 'Deaths (200-1800)'
                  ]
    fig = make_subplots(rows=5, cols=3, subplot_titles=sub_groups
                        )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_0-100'],
                             name="Cases (0-100)", line=dict({'width':1.5, 'color':red_fillcolor})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash':'dot'})), row=1, col=1)
    #-------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_0-100'],
                             name="admissions (0-100)", line=dict({'width': 1.5, 'color':green_fillcolor})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})), row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_0-100'],
                             name="deaths (0-100)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_100-150'],
                             name="cases (100-150)", line=dict({'width': 2, 'color': red_fillcolor})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_100-150'],
                             name="admissions (100-150)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_100-150'],
                             name="deaths (100-150)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=2,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_150-175'],
                             name="cases (150-175)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=3,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_150-175'],
                             name="admissions (150-175)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_150-175'],
                             name="deaths (150-175)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=3,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_175-200'],
                             name="cases (175-200)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=4,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_175-200'],
                             name="admissions (175-200)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=4, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=4, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_175-200'],
                             name="deaths (175-200)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=4, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=4,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_200-1800'],
                             name="cases (200-1800)", line=dict({'width': 1.5, 'color': red_fillcolor})), row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1.5, 'color': 'red', 'dash': 'dot'})), row=5,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_200-1800'],
                             name="admissions (200-1800)", line=dict({'width': 1.5, 'color': green_fillcolor})), row=5, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1.5, 'color': 'green', 'dash': 'dot'})),
                  row=5, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_200-1800'],
                             name="deaths (200-1800)", line=dict({'width': 1.5, 'color': gray_fillcolor})), row=5, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1.5, 'color': 'black', 'dash': 'dot'})), row=5,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
   # fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(showlegend=True, autosize=True, width=950, height=1700,
                      legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyFPL-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    # fig.show()
    return fig
    
def get_RMSE(zip, actual, simulated):
    mse = mean_squared_error(actual, simulated)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, simulated)
    return rmse


def load_SEIR(mode):
    if mode=='All cases-Not filtered':
        dlist = []
        for root, dirs, files in os.walk(path):
        #for root, dirs, files in os.walk(path2):
            for file in files:
                if file.startswith("plot_"):
                    #print(file)
                    no = file.split('_')[1].split('.')[0]
                    d = pd.read_csv(os.path.join(root, file))
                    # d['chunk'] = no
                    #if get_RMSE(no, d['vcases'], d['cases']) < 400:
                    dlist.append(d)

        print('All plots-Loading completed!')
        plotdf = pd.concat(dlist, axis=1)
        dates = dlist[0]['date'].tolist()
        plotdf.drop('date', axis=1, inplace=True)

        max = plotdf.groupby(plotdf.columns, axis=1).max()
        mean = plotdf.groupby(plotdf.columns, axis=1).mean()
        min = plotdf.groupby(plotdf.columns, axis=1).min()
        max['date'] = dates
        mean['date'] = dates
        min['date'] = dates
        #print('Loading completed!')
        return plot(min, mean, max)
    else:
        dlist = []
        for root, dirs, files in os.walk(path):
        #for root, dirs, files in os.walk(path2):
            for file in files:
                if file.startswith("plot_"):
                    no = file.split('_')[1].split('.')[0]
                    d = pd.read_csv(os.path.join(root, file))
                    d['chunk'] = no
                    dlist.append(d)

        print('Filtered plots-Loading completed!')
        plotdf = pd.concat(dlist)
        plotdf = plotdf.sort_values(by='date')

        if mode == 'By Age':
            return plot_age(plotdf)
        elif mode == 'By Gender':
            return plot_gender(plotdf)
        elif mode == 'By Race':
            return plot_race(plotdf)
        elif mode == 'By FPL':
            return plot_FPL(plotdf)

"""
load_scatter using_read_parquet
"""
def load_scatter_read_parquet(zipcode=default_zipcode):
    if zipcode is None:
        zipcode=default_zipcode
    print("zipcode:", zipcode)
    columns_being_used_in_scatter=[
        'step',
        #'pid',
        'x',
        'y',
        #'location',
        #'ZIP',
        #'type',
        'state',
        #'color',
    ]
    #tp = vaex.from_csv(os.path.join(path, 'scatter.csv'), copy_index=False)
 
    #tp = pd.read_csv(os.path.join(path, 'scatter.csv'), iterator=True, chunksize=CHUNK_SIZE1, skiprows=lambda x: x % SKIP_EVERY_NTH_1)
    #pdf = pd.concat(tp, ignore_index=True)

    #pdf = dd.read_csv(os.path.join(path, 'scatter.csv'))

    #pdf=pdf.drop(['pid', 'location', 'ZIP', 'type'], axis=1)
    t1=datetime.now()
    pdf = pd.read_parquet(os.path.join(path, 'scatter.parquet'),
                    filters=[('ZIP','in', [zipcode])],
                    columns=columns_being_used_in_scatter,
    )
    print("scatter read_parquet time spent", datetime.now()-t1)
    pdf["state"] =  pdf["state"].astype("category")
    #pdf["step"] =  pdf["step"].astype("category")
    pdf["step"] = pd.to_numeric(pdf["step"], downcast="unsigned")
    pdf[['x','y']] = pdf[['x','y']].apply(pd.to_numeric, downcast="float")

    #datelist = pd.date_range(startdate, enddate).tolist()
    pdf['date']=[startdate+timedelta(days=d) for d in pdf['step']]
    pdf['date']=pdf['date'].astype(str)
    
    #scatter=db.scatter
    #list_scatter=list(scatter.find({}))
    ####list_scatter=list(scatter.find({}))[::100] # every other item
    #print(len(list_scatter))
    #pdf = pd.DataFrame(list_scatter)

    print('scatter data size(before '+ str(SAMPLING_PERCENT_1) +' sampling)', pdf.size)
    pdf = pdf.sample(frac=SAMPLING_PERCENT_1) # (???) similar to geting every 4th rows
    print('scatter data size(after '+ str(SAMPLING_PERCENT_1) +' sampling)', pdf.size)

    #print('scatter memory usage', pdf.info())
    print('scatter memory size(MB)', sys.getsizeof(pdf)/(1024*1024))
    pdf = pdf.sort_values(by='step') # should be run after sampling
    pdf.drop('step',1, inplace=True)

    fig = px.scatter_mapbox(pdf,
                            #title="Scatter_Map",
                            #color='color',
                            #animation_frame='step',
                            animation_frame='date',
                            #animation_group='date',
                            color='state',
                            #text='state',
                            color_discrete_map=legend_map,
                            lat='y',
                            lon='x',                            
                            # lat=pdf['y'],
                            # lon=pdf['x'],
                            zoom=9, #default 8 (0-20)
                            height=800,
                            width=1000,
                            center=dict(lat=28.03711, lon=-82.46390),
                            #mapbox_style='open-street-map',
                            #mapbox_style='carto-darkmatter',
                            mapbox_style='carto-positron',
    )
    # Add the datashader image as a mapbox layer image
    '''
    fig.update_layout(mapbox_style='carto-positron',
                    #mapbox_style="carto-darkmatter",
                    mapbox_layers = [
                    {
                        "sourcetype": "image",
                        "source": img,
                        "coordinates": coordinates
                    }]
    )
    '''
    #fig.update_traces(marker=dict(size=10))
    #fig.update_traces(marker=dict(size=6))
    fig.update_layout(legend=dict(
        orientation="h",
        xanchor="left",
        yanchor="bottom",
        x=0,
        y=-0.1,
        #title_font_family="Times New Roman",
        font=dict(
            family="Courier",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
        )
    )
    return fig

import pymongo
from pymongo import MongoClient
client = MongoClient()
print(client)
db = client["abm"]
"""
load_scatter via_mongodb
"""
def load_scatter_mongodb(zipcode=default_zipcode):
    #tp = vaex.from_csv(os.path.join(path, 'scatter.csv'), copy_index=False)
    if zipcode is None:
        zipcode=default_zipcode
    print("zipcode:", zipcode)

    #tp = pd.read_csv(os.path.join(path, 'scatter.csv'), iterator=True, chunksize=CHUNK_SIZE1, skiprows=lambda x: x % SKIP_EVERY_NTH_1)
    #pdf = pd.concat(tp, ignore_index=True)

    #pdf = dd.read_csv(os.path.join(path, 'scatter.csv'))

    #pdf=pdf.drop(['pid', 'location', 'ZIP', 'type'], axis=1)

    t1=datetime.now()
    list_scatter=list(db.scatter.find({"ZIP":int(zipcode)}))
    print("mongodb search result for ", zipcode, len(list_scatter),"time spent", datetime.now()-t1)
    pdf = pd.DataFrame(list_scatter)

    pdf["state"] =  pdf["state"].astype("category")
    #pdf["step"] =  pdf["step"].astype("category")
    pdf["step"] = pd.to_numeric(pdf["step"], downcast="unsigned")
    pdf[['x','y']] = pdf[['x','y']].apply(pd.to_numeric, downcast="float")

    #datelist = pd.date_range(startdate, enddate).tolist()
    pdf['date']=[startdate+timedelta(days=d) for d in pdf['step']]
    pdf['date']=pdf['date'].astype(str)
    

    print('scatter data size(before '+ str(SAMPLING_PERCENT_1) +' sampling)', pdf.size)
    pdf = pdf.sample(frac=SAMPLING_PERCENT_1) # (???) similar to geting every 4th rows
    print('scatter data size(after '+ str(SAMPLING_PERCENT_1) +' sampling)', pdf.size)

    #print('scatter memory usage', pdf.info())
    print('scatter memory size(MB)', sys.getsizeof(pdf)/(1024*1024))
    pdf = pdf.sort_values(by='step') # should be run after sampling
    pdf.drop('step',1, inplace=True)

    fig = px.scatter_mapbox(pdf,
                            #title="Scatter_Map",
                            #color='color',
                            #animation_frame='step',
                            animation_frame='date',
                            #animation_group='date',
                            color='state',
                            #text='state',
                            color_discrete_map=legend_map,
                            lat='y',
                            lon='x',                            
                            # lat=pdf['y'],
                            # lon=pdf['x'],
                            zoom=9, #default 8 (0-20)
                            height=800,
                            width=1000,
                            center=dict(lat=28.03711, lon=-82.46390),
                            #mapbox_style='open-street-map',
                            #mapbox_style='carto-darkmatter',
                            mapbox_style='carto-positron',
    )
    # Add the datashader image as a mapbox layer image
    '''
    fig.update_layout(mapbox_style='carto-positron',
                    #mapbox_style="carto-darkmatter",
                    mapbox_layers = [
                    {
                        "sourcetype": "image",
                        "source": img,
                        "coordinates": coordinates
                    }]
    )
    '''
    #fig.update_traces(marker=dict(size=10))
    #fig.update_traces(marker=dict(size=6))
    fig.update_layout(legend=dict(
        orientation="h",
        xanchor="left",
        yanchor="bottom",
        x=0,
        y=-0.1,
        #title_font_family="Times New Roman",
        font=dict(
            family="Courier",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
        )
    )
    return fig
"""
load_heatmap using_read_parquet
"""
def load_heatmap_read_parquet(zipcode=default_zipcode):
    columns_being_used_in_heatmap=[
        'step',
        'x',
        'y',
        'z',
        #'zip',
    ]
    #tp = pd.read_csv(os.path.join(path, 'heatmap.csv'), iterator=True, chunksize=CHUNK_SIZE2, skiprows=lambda x: x % SKIP_EVERY_NTH_2)
    #pdf = pd.concat(tp, ignore_index=True)
    #pdf = dd.read_csv(os.path.join(path, 'heatmap.csv'))
    #pdf = pdf.drop(['zip'], axis=1)
    t1=datetime.now()
    pdf = pd.read_parquet(os.path.join(path, 'heatmap.parquet'),
                    filters=[('zip','in', [zipcode])],
                    columns=columns_being_used_in_heatmap,
    )
    print("heatmap read_parquet time spent", datetime.now()-t1)
    #pdf["step"] =  pdf["step"].astype("category")
    pdf["step"] = pd.to_numeric(pdf["step"], downcast="unsigned")
    pdf[['x','y']] = pdf[['x','y']].apply(pd.to_numeric, downcast="float")
    pdf["z"] = pd.to_numeric(pdf["z"], downcast="unsigned")

    #datelist = pd.date_range(startdate, enddate).tolist()
    pdf['date']=[startdate+timedelta(days=d) for d in pdf['step']]
    pdf['date']=pdf['date'].astype(str)

    #heatmap=db.heatmap
    #list_heatmap=list(heatmap.find({}))
    #list_heatmap=list(heatmap.find({}))[::10]
    #print(len(list_heatmap))
    #pdf = pd.DataFrame(list_heatmap)
    print('heatmap data size(before '+ str(SAMPLING_PERCENT_2) +' sampling)', pdf.size)
    pdf = pdf.sample(frac=SAMPLING_PERCENT_2)
    print('heatmap data size(after '+ str(SAMPLING_PERCENT_2) +' sampling)', pdf.size)
    #print('heatmap memory size', pdf.info())
    print('heatmap memory usage(MB)', sys.getsizeof(pdf)/(1024*1024))
    pdf = pdf.sort_values(by='step')
    pdf.drop('step',1, inplace=True)

    fig = px.density_mapbox(pdf,
                            color_continuous_scale='RdYlGn_r',
                            lat=pdf['y'],
                            lon=pdf['x'],
                            z=pdf['z'],
                            #animation_frame=pdf['step'],
                            animation_frame='date',
                            zoom=9,
                            opacity=0.75,
                            height=800,
                            width=1000,
                            center=dict(lat=28.03711, lon=-82.46390),
                            # mapbox_style='open-street-map'
                            mapbox_style='stamen-terrain')
    return fig

"""
load_heatmap using_mongodb
"""
def load_heatmap_mongodb(zipcode=default_zipcode):
    if zipcode is None:
        zipcode=default_zipcode
    print(zipcode)
    columns_being_used_in_heatmap=[
        'step',
        'x',
        'y',
        'z',
        #'zip',
    ]
    #tp = pd.read_csv(os.path.join(path, 'heatmap.csv'), iterator=True, chunksize=CHUNK_SIZE2, skiprows=lambda x: x % SKIP_EVERY_NTH_2)
    #pdf = pd.concat(tp, ignore_index=True)
    #pdf = dd.read_csv(os.path.join(path, 'heatmap.csv'))
    #pdf = pdf.drop(['zip'], axis=1)

    # pdf = pd.read_parquet(os.path.join(path, 'heatmap.parquet'),
    #                 filters=[('zip','in', [zipcode])],
    #                 columns=columns_being_used_in_heatmap,
    # )
    t1=datetime.now()
    list_heatmap=list(db.heatmap.find({"zip":int(zipcode)}))
    print("heatmap mongodbsearch result", len(list_heatmap),"time spent", datetime.now()-t1)
    pdf = pd.DataFrame(list_heatmap)

    # pdf["step"] =  pdf["step"].astype("category")
    pdf["step"] = pd.to_numeric(pdf["step"], downcast="unsigned")
    pdf[['x','y']] = pdf[['x','y']].apply(pd.to_numeric, downcast="float")
    pdf["z"] = pd.to_numeric(pdf["z"], downcast="unsigned")

    #datelist = pd.date_range(startdate, enddate).tolist()
    pdf['date']=[startdate+timedelta(days=d) for d in pdf['step']]
    pdf['date']=pdf['date'].astype(str)

    print('heatmap data size(before '+ str(SAMPLING_PERCENT_2) +' sampling)', pdf.size)
    pdf = pdf.sample(frac=SAMPLING_PERCENT_2)
    print('heatmap data size(after '+ str(SAMPLING_PERCENT_2) +' sampling)', pdf.size)
    #print('heatmap memory size', pdf.info())
    print('heatmap memory usage(MB)', sys.getsizeof(pdf)/(1024*1024))
    pdf = pdf.sort_values(by='step')
    pdf.drop('step',1, inplace=True)

    fig = px.density_mapbox(pdf,
                            color_continuous_scale='RdYlGn_r',
                            lat=pdf['y'],
                            lon=pdf['x'],
                            z=pdf['z'],
                            #animation_frame=pdf['step'],
                            animation_frame='date',
                            zoom=9,
                            opacity=0.75,
                            height=800,
                            width=1000,
                            center=dict(lat=28.03711, lon=-82.46390),
                            # mapbox_style='open-street-map'
                            mapbox_style='stamen-terrain')
    return fig

figure1 = load_SEIR('All cases-Not filtered')
print("Reading heatmap data...")
figure3 = load_heatmap(default_zipcode)
print("Reading scatter data...")
figure2 = load_scatter(default_zipcode)
print('SEIR/Scatter/Heatmap loading completed!')

colors = {
    #'background': '#111111',
    'text': '#318ce7'
}

tabs_styles = {
    'height': '44px',
    'align-items': 'center'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'border-radius': '15px',
    'background-color': '#F2F2F2',
    'box-shadow': '4px 4px 4px 4px lightgrey',
 
}
 
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'border-radius': '15px',
}


app = dash.Dash(__name__, title="COVID-19 Dashboard powered by EDEN (USF-COPH-Dr.Edwin Michael Lab)")

app.layout = html.Div(children=[
    html.Div(
        className="container",
        children=[
            html.Div(
                className="header",
                children=[
                    html.H2(children='Hillsborough County Health Care Services', style={'textAlign': 'center', 'color': colors['text']}),
                    html.H3("COVID-19 Dashboard using An Agent-based City Scale Digital Twin (EDEN)", style={'textAlign': 'center'}),
                ],
            ),
            html.Div(
                style={
                        #'backgroundColor':'darkslategray',
                        #'color':'lightsteelblue',
                        #'height':'100px',
                        'margin-left':'10px',
                        'width':'20%',
                        'text-align':'center',
                        'display':'inline-block'
                        },
                className="nav",
                children=[
                    # html.H4(
                    #     "(Coming soon) Select Forecast Range:",
                    #     className="control_label", style={'padding': 10, 'flex': 1}
                    # ),
                    # html.Br(),
                    # dcc.RangeSlider(
                    #     disabled=True,
                    #     id="year_slider",
                    #     min=0,
                    #     max=3,
                    #     step=None,
                    #     marks={
                    #         0: "2020",
                    #         1: "2021",
                    #         2: "2022",
                    #         3: "2023"
                    #     },
                    #     value=[0, 1],
                    #     #style={'padding': 10, 'flex': 1}
                    # ),
                    #html.Br(),
                    html.H4("(Coming soon) Social Distancing Measures:", className="control_label", style={'padding': 10, 'flex': 1}),
                    dcc.RadioItems(
                        id="social_distancing",
                        options=[{'disabled':True, 'label': i, 'value': i} for i in ['Current',
                                                                    '25% Increase(+)',
                                                                    '25% Decrease(-)']],

                        value="Current",
                        labelStyle={'display': 'block', 'text-align': 'left', 'margin-right': 20},
                        #labelStyle = {'display': 'inline-block', 'margin-right': 10},
                        style={'padding': 10, 'flex': 1}
                    ),
                    #html.Br(),
                    html.H4("(Coming soon) Vaccination rate:", className="control_label", style={'padding': 10, 'flex': 1}),
                    dcc.RadioItems(
                        id="vaccination_rate",
                        options=[{'disabled':True, 'label': i, 'value': i} for i in ['Current',
                                                                    '5% Increase(+)',
                                                                    '5% Decrease(-)']],

                        value="Current",
                        labelStyle={'display': 'block', 'text-align': 'left', 'margin-right': 20},
                        #labelStyle = {'display': 'inline-block', 'margin-right': 10},
                        style={'padding': 10, 'flex': 1}
                    ),
                    # html.H4("(Coming soon) COVID Variants:", className="control_label", style={'padding': 10, 'flex': 1}),
                    # dcc.RadioItems(
                    #     id="variants",
                    #     options=[{'disabled':True, 'label': i, 'value': i} for i in ['Current-Up to Delta', 'New-Omicron']],

                    #     value="Current",
                    #     labelStyle={'display': 'block', 'text-align': 'left', 'margin-right': 20},
                    #     #labelStyle = {'display': 'inline-block', 'margin-right': 10},
                    #     style={'padding': 10, 'flex': 1}
                    # ),
                    html.Br(),
                    html.Br(),
                    html.A("Contact Info.", href='https://health.usf.edu/publichealth/overviewcoph/faculty/edwin-michael', target="_blank"),
                    html.Br(),
                    html.Img(src=app.get_asset_url('usf-logo-white-bg.jfif'), style={'margin-left': 10, 'width':'200px'}),

                ]
            ),
            html.Div(
                style={
                        #'backgroundColor':'darkslategray',
                        #'color':'lightsteelblue',
                        #'height':'100px',
                        'margin-left':'10px',
                        'width':'70%',
                        'text-align':'center',
                        'display':'inline-block'
                        },
                className="section",
                children=[
                            dcc.Tabs(id="tabsgraph", value='moretab', children=[
                                dcc.Tab(label='About E.D.E.N.', value='moretab', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='COVID-19 Time Plots', value='graph1', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='COVID-19 Spatial Spread', value='graph2', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='COVID-19 Heatmap', value='graph3', style = tab_style, selected_style = tab_selected_style),
                            ], style = tabs_styles),
                            html.Div(id='tabs-contentgraph'),
                            html.Div(
                                className="footer",
                                children=[
                                    html.P('Paper: "An Agent-based City Scale Digital Twin (EDEN) for Pandemic Analytics and Scenario Planning, Imran Mahmood et al. (in publishing)"'),
                                    html.H3(children='* Simulation results are provided by Dr. Edwin Michael Lab, USF College of Public Health *'),
                                    html.H4('Team members: Edwin Michael (PI), Imran Mahmood Qureshi Hashmi, Yilian Alonso Otano, Soo I. Kim'),
                                    #html.Img(src=app.get_asset_url('usf-logo-white-color.svg'), style={'width':'20%'})
                                ]
                            ),                              
                ]
            ),

        ]
    ), 
])
time_plots_explain="""
Time plots are showing simulation results calculated using the zipcode-specific data until August 2021 which are provided by Hillsborough county Health Care Service. As soon as we obtain recent data, the plots will be updated."""

scatter_map_explain="""
This map shows the movements of agents on the map, where the color represent the current state of the individual.
These inviduals and their spread over time is represented by the spatial scatter plot. over the Hillsborough county map. During the simulation, a population is generated at the start and individuals, or agents, move across the space based on assigned individual characteristics, performing different activities and visiting different types of building where they can interact with an infected person and become exposed and later sick and infectious."""

heat_map_explain="The bar to the right of the map is a legend, assigning a color on a gradient based on the number of cases within a zip code. The zip code areas with the highest number of cases will be red and the zip code areas with the lowest number of cases will be dark green. A z-value is computed based on the aggerate cases at a certain location and represents the heat of that location. A higher z-value represents higher density of the cases at a given location."

@app.callback(Output('tabs-contentgraph', 'children'),
              Input('tabsgraph', 'value'))
def render_content(tab):
    if tab=='moretab':
        return html.Div([
            html.H3('(Except from ongoing paper)'),
            html.H4('An Agent-based City Scale Digital Twin (EDEN) for Pandemic Analytics and Scenario Planning" (Imran et al.)'),
            html.P('This paper presents the development of an agent-based city scale digital twin (EDEN) for the analysis and prediction of COVID-19 transmission across the population at City or County scale. EDEN is a Python based open-source geo-spatial, agent-based, parallel simulation framework. It incorporates GIS data, epidemiological disease parameters and multi-layered, multi-resolution synthetic populations for the study of infectious disease spread (e.g., COVID-19) in a geospatial virtual environment of a selected region. It models the transmission of a selected contagion and simulates its outbreak using computational disease dynamics at a selected spatial resolution (e.g., neighborhood, census tract, zip code, city, county, state, and or country). It forecasts the spread of infections over time, identify spatial hot spots across the region, and estimate the number of infected patients arriving at hospitals. It further allows to apply different public health interventions and to evaluate various lock down scenarios and counterfactuals, thus help critical decision making in rapid emergency response management.'),
            html.H4('Full text will be available here when published. For more info, contact Dr. Edwin Michael (emichael443@usf.edu).'),
            html.Br(),
            html.Br(),
            html.Img(src=app.get_asset_url('USF-EMichael-ABM-EDEN.png'), style={'margin-left': 2, 'width':'497px'}),
            #html.Br(),
            #html.Br(),
            #html.Img(src=app.get_asset_url('usf-logo-white-bg.jfif'), style={'margin-left': 10, 'width':'200px'}),
        ])
    elif tab == 'graph1':
        return html.Div([
            html.Br(),
            html.P(time_plots_explain),
            html.H4("Time Plots Filters:", className="control_label", style={'padding': 10, 'flex': 1}),
            dcc.RadioItems(
                id="filter_type",
                options=[{'label': i, 'value': i} for i in ['All cases-Not filtered',
                                                            'By Age',
                                                            'By Gender',
                                                            'By Race',
                                                            'By FPL']],

                value="All cases-Not filtered",
                #labelStyle={'display': 'block', 'text-align': 'left', 'margin-right': 20},
                #labelStyle = {'display': 'inline-block', 'margin-right': 10},
                #style={'padding': 10, 'flex': 1}
                inline=True,
            ),
            html.P("* FPL: Federal Poverty Level (%)", style={'padding': 10, 'flex': 1}),
            #html.A("Federal Poverty Level", href='https://www.healthcare.gov/glossary/federal-poverty-level-fpl/', target="_blank", style={'padding': 10, 'flex': 1}),
            #html.Br(),

            dcc.Graph(
                id='graph1',
                figure=figure1
            )
        ])
    elif tab == 'graph2':
        return html.Div([      
            html.Br(),
            html.H2("Spatial plot of individual daily case emergence and spread"),
            html.P(scatter_map_explain),
            #html.P("(Steps equals Days starting March 1, 2020.)"),
            html.P("Note: For the fast web response, only a fraction of data is being used here. For best result, please contact us. This page uses "+str(SAMPLING_PERCENT_1*100)+" %", style={'textAlign': 'center', 'color':'orange'}),
            dcc.Dropdown(
                id="zipcode",
                options=ZIPS,
                value=default_zipcode,
                style=dict(width='40%',
                    display='inline-block',
                    verticalAlign="middle"
                ),
                #style={'margin-right': 10, 'padding': 1, 'flex': 1}
                #style={"width": "200px", 'display':'flex', 'align-items':'center','justify-content':'center'},
            ),            
            dcc.Graph(
                id='graph2',
                figure=figure2
            ),
        ])
    elif tab == 'graph3':
        return html.Div([
            html.Br(),
            html.H2("Heatmap of the density of daily infectious cases"),
            #html.P("(Z values represents the number of cases within the same zipcode area.)"),
            #html.P("(Steps equals Days starting March 1, 2020)"),
            html.P(heat_map_explain),
            html.P("Note: For the fast web response, only a fraction of data is being used here. For best result, please contact us. This page uses "+str(SAMPLING_PERCENT_2*100)+" %", style={'textAlign': 'center', 'color':'orange'}),
            dcc.Dropdown(
                id="zipcode_heatmap",
                options=ZIPS,
                value=default_zipcode,
                style=dict(width='40%',
                    display='inline-block',
                    verticalAlign="middle"
                ),
                #style={'margin-right': 10, 'padding': 1, 'flex': 1}
                #style={"width": "200px", 'display':'flex', 'align-items':'center','justify-content':'center'},
            ),            
            dcc.Graph(
                id='graph3',
                figure=figure3
            )
        ])

@app.callback(
    Output("graph1", 'figure'), Input("filter_type", "value"))
def update_SEIR(filter_type):
    figure1 = load_SEIR(filter_type)
    return figure1

#@app.callback(Output("graph2", 'figure'), Input("zipcode", "value"))
@app.callback(Output("graph2", 'figure'), Input("zipcode", "value"))
def update_scatter_by_zipcode(zipcode):
    time.sleep(1)
    figure2 = load_scatter(zipcode)
    return figure2

#@app.callback(Output("graph2", 'figure'), Input("zipcode", "value"))
@app.callback(Output("graph3", 'figure'), Input("zipcode_heatmap", "value"))
def update_heatmap_by_zipcode(zipcode_heatmap):
    time.sleep(1)
    figure3 = load_heatmap(zipcode_heatmap)
    return figure3
if __name__ == '__main__':
    app.run_server(debug=False,host="0.0.0.0",port=8050)

