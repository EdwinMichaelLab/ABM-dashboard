from collections import OrderedDict

#from zmq import VERSION_PATCH
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

# import pymongo
# from pymongo import MongoClient
# client = MongoClient()
# print(client)
# db = client["abm"]

from flask_caching import Cache

ZIPS = ['33510', '33511', '33527', '33534', '33547', '33548', '33549', '33556', '33558', '33559', '33563', '33565',
            '33566', '33567', '33569', '33570', '33572', '33573', '33578', '33579', '33584', '33592', '33594', '33596',
            '33598', '33602', '33603', '33604', '33605', '33606', '33607', '33609', '33610', '33611', '33612', '33613',
            '33614', '33615', '33616', '33617', '33618', '33619', '33624', '33625', '33626', '33629', '33634', '33635',
            '33637', '33647']

ZIPS_centers={ '33510':[27.96, -82.30], 
                '33511':[27.90, -82.30],
                '33527':[27.97, -82.22],
                '33534':[27.83, -82.38],
                '33547':[27.8, -82.1],
                '33548':[28.15, -82.48],
                '33549':[28.14, -82.45],
                '33556':[28.2, -82.6],
                '33558':[28.16, -82.51],
                '33559':[28.16, -82.40],
                '33563':[28.02, -82.13],
                '33565':[28.10, -82.15],
                '33566':[27.99, -82.13],
                '33567':[27.91, -82.12],
                '33569':[27.85, -82.29],
                '33570':[27.70, -82.47],
                '33572':[27.77, -82.40],
                '33573':[27.73, -82.36],
                '33578':[27.84, -82.35],
                '33579':[27.80, -82.28],
                '33584':[28.00, -82.29],
                '33592':[28.10, -82.28],
                '33594':[27.94, -82.24],
                '33596':[27.89, -82.23],
                '33598':[27.7, -82.3],
                '33602':[27.95, -82.46],
                '33603':[27.99, -82.46],
                '33604':[28.01, -82.45],
                '33605':[27.94, -82.43],
                '33606':[27.93, -82.46],
                '33607':[27.96, -82.54],
                '33609':[27.94, -82.52],
                '33610':[28.00, -82.38],
                '33611':[27.89, -82.51],
                '33612':[28.05, -82.45],
                '33613':[28.09, -82.45],
                '33614':[28.01, -82.50],
                '33615':[28.00, -82.58],
                '33616':[27.86, -82.53],
                '33617':[28.04, -82.39],
                '33618':[28.08, -82.50],
                '33619':[27.90, -82.38],
                '33624':[28.08, -82.52],
                '33625':[28.07, -82.56],
                '33626':[28.06, -82.61],
                '33629':[27.92, -82.51],
                '33634':[28.00, -82.54],
                '33635':[28.02, -82.61],
                '33637':[28.05, -82.36],
                '33647':[28.12, -82.35]}
MAX_ROWS=10000000
path = os.path.join('..', 'EDEN-ABM-Simulator', 'SimulationEngine', 'output', '2021-12-29', 'run4')
#path = os.path.join('..', 'EDEN-ABM-Simulator-new', 'SimulationEngine', 'output', '2021-12-29', 'run4')
print(path)

SF1 = 22
SF2 = 5

#CHUNK_SIZE1=2000000
#CHUNK_SIZE2=2000000

# scatter
#SKIP_EVERY_NTH_1=100 # best at 2
#SAMPLING_PERCENT_1=0.5 # default 0.25

# heatmap
#SKIP_EVERY_NTH_2=10 # best at 2
#SAMPLING_PERCENT_2=0.5 # default 0.5

startdate = date(2020, 3, 1)
enddate = date(2021, 8, 31)

default_zipcode ="33510"
default_year ="2021"
default_sampling=0.25
default_zoom=9

year_for_all="2021"
zipcode_for_all="33510"
sampling_for_all=0.25 # 0.5=50%
show_whole_county=False

heatmap_size=0
scatter_size=0

graph_width=900
graph_height=750


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

red_fillcolor='rgba(245,153,204,1)'
green_fillcolor='rgba(0,205,0,0.5)'
gray_fillcolor='rgba(120,120,120,0.5)'

fillcolor1='rgb(184, 247, 212)'
fillcolor2='rgb(111, 231, 219)'
fillcolor3='rgb(127, 166, 238)'
fillcolor4='rgb(131, 90, 241)'
fillcolor5='rgb(141, 80, 251)'
'''
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
    fig.update_layout(
                    # showlegend=True, 
                    autosize=True, 
                    # width=800, height=900,
                      legend=dict(orientation="h",x=0, y=-0.12, traceorder="normal"),
                      font=dict(family="Arial", size=12))
    #fig.show()
    return fig


def plot_age(df):
    sub_groups = ['Cases >65', 'Admissions >65', 'Deaths >65', 'Cases >18', 'Admissions >18', 'Deaths >18',
                  'Cases >1', 'Admissions >1', 'Deaths >1']
    fig = make_subplots(rows=3, cols=3, subplot_titles=sub_groups, shared_xaxes=True)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_65']/SF1,
                             name="cases >65", 
                             line=dict({'width': 2, 'color': red_fillcolor})), 
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), 
                row=1, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_65']/SF1,
                             name="admissions >65", line=dict({'width': 2, 'color': green_fillcolor})), 
                row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_65']/SF2,
                             name="deaths >65", line=dict({'width': 2, 'color':gray_fillcolor })), 
                row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})),
                row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_18']/SF1,
                             name="cases >18", 
                             line=dict({'width': 2, 'color': red_fillcolor})), 
                row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), 
                row=2,col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_18']/SF1,
                             name="admissions >18", 
                             line=dict({'width': 2, 'color': green_fillcolor})), 
                row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                             line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_18']/SF2,
                             name="deaths >18", 
                             line=dict({'width': 2, 'color': gray_fillcolor})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})),
                row=2, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_1']/SF1,
                             name="cases >1", 
                             line=dict({'width': 2, 'color': red_fillcolor})), 
                row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), 
                row=3,col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_1']/SF1,
                             name="admissions >1", 
                             line=dict({'width': 2, 'color': green_fillcolor})), 
                row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                             line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_1']/SF2,
                             name="deaths >1", 
                             line=dict({'width': 2, 'color': gray_fillcolor})), 
                row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths",
                             line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})),
                row=3,col=3)
    # -------------------------------------------------------------------------------------------------------------------

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(
        # showlegend=True,
        autosize=True, 
        # width=950, height=1100,
                      legend=dict(orientation="h", x=0, y=-0.12, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyAge-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    #fig.show()
    return fig


def plot_gender(df):
    sub_groups = ['Cases Male', 'Admissions Male', 'Deaths Male',
                  'Cases Female', 'Admissions Female', 'Deaths Female']
    fig = make_subplots(rows=3, cols=3, subplot_titles=sub_groups, shared_xaxes=True)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_male']/SF1,
                             name="cases Male", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=1,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_male']/SF1,
                             name="admissions Male", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_male']/SF2,
                             name="deaths Male", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})),
                  row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_female']/SF1,
                             name="cases Female", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_female']/SF1,
                             name="admissions Female", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_female']/SF2,
                             name="deaths Female", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})),
                  row=2,
                  col=3)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(
        # showlegend=True, 
        autosize=True, 
        # width=950, height=1000,
                      legend=dict(orientation="h", x=0, y=-0.1, traceorder="normal"),
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
    fig = make_subplots(rows=5, cols=3, subplot_titles=sub_groups, shared_xaxes=True)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_white']/SF1,
                             name="Cases (white)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=1,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_white']/SF1,
                             name="admissions (white)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_white']/SF2,
                             name="deaths (white)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=1,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_black']/SF1,
                             name="cases (black)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_black']/SF1,
                             name="admissions (black)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_black']/SF2,
                             name="deaths (black)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=2,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_asian']/SF1,
                             name="cases (asian)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=3,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_asian']/SF1,
                             name="admissions (asian)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_asian']/SF2,
                             name="deaths (asian)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=3,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_other']/SF1,
                             name="cases (other)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=4,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_other']/SF1,
                             name="admissions (other)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=4, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=4, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_other']/SF2,
                             name="deaths (other)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=4, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=4,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_two']/SF1,
                             name="cases (two)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=5,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_two']/SF1,
                             name="admissions (two)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=5, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=5, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_two']/SF2,
                             name="deaths (two)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=5, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=5,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True) #autorange=True
    fig.update_layout(
        # showlegend=True, 
        autosize=True, 
                    # width=950, height=1700,
                      legend=dict(orientation="h", x=0, y=-0.12, traceorder="normal"),
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
    fig = make_subplots(rows=5, cols=3, subplot_titles=sub_groups, shared_xaxes=True)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_0-100']/SF1,
                             name="Cases (0-100)", line=dict({'width':2, 'color':'rgba(255,153,204,0.5)'})), row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash':'dot'})), row=1, col=1)
    #-------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_0-100']/SF1,
                             name="admissions (0-100)", line=dict({'width':2, 'color':'rgba(217, 255, 217,0.75)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})), row=1, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_0-100']/SF2,
                             name="deaths (0-100)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=1, col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_100-150']/SF1,
                             name="cases (100-150)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=2,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_100-150']/SF1,
                             name="admissions (100-150)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=2, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_100-150']/SF2,
                             name="deaths (100-150)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=2,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_150-175']/SF1,
                             name="cases (150-175)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=3,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_150-175']/SF1,
                             name="admissions (150-175)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=3, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_150-175']/SF2,
                             name="deaths (150-175)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=3,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_175-200']/SF1,
                             name="cases (175-200)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=4,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_175-200']/SF1,
                             name="admissions (175-200)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=4, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=4, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_175-200']/SF2,
                             name="deaths (175-200)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=4, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=4,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_200-1800']/SF1,
                             name="cases (200-1800)", line=dict({'width': 2, 'color': 'rgba(255,153,204,0.5)'})), row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", line=dict({'width': 1, 'color': 'red', 'dash': 'dot'})), row=5,
                  col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_200-1800']/SF1,
                             name="admissions (200-1800)", line=dict({'width': 2, 'color': 'rgba(217, 255, 217,0.75)'})), row=5, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green', 'dash': 'dot'})),
                  row=5, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_200-1800']/SF2,
                             name="deaths (200-1800)", line=dict({'width': 2, 'color': 'rgba(230, 230, 230,0.75)'})), row=5, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), row=5,
                  col=3)
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    # fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, autorange=True)
    fig.update_layout(
                    # showlegend=True, 
                    # autosize=True, 
                    #   width=950, height=1700,
                      legend=dict(orientation="h",x=0, y=-0.12, traceorder="normal"),
                      font=dict(family="Arial", size=10))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyFPL-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    #fig.show()
    return fig
 
def get_RMSE(zip, actual, simulated):
    mse = mean_squared_error(actual, simulated)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, simulated)
    return rmse
'''
def calc_7_day_average(df):
    return df.rolling(window=7).mean()

def plot2(df):
    # Create figure
    fig = go.Figure()

    sub_groups = ['Simulated cases', 'Actual cases', 'Simulated admissions', 'Actual admissions', 'Simulated deaths', 'Actual deaths']
    fig = make_subplots(rows=6, cols=1, 
        subplot_titles=sub_groups, 
        shared_xaxes=True, 
        vertical_spacing = 0.05,
        row_width=[0.1, 0.25,0.1, 0.25,0.1, 0.25])
    
    df['cases'] =calc_7_day_average(df['cases'])
    df['admissions'] = calc_7_day_average(df['admissions'])
    df['deaths']= calc_7_day_average(df['deaths'])

    fig.add_trace(go.Scatter( fill='tozeroy', x=df['date'], y=df['cases'],
                             name="cases", line=dict({'width': 2, 'color': 'red'})),
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="vcases", line=dict({'width': 1, 'color': 'red'})),
                row=2, col=1)

    fig.add_trace(go.Scatter( fill='tozeroy', x=df['date'], y=df['admissions'],
                             name="admissions", line=dict({'width': 2, 'color': 'green'})),
                row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", line=dict({'width': 1, 'color': 'green'})), 
                row=4, col=1)

    fig.add_trace(go.Scatter( fill='tozeroy', x=df['date'], y=df['deaths'],
                             name="deaths", line=dict({'width': 2, 'color': 'black'})), 
                row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'gray'})), 
                row=6, col=1)
    # df['vdeaths2']=df['vdeaths'].rolling(window=7).mean()
    # fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths2'],
    #                          name="actual deaths-7dayAverage", line=dict({'width': 1, 'color': 'black', 'dash': 'dot'})), 
    #             row=6, col=1)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, autosize=True, width=800, height=900,
                      legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
                      font=dict(family="Arial", size=12))
    #fig.show()
    return fig
    return fig

def plot_age2(df):
    sub_groups = ['Simulated cases', 'Actual cases', 'Simulated admissions', 'Actual admissions', 'Simulated deaths', 'Actual deaths']
    fig = make_subplots(rows=6, cols=1, 
        subplot_titles=sub_groups, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_width=[0.1, 0.25,0.1, 0.25,0.1, 0.25])

    df['cases_65']=calc_7_day_average(df['cases_65'])
    df['cases_18']=calc_7_day_average(df['cases_65'])
    df['cases_1']=calc_7_day_average(df['cases_1'])
    df['admissions_65']=calc_7_day_average(df['admissions_65'])
    df['admissions_18']=calc_7_day_average(df['admissions_18'])
    df['admissions_1']=calc_7_day_average(df['admissions_1'])
    df['deaths_65']=calc_7_day_average(df['deaths_65'])
    df['deaths_18']=calc_7_day_average(df['deaths_18'])
    df['deaths_1']=calc_7_day_average(df['deaths_1'])

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_65'], stackgroup='one', groupnorm='percent',
                             name="cases >65", 
                             line=dict(width=0.5, color='rgb(131, 90, 241)')), 
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_18'], stackgroup='one', groupnorm='percent',
                             name="cases >18", 
                             line=dict(width=0.5, color='rgb(111, 231, 219)')), 
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_1'], stackgroup='one', groupnorm='percent',
                             name="cases >1", 
                             line=dict(width=0.5, color='rgb(184, 247, 212)')), 
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'red'})), 
                row=2, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_65'], stackgroup='two', groupnorm='percent',
                             name="admissions >65", 
                             line=dict(width=0.5, color='rgb(131, 90, 241)')), 
                row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_18'], stackgroup='two', groupnorm='percent',
                             name="admissions >18", 
                             line=dict(width=0.5, color='rgb(111, 231, 219)')), 
                row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_1'], stackgroup='two', groupnorm='percent',
                             name="admissions >1", 
                             line=dict(width=0.5, color='rgb(184, 247, 212)')), 
                row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                             line=dict({'width': 1, 'color': 'green'})),
                row=4, col=1)

    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_65'], stackgroup='three', groupnorm='percent',
                             name="deaths >65", 
                             line=dict(width=0.5, color='rgb(131, 90, 241)')), 
                row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_18'], stackgroup='three', groupnorm='percent',
                             name="deaths >18", 
                             line=dict(width=0.5, color='rgb(111, 231, 219)')), 
                row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_1'], stackgroup='three', groupnorm='percent',
                             name="deaths >1", 
                             line=dict(width=0.5, color='rgb(184, 247, 212)')), 
                row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", line=dict({'width': 1, 'color': 'black'})),
                row=6, col=1)
    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    return fig

def plot_gender2(df):
    sub_groups = ['Simulated cases', 'Actual cases', 'Simulated admissions', 'Actual admissions', 'Simulated deaths', 'Actual deaths']
    fig = make_subplots(rows=6, cols=1, 
    subplot_titles=sub_groups, 
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_width=[0.1, 0.25,0.1, 0.25,0.1, 0.25])

    df['cases_male']=calc_7_day_average(df['cases_male'])
    df['cases_female']=calc_7_day_average(df['cases_female'])
    df['admissions_male']=calc_7_day_average(df['admissions_male'])
    df['admissions_female']=calc_7_day_average(df['admissions_female'])
    df['deaths_male']=calc_7_day_average(df['deaths_male'])
    df['deaths_female']=calc_7_day_average(df['deaths_female'])

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_male'], stackgroup='one', groupnorm='percent',
                             name="cases Male", 
                             line=dict(width=0.5, color='rgb(131, 90, 241)')), 
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_female'], stackgroup='one', groupnorm='percent',
                             name="cases Female", 
                             line=dict(width=0.5, color='rgb(111, 231, 219)')),
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'red'})), 
                row=2,col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_male'], stackgroup='two', groupnorm='percent',
                             name="admissions Male", 
                             line=dict(width=0.5, color='rgb(131, 90, 241)')), 
                    row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_female'], stackgroup='two', groupnorm='percent',
                             name="admissions Female", 
                             line=dict(width=0.5, color='rgb(111, 231, 219)')), 
                    row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                            line=dict({'width': 1, 'color': 'green'})),
                  row=4, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_male'], stackgroup='three', groupnorm='percent',
                             name="deaths Male", 
                             line=dict(width=0.5, color='rgb(131, 90, 241)')), 
                    row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_female'], stackgroup='three', groupnorm='percent',
                             name="deaths Female", 
                             line=dict(width=0.5, color='rgb(111, 231, 219)')), 
                    row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", 
                            line=dict({'width': 1, 'color': 'black'})),
                  row=6, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    return fig

def plot_race2(df):
    sub_groups = ['Simulated cases', 'Actual cases', 'Simulated admissions', 'Actual admissions', 'Simulated deaths', 'Actual deaths']
    fig = make_subplots(rows=6, cols=1, 
    subplot_titles=sub_groups, 
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_width=[0.1, 0.25,0.1, 0.25,0.1, 0.25])

    df['cases_white']=calc_7_day_average(df['cases_white'])
    df['cases_black']=calc_7_day_average(df['cases_black'])
    df['cases_asian']=calc_7_day_average(df['cases_asian'])
    df['cases_other']=calc_7_day_average(df['cases_other'])
    df['cases_two']=calc_7_day_average(df['cases_two'])
    df['admissions_white']=calc_7_day_average(df['admissions_white'])
    df['admissions_black']=calc_7_day_average(df['admissions_black'])
    df['admissions_asian']=calc_7_day_average(df['admissions_asian'])
    df['admissions_other']=calc_7_day_average(df['admissions_other'])
    df['admissions_two']=calc_7_day_average(df['admissions_two'])
    df['deaths_black']=calc_7_day_average(df['deaths_black'])
    df['deaths_asian']=calc_7_day_average(df['deaths_asian'])
    df['deaths_other']=calc_7_day_average(df['deaths_other'])
    df['deaths_two']=calc_7_day_average(df['deaths_two'])

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_white'], stackgroup='one', groupnorm='percent',
                             name="Cases (white)", 
                             line=dict({'width': 0.5, 'color': fillcolor1})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_black'], stackgroup='one', groupnorm='percent',
                             name="cases (black)", 
                             line=dict({'width': 0.5, 'color': fillcolor2})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_asian'], stackgroup='one', groupnorm='percent',
                             name="cases (asian)", 
                             line=dict({'width': 0.5, 'color': fillcolor3})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_other'], stackgroup='one', groupnorm='percent',
                             name="cases (other)", 
                             line=dict({'width': 0.5, 'color': fillcolor4})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_two'], stackgroup='one', groupnorm='percent',
                             name="cases (two)", 
                             line=dict({'width': 0.5, 'color': fillcolor5})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'red'})), 
                             row=2, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_white'], stackgroup='two', groupnorm='percent',
                             name="admissions (white)", 
                             line=dict({'width': 0.5, 'color': fillcolor1})), 
                        row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_black'], stackgroup='two', groupnorm='percent',
                             name="admissions (black)", 
                             line=dict({'width': 0.5, 'color': fillcolor2})), 
                        row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_asian'], stackgroup='two', groupnorm='percent',
                             name="admissions (asian)", 
                             line=dict({'width': 0.5, 'color': fillcolor3})), 
                        row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_other'], stackgroup='two', groupnorm='percent',
                             name="admissions (other)", 
                             line=dict({'width': 0.5, 'color': fillcolor4})), 
                        row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_two'], stackgroup='two', groupnorm='percent',
                             name="admissions (two)", 
                             line=dict({'width': 0.5, 'color': fillcolor5})), 
                         row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                             line=dict({'width': 1, 'color': 'green'})),
                        row=4, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_white'], stackgroup='three', groupnorm='percent',
                             name="deaths (white)", 
                             line=dict({'width': 0.5, 'color': fillcolor1})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_black'], stackgroup='three', groupnorm='percent',
                             name="deaths (black)", 
                             line=dict({'width': 0.5, 'color': fillcolor2})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_asian'], stackgroup='three', groupnorm='percent',
                             name="deaths (asian)", 
                             line=dict({'width': 0.5, 'color': fillcolor3})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_other'], stackgroup='three', groupnorm='percent',
                             name="deaths (other)", 
                             line=dict({'width': 0.5, 'color': fillcolor4})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_two'], stackgroup='three', groupnorm='percent',
                             name="deaths (two)", 
                             line=dict({'width': 0.5, 'color': fillcolor5})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", 
                             line=dict({'width': 1, 'color': 'black'})), 
                             row=6,col=1)
    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    return fig

def plot_FPL2(df):
    sub_groups = ['Simulated cases', 'Actual cases', 'Simulated admissions', 'Actual admissions', 'Simulated deaths', 'Actual deaths']
    fig = make_subplots(rows=6, cols=1, 
        subplot_titles=sub_groups, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_width=[0.1, 0.25,0.1, 0.25,0.1, 0.25])

    df['cases_0-100']=calc_7_day_average(df['cases_0-100'])
    df['cases_100-150']=calc_7_day_average(df['cases_100-150'])
    df['cases_150-175']=calc_7_day_average(df['cases_150-175'])
    df['cases_175-200']=calc_7_day_average(df['cases_175-200'])
    df['cases_200-1800']=calc_7_day_average(df['cases_200-1800'])
    df['admissions_0-100']=calc_7_day_average(df['admissions_0-100'])
    df['admissions_100-150']=calc_7_day_average(df['admissions_100-150'])
    df['admissions_150-175']=calc_7_day_average(df['admissions_150-175'])
    df['admissions_175-200']=calc_7_day_average(df['admissions_175-200'])
    df['admissions_200-1800']=calc_7_day_average(df['admissions_200-1800'])
    df['deaths_0-100']=calc_7_day_average(df['deaths_0-100'])
    df['deaths_100-150']=calc_7_day_average(df['deaths_100-150'])
    df['deaths_150-175']=calc_7_day_average(df['deaths_150-175'])
    df['deaths_175-200']=calc_7_day_average(df['deaths_175-200'])
    df['deaths_200-1800']=calc_7_day_average(df['deaths_200-1800'])

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_0-100'], stackgroup='one', groupnorm='percent',
                             name="Cases (0-100)", 
                             line=dict({'width':0.5, 'color':fillcolor1})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_100-150'], stackgroup='one', groupnorm='percent',
                             name="cases (100-150)", 
                             line=dict({'width': 0.5, 'color':fillcolor2})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_150-175'], stackgroup='one', groupnorm='percent',
                             name="cases (150-175)", 
                             line=dict({'width': 0.5, 'color':fillcolor3})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_175-200'], stackgroup='one', groupnorm='percent',
                             name="cases (175-200)", 
                             line=dict({'width': 0.5, 'color':fillcolor4})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_200-1800'], stackgroup='one', groupnorm='percent',
                             name="cases (200-1800)", 
                             line=dict({'width': 0.5, 'color':fillcolor5})), 
                             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'red'})), 
                             row=2, col=1)
    #-------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_0-100'], stackgroup='two', groupnorm='percent',
                             name="admissions (0-100)", 
                             line=dict({'width':0.5, 'color':fillcolor1})), 
                             row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_100-150'], stackgroup='two', groupnorm='percent',
                             name="admissions (100-150)", 
                             line=dict({'width': 0.5, 'color':fillcolor2})), 
                             row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_150-175'], stackgroup='two', groupnorm='percent',
                             name="admissions (150-175)", 
                             line=dict({'width': 0.5, 'color':fillcolor3})), 
                             row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_175-200'], stackgroup='two', groupnorm='percent',
                             name="admissions (175-200)", 
                             line=dict({'width': 0.5, 'color':fillcolor4})), 
                             row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_200-1800'], stackgroup='two', groupnorm='percent',
                             name="admissions (200-1800)", 
                             line=dict({'width': 0.5, 'color':fillcolor5})), 
                             row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                             line=dict({'width': 1, 'color': 'green'})), 
                             row=4, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_0-100'], stackgroup='three', groupnorm='percent',
                             name="deaths (0-100)", 
                             line=dict({'width': 0.5, 'color':fillcolor1})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_100-150'], stackgroup='three', groupnorm='percent',
                             name="deaths (100-150)", 
                             line=dict({'width': 0.5, 'color':fillcolor2})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_150-175'], stackgroup='three', groupnorm='percent',
                             name="deaths (150-175)", 
                             line=dict({'width': 0.5, 'color':fillcolor3})),
                              row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_175-200'], stackgroup='three', groupnorm='percent',
                             name="deaths (175-200)", 
                             line=dict({'width': 0.5, 'color':fillcolor4})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_200-1800'], stackgroup='three', groupnorm='percent',
                             name="deaths (200-1800)", 
                             line=dict({'width': 0.5, 'color':fillcolor5})), 
                             row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", 
                             line=dict({'width': 1, 'color': 'black'})), 
                             row=6, col=1)
    # -------------------------------------------------------------------------------------------------------------------

    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    return fig

def load_SEIR(mode):
    if mode=='All cases':
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
        #####plotdf.drop('date', axis=1, inplace=True)

        # max = plotdf.groupby(plotdf.columns, axis=1).max()
        # mean = plotdf.groupby(plotdf.columns, axis=1).mean()
        # min = plotdf.groupby(plotdf.columns, axis=1).min()
        # max['date'] = dates
        # mean['date'] = dates
        # min['date'] = dates

        plotdf=plotdf.groupby(plotdf.columns, axis=1).sum()
        plotdf['date']= dlist[0]['date'].tolist()
        df2 =pd.DataFrame(plotdf, columns=['date', 'cases', 'deaths','admissions', 'vcases', 'vdeaths', 'vadmissions'])
        # temporary solutions for empty vcases/vdeaths/vadmissions
        if set(['vcases','vdeaths', 'vadmissions']).issubset(df2.columns) is False:
            df2['vcases']=df2['cases']
            df2['vdeaths']=df2['deaths']
            df2['vadmissions']=df2['admissions']
            print("Warning: vcases/vadmissions/vdeaths not found. Use cases/admissions/deathss")
        #print('Loading completed!')
        
        #return plot(min, mean, max)
        return plot2(df2)

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
        # temporary solutions for empty vcases/vdeaths/vadmissions
        if set(['vcases','vdeaths', 'vadmissions']).issubset(plotdf.columns)==False:
            plotdf['vcases']=plotdf['cases']
            plotdf['vdeaths']=plotdf['deaths']
            plotdf['vadmissions']=plotdf['admissions']
            print("Warning: vcases/vadmissions/vdeaths not found. Use cases/admissions/deathss")
        if mode == 'By Age':
            #return plot_age(plotdf)
            return plot_age2(plotdf)
        elif mode == 'By Gender':
            #return plot_gender(plotdf)
            return plot_gender2(plotdf)
        elif mode == 'By Race':
            #return plot_race(plotdf)
            return plot_race2(plotdf)
        elif mode == 'By FPL':
            #return plot_FPL(plotdf)
            return plot_FPL2(plotdf)

def load_scatter(zipcode, year, sampling, width, height, show_whole_county):
    #return load_scatter_mongodb(zipcode, year)
    return load_scatter_read_parquet(zipcode, year, sampling, width, height, show_whole_county)

def load_heatmap(zipcode,year, sampling, width, height, show_whole_county):
    #return load_heatmap_mongodb(zipcode, year)
    return load_heatmap_read_parquet(zipcode,year, sampling, width, height, show_whole_county)

"""
load_scatter using_read_parquet
"""
def load_scatter_read_parquet(zipcode=default_zipcode, year=default_year, sampling=default_sampling, width=graph_width, height=graph_height, show_whole_county=False):
    # global year_for_all
    # global zipcode_for_all
    # global sampling_for_all
    # zipcode=zipcode_for_all
    # year=year_for_all
    sampling_rate=sampling

    total_steps = enddate - startdate
    step_until_lastday_of_2020=date(2020, 12, 31)-startdate
    min_step=0
    max_step=total_steps.days
    if year=="2020":
        min_step=0
        max_step=step_until_lastday_of_2020.days
    elif year=="2021":
        min_step=step_until_lastday_of_2020.days + 1
        max_step=total_steps.days
 
    # if zipcode is None:
    #     zipcode=default_zipcode
    # print("zipcode:", zipcode)

    print(zipcode, year, min_step, max_step,"show_whole_county=", show_whole_county)

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
    #tp = pd.read_csv(os.path.join(path, 'scatter.csv'), iterator=True, chunksize=CHUNK_SIZE1, skiprows=lambda x: x % SKIP_EVERY_NTH_1)
    #pdf = pd.concat(tp, ignore_index=True)

    #pdf = dd.read_csv(os.path.join(path, 'scatter.csv'))

    #pdf=pdf.drop(['pid', 'location', 'ZIP', 'type'], axis=1)
    t1=datetime.now()
    #pdf = pd.read_parquet(os.path.join(path, 'scatter.parquet'),
    #pdf = pd.read_parquet(os.path.join(path, 'scatter.parquet.gzip'),

    filters_settting=[('step','>=', min_step), ('step', '<=', max_step)]
    
    if show_whole_county is False:
        filters_settting.append(('ZIP','in', [zipcode]))  #ZIP in scatter, zip in heatmap
    # else:
    #     sampling_rate = sampling_rate/10.0

    pdf = pd.read_parquet(os.path.join(path, 'scatter.snappy.parquet'),
                    filters=filters_settting,
                    columns=columns_being_used_in_scatter
    )

    print("scatter read_parquet time spent", datetime.now()-t1)
    print("sampling rate", sampling_rate)
    pdf = pdf.sample(frac=sampling_rate)

    pdf["state"] =  pdf["state"].astype("category")
    #pdf["step"] =  pdf["step"].astype("category")
    pdf["step"] = pd.to_numeric(pdf["step"], downcast="unsigned")
    pdf[['x','y']] = pdf[['x','y']].apply(pd.to_numeric, downcast="float")

    #datelist = pd.date_range(startdate, enddate).tolist()
    pdf['Date']=[startdate+timedelta(days=d) for d in pdf['step']]
    pdf['Date']=pdf['Date'].astype(str)
    #pdf['Date']=pdf['Date'].astype('category')
    # print('scatter data size(before '+ str(SAMPLING_PERCENT_1) +' sampling)', pdf.size)
    # pdf = pdf.sample(frac=SAMPLING_PERCENT_1) # (???) similar to geting every 4th rows
    # print('scatter data size(after '+ str(SAMPLING_PERCENT_1) +' sampling)', pdf.size)
    #print("unique states", pd.unique(pdf["state"]))

    #print('scatter memory usage', pdf.info())
    print('scatter memory size(MB)', sys.getsizeof(pdf)/(1024*1024))
    pdf = pdf.sort_values(by='step') # should be run after sampling
    pdf.drop('step',axis=1, inplace=True)

    return draw_scatter(pdf, zipcode, width, height, show_whole_county)

"""
load_heatmap using_read_parquet
"""
def load_heatmap_read_parquet(zipcode=default_zipcode, year=default_year, sampling=default_sampling, width=graph_width, height=graph_height, show_whole_county=False):
    # global year_for_all
    # global zipcode_for_all
    # global sampling_for_all

    # zipcode=zipcode_for_all
    # year=year_for_all
    sampling_rate=sampling

    total_steps = enddate - startdate
    step_until_lastday_of_2020=date(2020, 12, 31)-startdate
    min_step=0
    max_step=total_steps.days
    if year=="2020":
        min_step=0
        max_step=step_until_lastday_of_2020.days
    elif year=="2021":
        min_step=step_until_lastday_of_2020.days + 1
        max_step=total_steps.days
 
    # if zipcode is None:
    #     zipcode=default_zipcode
    # print("zipcode:", zipcode)
    print(zipcode, year, min_step, max_step, "show_whole_county=", show_whole_county)
    
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
    #pdf = pd.read_parquet(os.path.join(path, 'heatmap.parquet'),
    #pdf = pd.read_parquet(os.path.join(path, 'heatmap.parquet.gzip'),

    filters_settting=[('step','>=', min_step), ('step', '<=', max_step)]
    if show_whole_county is False:
        filters_settting.append(('zip','in', [zipcode])) #ZIP in scatter, zip in heatmap
    # else:
    #     sampling_rate = sampling_rate/10.0

    pdf = pd.read_parquet(os.path.join(path, 'heatmap.snappy.parquet'),
                    filters=filters_settting,
                    columns=columns_being_used_in_heatmap,
    )
    print("heatmap read_parquet time spent", datetime.now()-t1)
    print("sampling rate", sampling_rate)
    pdf = pdf.sample(frac=sampling_rate)
    #pdf["step"] =  pdf["step"].astype("category")
    pdf["step"] = pd.to_numeric(pdf["step"], downcast="unsigned")
    pdf[['x','y']] = pdf[['x','y']].apply(pd.to_numeric, downcast="float")
    pdf["z"] = pd.to_numeric(pdf["z"], downcast="unsigned")

    #datelist = pd.date_range(startdate, enddate).tolist()
    pdf['Date']=[startdate+timedelta(days=d) for d in pdf['step']]
    pdf['Date']=pdf['Date'].astype(str)
    #pdf['Date']=pdf['Date'].astype('category')
    #heatmap=db.heatmap
    #list_heatmap=list(heatmap.find({}))
    #list_heatmap=list(heatmap.find({}))[::10]
    #print(len(list_heatmap))
    #pdf = pd.DataFrame(list_heatmap)

    # print('heatmap data size(before '+ str(SAMPLING_PERCENT_2) +' sampling)', pdf.size)
    # pdf = pdf.sample(frac=SAMPLING_PERCENT_2)
    # print('heatmap data size(after '+ str(SAMPLING_PERCENT_2) +' sampling)', pdf.size)

    #print('heatmap memory size', pdf.info())
    print('heatmap memory usage(MB)', sys.getsizeof(pdf)/(1024*1024))
    pdf = pdf.sort_values(by='step')
    pdf.drop('step',axis=1, inplace=True)

    return draw_heatmap(pdf, zipcode, width, height, show_whole_county)

"""
load_scatter via_mongodb
"""
# def load_scatter_mongodb(zipcode=default_zipcode, year=default_year, width=graph_width, height=graph_height):
#     total_steps = enddate - startdate
#     step_until_lastday_of_2020=date(2020, 12, 31)-startdate
#     min_step=0
#     max_step=total_steps.days
#     if year=="2020":
#         min_step=0
#         max_step=step_until_lastday_of_2020.days
#     elif year=="2021":
#         min_step=step_until_lastday_of_2020.days + 1
#         max_step=total_steps.days
 
#     if zipcode is None:
#         zipcode=default_zipcode
#     print("zipcode:", zipcode)
#     print(year, min_step, max_step)

#     #tp = vaex.from_csv(os.path.join(path, 'scatter.csv'), copy_index=False)

#     #tp = pd.read_csv(os.path.join(path, 'scatter.csv'), iterator=True, chunksize=CHUNK_SIZE1, skiprows=lambda x: x % SKIP_EVERY_NTH_1)
#     #pdf = pd.concat(tp, ignore_index=True)

#     #pdf = dd.read_csv(os.path.join(path, 'scatter.csv'))

#     #pdf=pdf.drop(['pid', 'location', 'ZIP', 'type'], axis=1)

#     t1=datetime.now()
#     list_scatter=list(db.scatter.find({'$and':[
#                                     {"ZIP":int(zipcode)}, 
#                                     {'step':{'$gt':min_step}},
#                                     {'step':{'$lt':max_step}}
#     ]}
#     ))
#     print("mongodb search result for ", zipcode, len(list_scatter),"time spent", datetime.now()-t1)
#     pdf = pd.DataFrame(list_scatter)

#     pdf["state"] =  pdf["state"].astype("category")
#     #pdf["step"] =  pdf["step"].astype("category")
#     pdf["step"] = pd.to_numeric(pdf["step"], downcast="unsigned")
#     pdf[['x','y']] = pdf[['x','y']].apply(pd.to_numeric, downcast="float")

#     #datelist = pd.date_range(startdate, enddate).tolist()
#     pdf['Date']=[startdate+timedelta(days=d) for d in pdf['step']]
#     pdf['Date']=pdf['Date'].astype(str)
    

#     # print('scatter data size(before '+ str(SAMPLING_PERCENT_1) +' sampling)', pdf.size)
#     # pdf = pdf.sample(frac=SAMPLING_PERCENT_1) # (???) similar to geting every 4th rows
#     # print('scatter data size(after '+ str(SAMPLING_PERCENT_1) +' sampling)', pdf.size)

#     #print('scatter memory usage', pdf.info())
#     print('scatter memory size(MB)', sys.getsizeof(pdf)/(1024*1024))
#     pdf = pdf.sort_values(by='step') # should be run after sampling
#     pdf.drop('step',axis=1, inplace=True)

#     return draw_scatter(pdf, zipcode, width, height)

"""
load_heatmap using_mongodb
"""
# def load_heatmap_mongodb(zipcode=default_zipcode, year=default_year, width=graph_width, height=graph_height):
#     total_steps = enddate - startdate
#     step_until_lastday_of_2020=date(2020, 12, 31)-startdate
#     min_step=0
#     max_step=total_steps.days
#     if year=="2020":
#         min_step=0
#         max_step=step_until_lastday_of_2020.days
#     elif year=="2021":
#         min_step=step_until_lastday_of_2020.days + 1
#         max_step=total_steps.days
 
#     if zipcode is None:
#         zipcode=default_zipcode
#     print("zipcode:", zipcode)
#     print(year, min_step, max_step)

#     t1=datetime.now()
#     #list_heatmap=list(db.heatmap.find({"zip":int(zipcode)}))
#     list_heatmap=list(db.heatmap.find({'$and':[
#                                     {"zip":int(zipcode)}, 
#                                     {'step':{'$gt':min_step}},
#                                     {'step':{'$lt':max_step}}
#     ]}
#     ))
#     print("heatmap mongodbsearch result", len(list_heatmap),"time spent", datetime.now()-t1)
#     pdf = pd.DataFrame(list_heatmap)

#     # pdf["step"] =  pdf["step"].astype("category")
#     pdf["step"] = pd.to_numeric(pdf["step"], downcast="unsigned")
#     pdf[['x','y']] = pdf[['x','y']].apply(pd.to_numeric, downcast="float")
#     pdf["z"] = pd.to_numeric(pdf["z"], downcast="unsigned")

#     #datelist = pd.date_range(startdate, enddate).tolist()
#     pdf['Date']=[startdate+timedelta(days=d) for d in pdf['step']]
#     pdf['Date']=pdf['Date'].astype(str)

#     # print('heatmap data size(before '+ str(SAMPLING_PERCENT_2) +' sampling)', pdf.size)
#     # pdf = pdf.sample(frac=SAMPLING_PERCENT_2)
#     # print('heatmap data size(after '+ str(SAMPLING_PERCENT_2) +' sampling)', pdf.size)
    
#     #print('heatmap memory size', pdf.info())
#     print('heatmap memory usage(MB)', sys.getsizeof(pdf)/(1024*1024))
#     pdf = pdf.sort_values(by='step')
#     pdf.drop('step',axis=1, inplace=True)
    
#     return draw_heatmap(pdf, zipcode, width, height)

def draw_scatter(pdf, zipcode, width, height, show_whole_county):
    global scatter_size
    scatter_size=pdf.size
    print("scatter_size in draw", scatter_size)

    center_lat = 28.03711
    center_lon = -82.46390
    zoom_level=default_zoom
    if ZIPS_centers[zipcode]:
        center_lat=ZIPS_centers[zipcode][0]
        center_lon=ZIPS_centers[zipcode][1]
        zoom_level=12
    if show_whole_county:
        zoom_level=8
    print("centers", center_lat, center_lon, "zoom", zoom_level)
    fig = px.scatter_mapbox(pdf,
                            #title="Scatter_Map",
                            #color='color',
                            #animation_frame='step',
                            animation_frame='Date',
                            #animation_group='date',
                            #text='state',
                            color='state',
                            color_discrete_map=legend_map,
                            lat='y',
                            lon='x',                            
                            zoom=zoom_level, #default 8 (0-20)
                            width=width,
                            height=height,
                            center=dict(lat=center_lat, lon=center_lon),
                            #mapbox_style='open-street-map',
                            #mapbox_style='carto-darkmatter',
                            mapbox_style='carto-positron',
    )
    fig.update_layout(showlegend=False)
    #fig.update_traces(marker=dict(size=10))
    #fig.update_traces(marker=dict(size=6))
    fig.update_layout(legend=dict(
        orientation="h",
        # xanchor="left",
        # yanchor="bottom",
        # x=0,
        # y=-0.1,
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,        
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


def draw_heatmap(pdf, zipcode, width, height, show_whole_county):
    global heatmap_size
    heatmap_size=pdf.size   
    print("heatmap size in draw=", heatmap_size)
    center_lat = 28.03711
    center_lon = -82.46390
    zoom_level=9
    if ZIPS_centers[zipcode]:
        center_lat=ZIPS_centers[zipcode][0]
        center_lon=ZIPS_centers[zipcode][1]
        zoom_level=12
    if show_whole_county:
        zoom_level=8
    print("centers", center_lat, center_lon, "zoom", zoom_level)
    fig = px.density_mapbox(pdf,
                            color_continuous_scale='RdYlGn_r',
                            lat=pdf['y'],
                            lon=pdf['x'],
                            z=pdf['z'],
                            #animation_frame=pdf['step'],
                            animation_frame='Date',
                            opacity=0.75,
                            zoom=zoom_level, #default 8 (0-20)
                            width=width,
                            height=height,
                            center=dict(lat=center_lat, lon=center_lon),
                            # mapbox_style='open-street-map'
                            mapbox_style='stamen-terrain')
    return fig

def draw_legend_table():
    return html.Table(className='table', children = [
                html.Tr( [html.Td("☻ susceptible", style={"color":"blue"}), 
                    html.Td("☻ asymptomatic", style={"color":"purple"}),
                    html.Td("☻ vaccinated", style={"color":"olive"}), 
                    html.Td("☻ boosted", style={"color":"olive"}), 
                    html.Td("☻ recovered", style={"color":"green"}),
                    html.Td("")]),
                html.Tr( [html.Td("☻ critical", style={"color":"#F1948A"}), 
                    html.Td("☻ dead", style={"color":"black"}), 
                    html.Td("☻ exposed", style={"color":"orange"}), 
                    html.Td("☻ mild", style={"color":"#F5B7B1"}), 
                    html.Td("☻ presymptomatic", style={"color":"#F2D7D5"}), 
                    html.Td("☻ severe", style={"color":"#EC7063"})]),
            ], style={"border-style": "ridge", "text-align": "left", 'marginLeft': 'auto', 'marginRight': 'auto'})

"""
main plotly dash start here.
"""

# figure1 = load_SEIR('All cases')
# print("Reading heatmap data...")
# figure3 = load_heatmap(default_zipcode, default_year)
# print("Reading scatter data...")
# figure2 = load_scatter(default_zipcode, default_year)
# print('SEIR/Scatter/Heatmap loading completed!')

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, title="COVID-19 Dashboard powered by EDEN (USF-COPH-Dr.Edwin Michael Lab)")
#    external_stylesheets=external_stylesheets)

#app.config.suppress_callback_exceptions = True
app.prevent_initial_callbacks=True

cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis', # need local redis server installation using docker(windows) or apt install(linux-ubuntu)
    'CACHE_REDIS_URL': 'redis://localhost:6379'
})

CACHE_TIMEOUT=24*60*60

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
                # html.H4("Choose Year:", className="control_label"),
                # dcc.Dropdown(
                #     id="year_for_all",
                #     options=[{'label': i, 'value': i} for i in ['2020','2021']],
                #     value=default_year
                # ),
                # html.H4("Choose Zip Code: ", className="control_label"),
                # dcc.Dropdown(
                #     id="zipcode_for_all",
                #     options=[{'label': i, 'value': i} for i in ZIPS],
                #     value=default_zipcode
                # ),
                # html.H4("Sampling rate: ", className="control_label"),
                # dcc.Dropdown(
                #     id="sampling_for_all",
                #     options=[{'label': '10 %', 'value':0.1},
                #             {'label': '25 %', 'value':0.25},
                #             {'label': '50 %', 'value': 0.5},
                #             {'label': '75 %', 'value':0.75},
                #             {'label': "100 %", 'value':1.0}],
                #     value=0.25
                # ),
                # dcc.Store(id='year-store-value'),
                # dcc.Store(id='zipcode-store-value'),
                # dcc.Store(id='sampling-store-value'),
           
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
                                dcc.Tab(label='Time Plots', value='tab1', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='Spatial Spread by zipcode', value='tab2', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='Heatmap by zipcode', value='tab3', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='Spatial Spread & Heatmap (Hillsborough County)', value='tab4', style = tab_style, selected_style = tab_selected_style),
                            ], style = tabs_styles),
                            html.Div(
                                id='tabs-contentgraph'),
                                html.Div(
                                    className="footer",
                                    children=[
                                        html.P('Paper: "An Agent-based City Scale Digital Twin (EDEN) for Pandemic Analytics and Scenario Planning, Imran Mahmood et al. (in publishing)"'),
                                        html.H3(children='* Simulation results are provided by Dr. Edwin Michael Lab, USF College of Public Health *'),
                                        html.H4('Team members: Edwin Michael (PI), Imran Mahmood Qureshi Hashmi, Yilian Alonso Otano, Soo I. Kim, Shakir Bilal'),
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

# @app.callback([Output('year-store-value', 'data'), Output('zipcode-store-value', 'data'), Output('sampling-store-value', 'data')],
#             [Input('year_for_all', 'value'), Input('zipcode_for_all', 'value'), Input('sampling_for_all', 'value')],
#             prevent_initial_call=True)
# def store_data(year, zipcode, sampling):
#     global year_for_all
#     global zipcode_for_all
#     global sampling_for_all
#     year_for_all=year
#     zipcode_for_all=zipcode
#     sampling_for_all=sampling
#     print(year_for_all, zipcode_for_all, sampling_for_all)
#     return year_for_all, zipcode_for_all, sampling_for_all

@app.callback(Output('tabs-contentgraph', 'children'), 
    Input('tabsgraph', 'value')) # first page... if uncommented, it will not be displayed
@cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
#def render_content(tab):
def render_content(tab):
    global year_for_all
    global zipcode_for_all
    global sampling_for_all

    global scatter_size
    global heatmap_size

    scatter_size=scatter_size
    heatmap_size=heatmap_size

    print("scatter_size", scatter_size, "heatmap_size", heatmap_size)

    if tab=='moretab':
        return html.Div([
            html.H3('(Except from ongoing paper)'),
            html.H4('An Agent-based City Scale Digital Twin (EDEN) for Pandemic Analytics and Scenario Planning'),
            html.P('This paper presents the development of an agent-based city scale digital twin (EDEN) for the analysis and prediction of COVID-19 transmission across the population at City or County scale. EDEN is a Python based open-source geo-spatial, agent-based, parallel simulation framework. It incorporates GIS data, epidemiological disease parameters and multi-layered, multi-resolution synthetic populations for the study of infectious disease spread (e.g., COVID-19) in a geospatial virtual environment of a selected region. It models the transmission of a selected contagion and simulates its outbreak using computational disease dynamics at a selected spatial resolution (e.g., neighborhood, census tract, zip code, city, county, state, and or country). It forecasts the spread of infections over time, identify spatial hot spots across the region, and estimate the number of infected patients arriving at hospitals. It further allows to apply different public health interventions and to evaluate various lock down scenarios and counterfactuals, thus help critical decision making in rapid emergency response management.'),
            html.H4('Full text will be available here when published. For more info, contact Dr. Edwin Michael (emichael443@usf.edu).'),
            html.Br(),
            html.Br(),
            html.Img(src=app.get_asset_url('USF-EMichael-ABM-EDEN.png'), style={'margin-left': 2, 'width':'497px'}),
            #html.Br(),
            #html.Br(),
            #html.Img(src=app.get_asset_url('usf-logo-white-bg.jfif'), style={'margin-left': 10, 'width':'200px'}),
        ])
    elif tab == 'tab1':
        return html.Div([
            html.Br(),
            html.P(time_plots_explain),
            html.H4("Time Plots Filters:", className="control_label", style={'padding': 10, 'flex': 1}),
            dcc.RadioItems(
                id="filter_type",
                options=[{'label': i, 'value': i} for i in ['All cases',
                                                            'By Age',
                                                            'By Gender',
                                                            'By Race',
                                                            'By FPL']],

                value="All cases",
                labelStyle={'display': 'inline-block'}
            ),
            html.P("* FPL: Federal Poverty Level (%)", style={'padding': 1, 'flex': 1, 'display': 'inline-block'}),
            #html.A("Federal Poverty Level", href='https://www.healthcare.gov/glossary/federal-poverty-level-fpl/', target="_blank", style={'padding': 10, 'flex': 1}),
            #html.Br(),
            html.Div(children=[
                dcc.Graph(
                    id='graph1',
                    #figure=figure1,
                    figure=load_SEIR('All cases'),
                    config={
                        'displayModeBar': False
                    },
                    responsive=True,
                    style={
                        "width": "90%",
                        "height": "100%",
                    }
                )],
                style={
                        "width": "800px",
                        "height": "900px",
                        "display": "inline-block",
                        "overflow": "hidden"
                }
            )
        ])
    elif tab == 'tab2':
        fig=load_scatter(zipcode_for_all, year_for_all, sampling_for_all, graph_width, graph_height, show_whole_county=False)
        return html.Div(id="tab2", children=[
            html.Br(),
            html.H2("Spatial plot of individual daily case emergence and spread"),
            html.P(scatter_map_explain),
            #html.P("(Steps equals Days starting March 1, 2020.)"),
            #html.P("Note: For the fast web response, only a fraction of data is being used here. This page uses "+str(SAMPLING_PERCENT_1*100)+" %", style={'textAlign': 'center', 'color':'orange'}),
            html.Div(children=[
                html.H4("Year:", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="year_scatter",
                    options=[{'label': i, 'value': i} for i in ['2020','2021']],
                    #value=default_year,
                    value=year_for_all,
                    style={'width':'100px', 'display':'inline-block', 'verticalAlign':'middle'}
                ),
                html.H4(", Zip Code: ", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="zipcode_scatter",
                    options=[{'label': i, 'value': i} for i in ZIPS],
                    #value=default_zipcode,
                    value=zipcode_for_all,
                    style={'width':'120px', 'display':'inline-block', 'verticalAlign':'middle'}
                ),
                html.H4(", Sampling rate: ", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="sampling_scatter",
                    options=[{'label': '10 %', 'value':0.1},
                            {'label': '25 %', 'value':0.25},
                            {'label': '50 %', 'value': 0.5},
                            {'label': '75 %', 'value':0.75},
                            {'label': "100 %", 'value':1.0}],
                    #value=0.25,
                    value=sampling_for_all,
                    style={'width':'100px', 'display':'inline-block', 'verticalAlign':'middle'}
                ),
            ], style={'width': '100%', 'display': 'inline-block'}),
            html.Div(id="scatter_size_num", children=[
                #html.P("(Data size="+str(scatter_size)+"), Sampling rate="+str(sampling_for_all), style={'textAlign': 'center', 'color':'orange'}),
                html.P("(Data size="+str(scatter_size)+", Sampling rate="+str(sampling_for_all), style={'textAlign': 'center', 'color':'orange'}),
            ]),
            html.Div(children=[dcc.Graph(id="graph2",
                        figure=fig,
                        config={
                            'displayModeBar': False
                        },
                        responsive=True,
                        style={
                            "width": "100%",
                            "height": "100%",
                        }
                    ),
                    draw_legend_table(),   
                ],
                style={
                        "width": "900px",
                        "height": "750px",
                        "display": "inline-block",
                        "padding-top": "0px",
                        "padding-left": "1px",
                        "overflow": "hidden"
                }
            ),
            html.Div(children=[
                html.A("More info of ZIP="+zipcode_for_all, href='https://www.unitedstateszipcodes.org/'+zipcode_for_all+'/', target="_blank"),
            ]),

        ])
    elif tab == 'tab3':
        fig=load_heatmap(zipcode_for_all, year_for_all, sampling_for_all, graph_width, graph_height, show_whole_county=False)
        return html.Div(id="tab3", children=[
            html.Br(),
            html.H2("Heatmap of the density of daily infectious cases"),
            #html.P("(Z values represents the number of cases within the same zipcode area.)"),
            #html.P("(Steps equals Days starting March 1, 2020)"),
            html.P(heat_map_explain),
            html.Div(children=[
                html.H4("Year:", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="year_heatmap",
                    options=[{'label': i, 'value': i} for i in ['2020','2021']],
                    #value=default_year,
                    value=year_for_all,
                   style={'width':'100px', 'display':'inline-block', 'verticalAlign':'middle'}
                ),
                html.H4(", Zip Code: ", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="zipcode_heatmap",
                    options=[{'label': i, 'value': i} for i in ZIPS],
                    #value=default_zipcode,
                    value=zipcode_for_all,
                    style={'width':'120px', 'display':'inline-block', 'verticalAlign':'middle'}
                ),
                html.H4("Sampling rate: ", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="sampling_heatmap",
                    options=[{'label': '10 %', 'value':0.1},
                            {'label': '25 %', 'value':0.25},
                            {'label': '50 %', 'value': 0.5},
                            {'label': '75 %', 'value':0.75},
                            {'label': "100 %", 'value':1.0}],
                    #value=0.25,
                    value=sampling_for_all,
                    style={'width':'100px', 'display':'inline-block', 'verticalAlign':'middle'}
                ),
            ], style={'width': '100%', 'display': 'inline-block'}),
            #html.P("Note: For the fast web response, only a fraction of data is being used here. This page uses "+str(SAMPLING_PERCENT_2*100)+" %", style={'textAlign': 'center', 'color':'orange'}),
            html.Div(id="heatmap_size_num", children=[
                html.P("(Data size="+str(heatmap_size)+"), Sampling rate="+str(sampling_for_all), style={'textAlign': 'center', 'color':'orange'}),
            ]),
            html.Div(children=[
                    dcc.Graph(
                        id="graph3",
                        figure=fig,
                        config={
                            'displayModeBar': False
                        },
                        responsive=True,
                        style={
                            "width": "100%",
                            "height": "100%",
                        }
                    )
                ],
                style={
                        "width": "900px",
                        "height": "750px",
                        "display": "inline-block",
                        "padding-top": "0px",
                        "padding-left": "1px",
                        "overflow": "hidden"
                }
            ),
            html.Div(children=[
                html.A("More info of ZIP="+zipcode_for_all, href='https://www.unitedstateszipcodes.org/'+zipcode_for_all+'/', target="_blank"),
            ]),

        ])
    elif tab == 'tab4':
        fig1=load_scatter("33510", "2021", 0.01, width=750, height=600, show_whole_county=True)
        fig2=load_heatmap("33510", "2021", 0.01, width=750, height=600, show_whole_county=True)
        return html.Div(id="tab4", children=[
            html.Br(),
            html.Div(children=[
                html.H4("Year:", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="year_for_all",
                    options=[{'label': i, 'value': i} for i in ['2020','2021']],
                    value=default_year,
                    style={'width':'100px', 'display':'inline-block', 'verticalAlign':'middle'}
                ),
                # html.H4(", Zip Code: ", className="control_label", style={'display': 'inline-block'}),
                # dcc.Dropdown(
                #     id="zipcode_for_all",

                #     options=[{'label': i, 'value': i} for i in ZIPS],
                #     value=default_zipcode,
                #     style={'width':'120px', 'display':'inline-block', 'verticalAlign':'middle'}
                # ),
                html.H4(", Sampling rate: ", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="sampling_for_all",
                    options=[{'label': '1 %', 'value':0.01},
                            {'label': '5 %', 'value':0.05},
                            {'label': '10 %', 'value': 0.1},
                            #{'label': '75 %', 'value':0.75},
                            #{'label': "100 %", 'value':1.0},
                    ],
                    value=0.01,
                    style={'width':'100px', 'display':'inline-block', 'verticalAlign':'middle'}
                ),
            ], style={'width': '100%', 'display': 'inline-block'}),
            html.Div(id="data_size_num", children=[
                html.P("(Scatter data size="+str(scatter_size)+", Heatmap data size="+str(heatmap_size)+")", style={'textAlign': 'center', 'color':'orange'}),
            ]),

            html.Div(children=[
                html.Div(
                    dcc.Graph(
                        id='graph22',
                        figure= fig1,
                        config={
                            'displayModeBar': False
                        }                
                    ), style={'display': 'inline-block'}),
                html.Div(
                    dcc.Graph(
                        id='graph33',
                        figure=fig2,
                        config={
                            'displayModeBar': False
                        }                
                    ), style={'display': 'inline-block'})
            ], style={'width': '100%', 'display': 'inline-block'})            
        ])

@app.callback(Output("graph1", 'figure'), Input("filter_type", "value"),
            prevent_initial_call=True)
@cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
def update_SEIR(filter_type):
    return load_SEIR(filter_type)

@app.callback(Output("graph2", 'figure'), Output("scatter_size_num", "children"),
            #[Input("zipcode-store-value", "data"), Input("year-store-value", "data"), Input('sampling-store-value', 'data')],
            [Input("zipcode_scatter", "value"), Input("year_scatter", "value"), Input('sampling_scatter', 'value')],
            prevent_initial_call=True)
@cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
def update_scatter_by_zipcode(zipcode, year, sampling):
    global scatter_size
    global sampling_for_all
    global year_for_all
    global zipcode_for_all

    sampling_for_all=sampling
    zipcode_for_all=zipcode
    year_for_all=year

    time.sleep(1)
    return load_scatter(zipcode, year, sampling, width=graph_width, height=graph_height, show_whole_county=False),html.Div(children=[
            html.P("(Data size="+str(scatter_size)+"), Sampling rate="+str(sampling)),
            html.Br(), 
            html.A("More info of ZIP="+zipcode, href='https://www.unitedstateszipcodes.org/'+zipcode+'/', target="_blank")
        ])

@app.callback(Output("graph3", 'figure'), Output("heatmap_size_num", "children"),
            #[Input("zipcode-store-value", "data"), Input("year-store-value", "data"), Input('sampling-store-value', 'data')],
            [Input("zipcode_heatmap", "value"), Input("year_heatmap", "value"), Input('sampling_heatmap', 'value')],
            prevent_initial_call=True)
@cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
def update_heatmap_by_zipcode(zipcode, year, sampling):
    global heatmap_size
    global sampling_for_all
    global year_for_all
    global zipcode_for_all

    sampling_for_all=sampling
    zipcode_for_all=zipcode
    year_for_all=year

    time.sleep(1)
    return load_heatmap(zipcode, year, sampling, width=graph_width, height=graph_height, show_whole_county=False), html.Div(children=[
            html.P("(Data size="+str(heatmap_size)+"), Sampling rate="+str(sampling)),
            html.Br(), 
            html.A("More info of ZIP="+zipcode, href='https://www.unitedstateszipcodes.org/'+zipcode+'/', target="_blank")
        ])
# @app.callback([Output("graph22", 'figure'), Output("graph33", 'figure')],
#             [Input("zipcode_for_all", "value"), Input("year_for_all", "value")],
#             prevent_initial_call=True)
# @cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
# def update_scatter_and_heatmap(zipcode_for_all, year_for_all):
#     time.sleep(1)
#     figure22 = load_scatter(zipcode_for_all, year_for_all, width=600, height=600)
#     figure33 = load_heatmap(zipcode_for_all, year_for_all, width=600, height=600)
#     return figure22, figure33

@app.callback([
            Output("graph22", 'figure'), 
            Output("graph33", 'figure'),
            Output("data_size_num", "children")
            ],
            [
            # Input("zipcode_for_all", "value"),
            Input("year_for_all", "value"),
            Input("sampling_for_all", "value")
            ],
            prevent_initial_call=True)
@cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
#def update_scatter_and_heatmap(zipcode, year, sampling):
def update_scatter_and_heatmap(year, sampling):
    global heatmap_size
    global scatter_size
    time.sleep(1)
    figure22 = load_scatter("33510", year, sampling, width=750, height=600, show_whole_county=True)
    figure33 = load_heatmap("33510", year, sampling, width=750, height=600, show_whole_county=True)

    return figure22, figure33, html.Div(children=[
                html.P("(Scatter data size="+str(scatter_size)+", Heatmap data size="+str(heatmap_size)+")"),
            ]),

if __name__ == '__main__':
    app.run_server(debug=False,
        #use_reloader=False,
        threaded=True,
        host="0.0.0.0",
        port=8050)

