import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import  plotly as py
import pandas as pd
import sys
import numpy as np
import os
import geopandas as gpd
import pandas as pd
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

SF1 = 22
SF2 = 5

path = os.path.join('C:\\', 'Users', 'ihashmi', 'Desktop', 'EDEN-ABM-Simulator', 'SimulationEngine', 'output', '2021-12-29', 'run4')
print(path)
def plot(min, mean, max):
    sub_groups = ['Cases', 'Admissions', 'Deaths']
    fig = make_subplots(rows=1, cols=3, subplot_titles=sub_groups
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
                             name="admissions", line=dict({'width': 2, 'color': 'green'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=min['date'], y=min['admissions']/(SF1/2),
                             name="admissions", fillcolor='rgba(217, 255, 217,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(204,255,204,1)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=max['date'], y=max['admissions']/SF1,
                             name="admissions", fillcolor='rgba(217, 242, 217,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(204,255,204,1)'})), row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vadmissions'],
                             name="actual admissions", line=dict({'width': 2, 'color': 'blue', 'dash': 'dot'})), row=1,
                  col=2)


    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['deaths'] / SF2,
                             name="deaths", line=dict({'width': 2, 'color': 'black'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=min['deaths'] / (SF2 / 2),
                             name="deaths", fillcolor='rgba(230, 230, 230,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(192,192,192,1)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=max['deaths'] / (SF2 * 2),
                             name="deaths", fillcolor='rgba(230, 230, 230,0.75)', fill='tonexty',
                             line=dict({'width': 1, 'color': 'rgba(192,192,192,1)'})), row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vdeaths'],
                             name="actual deaths", line=dict({'width': 2, 'color': 'blue', 'dash': 'dot'})), row=1,
                  col=3)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, autosize=True, width=1200, height=400,
                      legend=dict(orientation="h",x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

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
    fig.update_layout(showlegend=True, autosize=True, width=1200, height=1500,
                      legend=dict(orientation="h", x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyAge-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
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
    fig.update_layout(showlegend=True, autosize=True, width=1400, height=1500,
                      legend=dict(orientation="h", x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyAge-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
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
    fig.update_layout(showlegend=True, autosize=True, width=1200, height=1800,
                      legend=dict(orientation="h", x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyRace-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
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
    fig.update_layout(showlegend=True, autosize=True, width=1200, height=1800,
                      legend=dict(orientation="h",x=0, y=-0.5, traceorder="normal"),
                      font=dict(family="Arial", size=12))

    # py.offline.plot(fig, filename=os.path.join(os.path.dirname(os.getcwd()), 'SEIRbyFPL-' + datetime.now().strftime("%Y-%m-%d")+ '.html'))
    return fig

def get_RMSE(zip, actual, simulated):
    mse = mean_squared_error(actual, simulated)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, simulated)
    return rmse

def load_SEIR(mode):
    if mode=='All':
        dlist = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith("plot_"):
                    no = file.split('_')[1].split('.')[0]
                    d = pd.read_csv(os.path.join(root, file))
                    # d['chunk'] = no
                    # if get_RMSE(no, d['vcases'], d['cases']) < 4500:
                    dlist.append(d)

        print('Loading completed!')
        plotdf = pd.concat(dlist, axis=1)
        dates = dlist[0]['date'].tolist()
        plotdf.drop('date', axis=1, inplace=True)

        max = plotdf.groupby(plotdf.columns, axis=1).max()
        mean = plotdf.groupby(plotdf.columns, axis=1).mean()
        min = plotdf.groupby(plotdf.columns, axis=1).min()
        max['date'] = dates
        mean['date'] = dates
        min['date'] = dates
        print('Loading completed!')
        plot(min, mean, max)
    else:
        dlist = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith("plot_"):
                    no = file.split('_')[1].split('.')[0]
                    d = pd.read_csv(os.path.join(root, file))
                    d['chunk'] = no
                    dlist.append(d)

        print('Loading completed!')
        plotdf = pd.concat(dlist)
        plotdf = plotdf.sort_values(by='date')

        if mode == 'ByAge':
            plot_age(plotdf)
        elif mode == 'ByGender':
            plot_gender(plotdf)
        elif mode == 'ByRace':
            plot_race(plotdf)
        elif mode == 'ByFPL':
            plot_FPL(plotdf)

def load_scatter():
    pdf = pd.read_csv(os.path.join(path, 'scatter.csv'))
    pdf = pdf.sample(frac=0.05)
    pdf = pdf.sort_values(by='step')
    fig = px.scatter_mapbox(pdf,
                            color=pdf['state'],
                            lat=pdf['y'],
                            lon=pdf['x'],
                            animation_frame=pdf['step'],
                            zoom=10,
                            height=800,
                            width=1500,
                            center=dict(lat=28.03711, lon=-82.46390),
                            # mapbox_style='open-street-map'
                            mapbox_style='open-street-map'
                            )
    return fig


def load_heatmap():
    pdf = pd.read_csv(os.path.join(path, 'heatmap.csv'))
    pdf = pdf.sample(frac=0.05)
    pdf = pdf.sort_values(by='step')
    fig = px.density_mapbox(pdf,
                            color_continuous_scale='RdYlGn_r',
                            lat=pdf['y'],
                            lon=pdf['x'],
                            z=pdf['z'],
                            animation_frame=pdf['step'],
                            zoom=8,
                            opacity=0.75,
                            height=800,
                            width=1500,
                            center=dict(lat=28.03711, lon=-82.46390),
                            # mapbox_style='open-street-map'
                            mapbox_style='stamen-terrain')
    return fig

figure1 = load_SEIR('All')
# figure2 = load_scatter()
# figure3 = load_heatmap()
print('loading completed!')

app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.Div(
        className="container",
        children=[
            html.Div(
                className="header",
                children=[
                    html.H2(children='Department of Flordia Health'),
                    html.H3("Hillsborough COVID-19 Dashboard"),
                ]
            ),
            html.Div(
                className="nav",
                children=[
                    html.P(
                        "Select Forecast Range):",
                        className="control_label",
                    ),
                    html.Br(),
                    dcc.RangeSlider(
                        id="year_slider",
                        min=0,
                        max=3,
                        step=None,
                        marks={
                            0: "2020",
                            1: "2021",
                            2: "2022",
                            3: "2023"
                        },
                        value=[0, 1],
                    ),
                    html.Br(),
                    html.P("Filter by:", className="control_label"),
                    dcc.RadioItems(
                        id="filter_type",
                        options=[{'label': i, 'value': i} for i in ['All',
                                                                    'ByAge',
                                                                    'ByGender',
                                                                    'ByRace',
                                                                    'ByFPL']],

                        value="All",
                        labelStyle={'display': 'block'},
                    ),
                    html.Br(),
                    html.P("Mask adoption level (0 - 100 %):", className="control_label"),
                    dcc.Slider(
                        id='mask_adoption_level',
                        min=0,
                        max=100,
                        step=None,
                        value=50,
                        marks={
                            0: '0%',
                            25: '20%',
                            50: '50%',
                            75: '75%',
                            100: '100%'
                        },
                    ),
                    html.Br(),
                    html.P("Social Distancing Compliance:", className="control_label"),
                    dcc.RadioItems(
                        id="social_distancing",
                        options=[{'label': i, 'value': i} for i in ['Yes',
                                                                    'No']],

                        value="No",
                    ),
                ]
            ),
            html.Div(
                className="section",
                children=[
                            dcc.Tabs(id="tabsgraph", value='graph1', children=[
                                dcc.Tab(label='COVID-19 Time Plots', value='graph1'),
                                dcc.Tab(label='COVID-19 Spatial Spread', value='graph2'),
                                dcc.Tab(label='COVID-19 Heatmap', value='graph3'),
                            ]),
                            html.Div(id='tabs-contentgraph')
                ]
            ),
            html.Div(
                className="footer",
                children=[
                    html.H3(children='Footer Department of Flordia Health'),
                ]
            ),
        ]
    ),
])
@app.callback(Output('tabs-contentgraph', 'children'),
              Input('tabsgraph', 'value'))
def render_content(tab):
    if tab == 'graph1':
        return html.Div([
            dcc.Graph(
                id='graph1',
                figure=figure1
            )
        ])
    elif tab == 'graph2':
        return html.Div([
            dcc.Graph(
                id='graph2',
                figure=figure2
            )
        ])
    elif tab == 'graph3':
        return html.Div([
            dcc.Graph(
                id='graph3',
                figure=figure3
            )
        ])

@app.callback(
    Output("graph1", 'figure'),
    [
        Input("filter_type", "value"),
    ],
)
def update_SEIR(filter_type):
    figure1 = load_SEIR(filter_type)
    return figure1

if __name__ == '__main__':
    app.run_server()
