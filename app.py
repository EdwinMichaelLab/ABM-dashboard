from collections import OrderedDict
from sqlite3 import DateFromTicks

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
from flask_caching import Cache
import numbers

import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly

folder_name=""
"""
scaling factors

Imran says: Due to computation limit, there's no interaction between ZIPCODES. 
So simulated results of cases/admissions/deaths becomes over-estimated. So that's why using scaling factors here.
In future, logics of calibration will be applied instead to address this issue.
"""

# SF_cases=12
# SF_admissions=3.125
# SF_deaths=10
SF_cases=0.02
SF_admissions=0.05
SF_deaths=1

#import vaex

# import pymongo
# from pymongo import MongoClient
# client = MongoClient()
# print(client)
# db = client["abm"]

ZIPS = ['33510', '33511', '33527', '33534', '33547', '33548', '33549', '33556', '33558', '33559', '33563', '33565',
            '33566', '33567', '33569', '33570', '33572', '33573', '33578', '33579', '33584', '33592', '33594', '33596',
            '33598', '33602', '33603', '33604', '33605', '33606', '33607', '33609', '33610', '33611', '33612', '33613',
            '33614', '33615', '33616', '33617', '33618', '33619', '33624', '33625', '33626', '33629', '33634', '33635',
            '33637', '33647',
            '33620', '33503']
colors_list=[
"aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
                "beige", "bisque", "black", "blanchedalmond", "blue",
                "blueviolet", "brown", "burlywood", "cadetblue",
                "chartreuse", "chocolate", "coral", "cornflowerblue",
                "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
                "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
                "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
                "darkorchid", "darkred", "darksalmon", "darkseagreen",
                "darkslateblue", "darkslategray", "darkslategrey",
                "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
                "dimgray", "dimgrey", "dodgerblue", "firebrick",
                "floralwhite", "forestgreen", "fuchsia", 
                #"gainsboro",
                #"ghostwhite", "gold", "goldenrod", 
                #"gray", "grey", 
                #"green",
                #"greenyellow", "honeydew", "hotpink", "indianred", 
                #"indigo",
                #"ivory", "khaki", "lavender", "lavenderblush", 
                #"lawngreen",
                #"lemonchiffon", 
                "lightblue", "lightcoral", "lightcyan",
                "lightgoldenrodyellow", "lightgray", "lightgrey",
                "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
                "lightskyblue", "lightslategray", "lightslategrey",
                "lightsteelblue", "lightyellow", "lime", "limegreen",
                "linen", "magenta", "maroon", "mediumaquamarine",
                "mediumblue", "mediumorchid", "mediumpurple",
                "mediumseagreen", "mediumslateblue", "mediumspringgreen",
                "mediumturquoise", "mediumvioletred", "midnightblue",
                "mintcream", "mistyrose", "moccasin", "navajowhite", "navy",
                "oldlace", "olive", "olivedrab", "orange", "orangered",
                "orchid", "palegoldenrod", "palegreen", "paleturquoise",
                "palevioletred", "papayawhip", "peachpuff", "peru", "pink",
                "plum", "powderblue", "purple", "red", "rosybrown",
                "royalblue", "saddlebrown", "salmon", "sandybrown",
                "seagreen", "seashell", "sienna", "silver", "skyblue",
                "slateblue", "slategray", "slategrey", "snow", "springgreen",
                "steelblue", "tan", "teal", "thistle", "tomato", "turquoise",
                "violet", "wheat", 
                #"white", 
                "whitesmoke", "yellow",
                "yellowgreen"
]
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
                '33647':[28.12, -82.35],
                '33620':[28.062, -82.410],
                '33503':[27.753, -82.2883]}
MAX_ROWS=10000000

if len(sys.argv)==2:
    print("Try to launch plotly-dash webapp using results in ../output/"+str(sys.argv[1]))
    folder_name=sys.argv[1]
    if (os.path.exists(os.path.join('..', 'ABM-simulator', 'SimulationEngine', 'output',folder_name))):
        pass
    else:
        print("Floder("+folder_name+") does not exist")
        exit()
else:
    print("Please type result-folder name.")
    print('Usage: python '+str(sys.argv[0])+' result_folder_name')
    exit()

path = os.path.join('..', 'ABM-simulator', 'SimulationEngine', 'output', folder_name)

print(path)

# SF1 = 22
# SF2 = 5

#CHUNK_SIZE1=2000000
#CHUNK_SIZE2=2000000

# scatter
#SKIP_EVERY_NTH_1=100 # best at 2
#SAMPLING_PERCENT_1=0.5 # default 0.25

# heatmap
#SKIP_EVERY_NTH_2=10 # best at 2
#SAMPLING_PERCENT_2=0.5 # default 0.5

startdate = date(2020, 3, 1)
enddate = date(2022, 1, 31)

default_zipcode ="33510"
default_year ="2021"
default_sampling=0.25
default_zoom=9

year_for_all="2021"
zipcode_for_all="33510"
sampling_for_all=0.25 # 0.5=50%
show_whole_county=False

filter_type="All cases" #id=filter_type's values

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

fillcolor1='rgb(184, 247, 212)'
fillcolor2='rgb(111, 231, 219)'
fillcolor3='rgb(127, 166, 238)'
fillcolor4='rgb(131, 90, 241)'
fillcolor5='rgb(141, 80, 251)'

def get_RMSE(zip, actual, simulated):
    mse = mean_squared_error(actual, simulated)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, simulated)
    return rmse

def calc_7_day_average(df):
    return df.rolling(window=7).mean()

def calc_N_day_average(df, ndays=7):
    return df.rolling(window=ndays).mean()

def calc_mean(df):
    df=df.transpose()
    df = df.groupby(by=df.index, axis=0).apply(lambda g: g.mean() if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[0])
    return df.transpose()

# to smooth spotty data by connecting peaks
def upper_envelope(df, windowsize=20):
    return df.rolling(window=windowsize).max().shift(int(-windowsize/2))

# def plot2(min, mean, max):
#     # Create figure
#     fig = go.Figure()

#     sub_groups = ['Cases', #'Actual cases', 
#                 'Admissions', #'Actual admissions', 
#                 'Deaths', #'Actual deaths'
#                 '(Accumulated) Deaths', #'Actual deaths'
#                 ]
#     fig = make_subplots(rows=4, cols=1, 
#         subplot_titles=sub_groups, 
#         shared_xaxes=True, 
#         #specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]],
#         vertical_spacing = 0.05,
#         row_width=[0.2, 0.2, 0.25, 0.25])

#     mean['vcases']=upper_envelope(mean['vcases'], 7)
#     mean['vadmissions']=upper_envelope(mean['vadmissions'], 7)
#     mean['vdeaths']=upper_envelope(mean['vdeaths'], 7)
#     mean['cases']=upper_envelope(mean['cases'], 7)
#     mean['admissions']=upper_envelope(mean['admissions'], 7)
#     mean['deaths']=upper_envelope(mean['deaths'], 7)

#     min['deaths_cumsum']=min['deaths'].cumsum()
#     max['deaths_cumsum']=max['deaths'].cumsum()
#     mean['deaths_cumsum']=mean['deaths'].cumsum()
#     mean['vdeaths_cumsum']=mean['vdeaths'].cumsum()
  
#     fig.add_trace(go.Scatter(fill='tonexty', x=max['date'], y=max['cases']/SF_cases,
#                              name="max cases", 
#                              line=dict({'width': 1, 'color': 'pink'})),
#                 row=1, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['cases']/SF_cases,
#                              name="mean cases", 
#                              line=dict({'width': 1, 'color': 'red'})),
#                 row=1, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=min['date'], y=min['cases']/SF_cases,
#                              name="min cases",
#                              line=dict({'width': 1, 'color': 'darkorange'})),
#                 row=1, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vcases'],
#                              name="vcases", 
#                              line=dict({'width': 1, 'color': 'black', 'dash':'dot'})),
#                              #line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True,
#                 row=1, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=max['date'], y=max['admissions']/SF_admissions,
#                              name="max admissions", 
#                              line=dict({'width': 1, 'color': 'lightgreen'})),
#                 row=2, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['admissions']/SF_admissions,
#                              name="mean admissions", 
#                              line=dict({'width': 1, 'color': 'darkgreen'})),
#                 row=2, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=min['date'], y=min['admissions']/SF_admissions,
#                              name="min admissions", 
#                              line=dict({'width': 1, 'color': 'green'})),
#                 row=2, col=1)

#     fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vadmissions'],
#                              name="actual admissions", 
#                             line=dict({'width': 1, 'color': 'black', 'dash':'dot'})),
#                               #line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True,
#                 row=2, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=max['date'], y=max['deaths']/SF_deaths,
#                              name="max deaths", 
#                              line=dict({'width': 1, 'color': 'darkgrey'})), 
#                 row=3, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['deaths']/SF_deaths,
#                              name="mean deaths", 
#                              line=dict({'width': 1, 'color': 'black'})),
#                 row=3, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=min['date'], y=min['deaths']/SF_deaths,
#                              name="min deaths", 
#                              line=dict({'width': 1, 'color': 'darkgrey'})),
#                 row=3, col=1)
#     fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vdeaths'],
#                              name="actual deaths", 
#                              line=dict({'width': 1, 'color': 'black', 'dash':'dot'})),
#                              #line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True,
#                 row=3, col=1)
                
#     fig.add_trace(go.Scatter(fill='tonexty', x=max['date'], y=max['deaths_cumsum']/SF_deaths,
#                              name="max deaths", 
#                              line=dict({'width': 1, 'color': 'darkgrey'})), 
#                 row=4, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['deaths_cumsum']/SF_deaths,
#                              name="mean deaths", 
#                              line=dict({'width': 1, 'color': 'black'})),
#                 row=4, col=1)
#     fig.add_trace(go.Scatter(fill='tonexty', x=min['date'], y=min['deaths_cumsum']/SF_deaths,
#                              name="min deaths", 
#                              line=dict({'width': 1, 'color': 'darkgrey'})),
#                 row=4, col=1)
#     fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vdeaths_cumsum'],
#                              name="actual deaths", 
#                              line=dict({'width': 1, 'color': 'black', 'dash':'dot'})),
#                              #line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True,
#                 row=4, col=1)
#     fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
#     fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#     fig.update_layout(showlegend=True, autosize=True, 
#                     #width=900, height=800,
#                     #legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
#                       font=dict(family="Arial", size=11))

#     #fig.show()
#     return fig

def load_allzipcodes():
    dlist = []
    for root, dirs, files in os.walk(path):
        #print(root)
        #print(dirs)
        #print(files)
        for file in files:
            if file.startswith("plot_"):
                zip = file.split('_')[1]
                zip = zip.replace('.csv', '')
                if zip in ZIPS:
                    fpath=os.path.join(root,file)
                    d = pd.read_csv(fpath)
                    d['zipcode']=zip
                    dlist.append(d)
    df2 = pd.concat(dlist)
    df2.drop('date', axis=1, inplace=True)
    dates = dlist[0]['date'].tolist()
    del dlist # free up memory

    allcases=pd.DataFrame(columns=ZIPS)
    alladmissions=pd.DataFrame(columns=ZIPS)
    alldeaths=pd.DataFrame(columns=ZIPS)
    allcases['date']=dates
    alladmissions['date']=dates
    alldeaths['date']=dates
    days_interval=10
    # remove not-simulated zips
    ZIPS.remove('33503')
    ZIPS.remove('33620')
    for zip in ZIPS:
        allcases[zip]=calc_N_day_average(df2[df2['zipcode']==zip]['cases'].to_frame(), days_interval)
        alladmissions[zip]=calc_N_day_average(df2[df2['zipcode']==zip]['admissions'].to_frame(), days_interval)
        alldeaths[zip]=calc_N_day_average(df2[df2['zipcode']==zip]['deaths'].to_frame(), days_interval)

    # fig_cases= go.Figure()
    # fig_admissions= go.Figure()
    # fig_deaths= go.Figure()
    # for zip in ZIPS:
    #     fig_cases.add_trace(go.Scatter(x=allcases['date'], y=allcases[zip],
    #         mode='lines', name=zip
    #     ))
    #     fig_admissions.add_trace(go.Scatter(x=alladmissions['date'], y=alladmissions[zip],
    #         mode='lines', name=zip
    #     ))
    #     fig_deaths.add_trace(go.Scatter(x=alldeaths['date'], y=alldeaths[zip],
    #         mode='lines', name=zip
    #     ))
    # return fig_cases, fig_admissions, fig_deaths

    sub_groups = ['Cases (All zip codes)', 
                'Admissions (All zip codes)', 
                'Deaths (All zip codes)', ]

    fig_subplots = make_subplots(rows=3, cols=1, 
        subplot_titles=sub_groups, 
        shared_xaxes=True, 
        shared_yaxes=True,
        vertical_spacing = 0.08,
        row_width=[0.1, 0.1, 0.1])
    for zip in ZIPS:
        c=colors_list[ZIPS.index(zip)]

        fig_subplots.add_trace(go.Scatter(x=allcases['date'], y=allcases[zip],
            mode='lines', name=zip, text=zip, line_shape='spline',line=dict(color=c),
        ),row=1, col=1)
        fig_subplots.add_trace(go.Scatter(x=alladmissions['date'], y=alladmissions[zip],
            mode='lines', name=zip, text=zip, line_shape='spline',showlegend=False,line=dict(color=c),
        ),row=2, col=1)
        fig_subplots.add_trace(go.Scatter(x=alldeaths['date'], y=alldeaths[zip],
            mode='lines', name=zip, text=zip, line_shape='spline',showlegend=False,line=dict(color=c),
        ),row=3, col=1)
    fig_subplots.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig_subplots.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig_subplots.update_layout(showlegend=True, autosize=True,
                    #width=900, height=800,
                    #legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
                    font=dict(family="Arial", size=11))
    return fig_subplots

def plot3(min, mean, max):
    # Create figure
    fig = go.Figure()

    sub_groups = ['Cases', #'Actual cases', 
                'Admissions', #'Actual admissions', 
                'Deaths', #'Actual deaths'
                '(Accumulated) Deaths', #'Actual deaths'
                ]
    fig = make_subplots(rows=4, cols=1, 
        subplot_titles=sub_groups, 
        shared_xaxes=True, 
        #specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]],
        vertical_spacing = 0.05,
        row_width=[0.1, 0.1, 0.2, 0.2])

    # days_interval=10
    # mean['vcases']=upper_envelope(mean['vcases'], days_interval)
    # mean['vadmissions']=upper_envelope(mean['vadmissions'], days_interval)
    # mean['vdeaths']=upper_envelope(mean['vdeaths'], days_interval)

    # mean['cases']=upper_envelope(mean['cases'], days_interval)
    # mean['admissions']=upper_envelope(mean['admissions'], days_interval)
    # mean['deaths']=upper_envelope(mean['deaths'], days_interval)

    # min['cases']=upper_envelope(min['cases'], days_interval)
    # min['admissions']=upper_envelope(min['admissions'], days_interval)
    # min['deaths']=upper_envelope(min['deaths'], days_interval)

    # max['cases']=upper_envelope(max['cases'], days_interval)
    # max['admissions']=upper_envelope(max['admissions'], days_interval)
    # max['deaths']=upper_envelope(max['deaths'], days_interval)

    days_interval=7
    mean['vcases']=calc_N_day_average(mean['vcases'], days_interval)
    mean['vadmissions']=calc_N_day_average(mean['vadmissions'], days_interval)
    mean['vdeaths']=calc_N_day_average(mean['vdeaths'], days_interval)

    days_interval=10
    mean['cases']=calc_N_day_average(mean['cases'], days_interval)
    mean['admissions']=calc_N_day_average(mean['admissions'], days_interval)
    mean['deaths']=calc_N_day_average(mean['deaths'], days_interval)

    min['cases']=calc_N_day_average(min['cases'], days_interval)
    min['admissions']=calc_N_day_average(min['admissions'], days_interval)
    min['deaths']=calc_N_day_average(min['deaths'], days_interval)

    max['cases']=calc_N_day_average(max['cases'], days_interval)
    max['admissions']=calc_N_day_average(max['admissions'], days_interval)
    max['deaths']=calc_N_day_average(max['deaths'], days_interval)


    mean['deaths_cumsum']=mean['deaths'].cumsum()
    mean['vdeaths_cumsum']=mean['vdeaths'].cumsum()
    min['deaths_cumsum']=min['deaths'].cumsum()
    max['deaths_cumsum']=max['deaths'].cumsum()

    # for Imran logic
    # SF_cases=5
    # SF_admissions=5
    # SF_deaths=1
    SF_cases=1/50
    SF_admissions=1/15
    SF_deaths=1

    fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['cases']/SF_cases,
                             name="mean cases", 
                             line_shape='spline',
                             line=dict({'width': 1, 'color': 'red'})),
                row=1, col=1)
    fig.add_trace(go.Scatter(fill='tonexty', x=max['date'], y=max['cases']/SF_cases,
                             name="max cases", 
                             line_shape='spline',
                             line=dict({'width': 1, 'color': 'pink'})),
                row=1, col=1)
    fig.add_trace(go.Scatter(fill='tonexty', x=min['date'], y=min['cases']/SF_cases,
                             name="min cases", 
                             line_shape='spline',
                             line=dict({'width': 1, 'color': 'darkorange'})),
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vcases'],
                             name="vcases",
                             line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
                row=1, col=1)

    fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['admissions']/SF_admissions,
                             name="mean admissions", 
                             line_shape='spline',
                             line=dict({'width': 1, 'color': 'darkgreen'})),
                row=2, col=1)
    fig.add_trace(go.Scatter(fill='tonexty', x=max['date'], y=max['admissions']/SF_admissions,
                             name="max admissions", 
                             line_shape='spline',
                             line=dict({'width': 1, 'color': 'lightgreen'})),
                row=2, col=1)
    fig.add_trace(go.Scatter(fill='tonexty', x=min['date'], y=min['admissions']/SF_admissions,
                             name="min admissions", 
                             line_shape='spline',
                             line=dict({'width': 1, 'color': 'green'})),
                row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vadmissions'],
                             name="actual admissions", 
                             line_shape='spline',
                            line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
                row=2, col=1)

    fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['deaths']/SF_deaths,
                             name="mean deaths", line_shape='spline',
                             line=dict({'width': 1, 'color': 'grey'})),
                row=3, col=1)
    fig.add_trace(go.Scatter(fill='tonexty', x=max['date'], y=max['deaths_cumsum']/SF_deaths,
                             name="max deaths", line_shape='spline',
                             line=dict({'width': 1, 'color': 'lightgrey'})), 
                row=4, col=1)
    fig.add_trace(go.Scatter(fill='tonexty', x=min['date'], y=min['deaths_cumsum']/SF_deaths,
                             name="min deaths", line_shape='spline',
                             line=dict({'width': 1, 'color': 'darkgrey'})),
                row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vdeaths'],
                             name="actual deaths", 
                             line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
                row=3, col=1)
                

    fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['deaths_cumsum']/SF_deaths,
                             name="mean deaths", 
                             line=dict({'width': 1, 'color': 'grey'})),
                row=4, col=1)

    fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vdeaths_cumsum'],
                             name="actual deaths", 
                             line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
                row=4, col=1)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, autosize=True, 
                    #width=900, height=800,
                    #legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
                      font=dict(family="Arial", size=11))

    #fig.show()
    return fig

def plot3_weekly(min, mean, max):

    # max of cases/admissions/deaths are... too jagged... smoothing them here
    # days_interval=10
    # max['cases']=upper_envelope(max['cases'], days_interval)
    # max['admissions']=upper_envelope(max['admissions'], days_interval)
    # max['deaths']=upper_envelope(max['deaths'], days_interval)

    # mean['Date'] = pd.to_datetime(mean['date'])
    # mean['Week_Number'] = mean['Date'].dt.isocalendar().week
    # mean['Year']=(mean['Date'] - mean['Date'].dt.weekday * timedelta(days=1)).dt.year
    # mean.set_index('Date')

    # min['Date'] = pd.to_datetime(min['date'])
    # min['Week_Number'] = min['Date'].dt.isocalendar().week
    # min['Year']=(min['Date'] - min['Date'].dt.weekday * timedelta(days=1)).dt.year
    # min.set_index('Date')

    # max['Date'] = pd.to_datetime(max['date'])
    # max['Week_Number'] = max['Date'].dt.isocalendar().week
    # max['Year']=(max['Date'] - max['Date'].dt.weekday * timedelta(days=1)).dt.year
    # max.set_index('Date')


    days_interval=7
    mean['vcases']=calc_N_day_average(mean['vcases'], days_interval)
    #mean['vadmissions']=calc_N_day_average(mean['vadmissions'], days_interval)
    #mean['vadmissions']=upper_envelope(mean['vadmissions'], days_interval)
    mean['vdeaths']=calc_N_day_average(mean['vdeaths'], days_interval)

    days_interval=10
    mean['cases']=calc_N_day_average(mean['cases'], days_interval)
    mean['admissions']=calc_N_day_average(mean['admissions'], days_interval)
    #mean['admissions']=upper_envelope(mean['admissions'], days_interval)
    mean['deaths']=calc_N_day_average(mean['deaths'], days_interval)

    # Create figure
    fig = go.Figure()

    sub_groups = ['Cases', #'Actual cases', 
                'Admissions', #'Actual admissions', 
                'Deaths', #'Actual deaths'
                #'(Accumulated) Deaths', #'Actual deaths'
                ]
    fig = make_subplots(rows=3, cols=1, 
        subplot_titles=sub_groups, 
        shared_xaxes=True, 
        #specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]],
        vertical_spacing = 0.05,
        row_width=[0.1, 0.2, 0.2])

    mean['deaths_cumsum']=mean['deaths'].cumsum()
    mean['vdeaths_cumsum']=mean['vdeaths'].cumsum()
    # min['deaths_cumsum']=min['deaths'].cumsum()
    # max['deaths_cumsum']=max['deaths'].cumsum()

    # mean = mean.groupby(['Year','Week_Number'], as_index=False).sum()

    #mean['year_week']=mean['Year'].astype(str).str[-2:]+'-W'+mean['Week_Number'].astype(str)

    # mean['year_week']=mean['Year'].astype(str)+'-W'+mean['Week_Number'].astype(str)
    # mean['year_week2']=mean['Year'].astype(int)*100+mean['Week_Number'].astype(int)
    # mean['date'] = pd.to_datetime(mean['year_week2'].astype(str) + '0', format='%Y%W%w') # do not use: last week of the year become weird...

    # min['year_week']=min['Year'].astype(str)+'-W'+min['Week_Number'].astype(str)
    # min['year_week2']=min['Year'].astype(int)*100+min['Week_Number'].astype(int)
    # min['date'] = pd.to_datetime(min['year_week2'].astype(str) + '0', format='%Y%W%w') # do not use: last week of the year become weird...

    # max['year_week']=max['Year'].astype(str)+'-W'+max['Week_Number'].astype(str)
    # max['year_week2']=max['Year'].astype(int)*100+max['Week_Number'].astype(int)
    # max['date'] = pd.to_datetime(max['year_week2'].astype(str) + '0', format='%Y%W%w') # do not use: last week of the year become weird...

    # fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['cases']/SF_cases,
    #                          name="Simulated cases", 
    #                          line=dict({'width': 1, 'color': 'red'})),
    #             row=1, col=1)

    # fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vcases'],
    #                          name="Actual cases", 
    #                          line=dict({'width': 2, 'color': 'darkred', 'dash':'dot'})),
    #             row=1, col=1)

    # fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['admissions']/SF_admissions,
    #                          name="Simulated admissions", 
    #                          line=dict({'width': 1, 'color': 'darkgreen'})),
    #             row=2, col=1)

    # fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vadmissions'],
    #                          name="Actual admissions", 
    #                         line=dict({'width': 2, 'color': 'teal', 'dash':'dot'})),
    #             row=2, col=1)

    # fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['deaths']/SF_deaths,
    #                          name="Simulated deaths", 
    #                          line=dict({'width': 1, 'color': 'grey'})),
    #             row=3, col=1)

    # fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vdeaths'],
    #                          name="Actual deaths", 
    #                          line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
    #             row=3, col=1)
                

    # fig.add_trace(go.Scatter(fill='tonexty', x=mean['date'], y=mean['deaths_cumsum']/SF_deaths,
    #                          name="Simulated deaths", 
    #                          line=dict({'width': 1, 'color': 'grey'})),
    #             row=4, col=1)

    # fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vdeaths_cumsum'],
    #                          name="Actual deaths", 
    #                          line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
    #             row=4, col=1)
    # for splitt pop and shakir logic
    SF_cases=50
    SF_admissions=60
    SF_deaths=1

    # SF_cases=0.1
    # SF_admissions=0.1
    # SF_deaths=1

    # fig.add_trace(go.Scatter(x=min['date'], y=min['cases']*SF_cases,fill='tonexty', 
    #                          name="min cases",
    #                          line=dict({'width': 1, 'color': 'darkorange'})),
    #             row=1, col=1)
    fig.add_trace(go.Scatter(x=mean['date'], y=mean['cases']*SF_cases,fill='tonexty', 
                             name="Simulated Cases", 
                             line=dict({'width': 1, 'color': 'orange'})),
                row=1, col=1)
    # fig.add_trace(go.Scatter(x=max['date'], y=max['cases']*SF_cases,fill='tonexty', 
    #                          name="max cases", 
    #                          line=dict({'width': 1, 'color': 'pink'})),
    #             row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=mean['date'], y=mean['vcases'],
                             name="Actual cases", 
                             line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
                row=1, col=1)

    # fig.add_trace(go.Scatter(x=min['date'], y=min['admissions']*SF_admissions,fill='tonexty', 
    #                          name="min admissions", 
    #                          line=dict({'width': 1, 'color': 'green'})),
    #             row=2, col=1)
    fig.add_trace(go.Scatter(x=mean['date'], y=mean['admissions']*SF_admissions,fill='tonexty', 
                             name="Simulated admissions", 
                             line=dict({'width': 1, 'color': 'darkgreen'})),
                row=2, col=1)
    # fig.add_trace(go.Scatter(x=max['date'], y=max['admissions']*SF_admissions,fill='tonexty', 
    #                          name="max admissions", 
    #                          line=dict({'width': 1, 'color': 'lightgreen'})),
    #             row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vadmissions'],
                             name="Actual admissions", 
                            line=dict({'width': 2, 'color': 'teal', 'dash':'dot'})),
                row=2, col=1)

    # fig.add_trace(go.Scatter(x=min['date'], y=min['deaths_cumsum']/SF_deaths,fill='tonexty', 
    #                          name="min deaths", 
    #                          line=dict({'width': 1, 'color': 'darkgrey'})),
    #             row=4, col=1)

    fig.add_trace(go.Scatter(x=mean['date'], y=mean['deaths'],fill='tonexty', 
                             name="Simulated deaths", 
                             line=dict({'width': 1, 'color': 'grey'})),
                row=3, col=1)
    # fig.add_trace(go.Scatter(x=max['date'], y=max['deaths_cumsum']/SF_deaths,fill='tonexty', 
    #                          name="max deaths", 
    #                          line=dict({'width': 1, 'color': 'lightgrey'})), 
    #             row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vdeaths'],
                             name="Actual deaths", 
                             line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
                row=3, col=1)
                

    # fig.add_trace(go.Scatter(x=mean['date'], y=mean['deaths_cumsum'], fill='tonexty', 
    #                          name="mean deaths", 
    #                          line=dict({'width': 1, 'color': 'grey'})),
    #             row=4, col=1)

    # fig.add_trace(go.Scatter(mode='lines',x=mean['date'], y=mean['vdeaths_cumsum'],
    #                          name="Actual Cumulative deaths", 
    #                          line=dict({'width': 2, 'color': 'black', 'dash':'dot'})),
    #             row=4, col=1)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1",tickangle=45,)
    #fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="4",tickangle=45,)

    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, autosize=True, 
                    font=dict(family="Arial", size=10),
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ]),
                        type="date"),#end xaxis  definition
                    
                    xaxis3_rangeslider_visible=True,
                    xaxis3_rangeslider_thickness = 0.04,
                    xaxis3_type="date"
    )
    #fig.show()
    return fig

# def plot_age2(df):
#     sub_groups = ['Cases by ages', 'Actual cases', #'Cases (>65)', 'Cases (18~65)', 'Cases (1~18)', 
#                 'Admissions by ages', 'Actual admissions',#'Admissions (>65)', 'Admissions (18~65)', 'Admissions (1~18)', 
#                 'Deaths by ages', 'Actual deaths',#'Deaths (>65)', 'Deaths (18~65)', 'Deaths (1~18)', 
#                 ]
#     fig = make_subplots(rows=6, cols=1, 
#         subplot_titles=sub_groups, 
#         shared_xaxes=True,
#         shared_yaxes=True, # use same y axis range
#         vertical_spacing=0.05,
#         horizontal_spacing=0.05,
#         row_width=[0.1, 0.4, 0.1, 0.4, 0.1, 0.4],
#         #column_width=[0.25, 0.25, 0.25]
#         )
#     df['cases_65'] = upper_envelope(df['cases_65'],7)
#     df['cases_18'] = upper_envelope(df['cases_18'],7)
#     df['cases_1'] = upper_envelope(df['cases_1'],7)
#     #df['vcases'] = upper_envelope(df['vcases'],7)
#     df['admissions_65'] = upper_envelope(df['admissions_65'],7)
#     df['admissions_18'] = upper_envelope(df['admissions_18'],7)
#     df['admissions_1'] = upper_envelope(df['admissions_1'],7)
#     #df['vadmissions'] = upper_envelope(df['vadmissions'],7)
#     df['deaths_65'] = upper_envelope(df['deaths_65'],7)
#     df['deaths_18'] = upper_envelope(df['deaths_18'],7)
#     df['deaths_1'] = upper_envelope(df['deaths_1'],7)
#     #df['vdeaths'] = upper_envelope(df['vdeaths'],7)

#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_65']/SF_cases, #line_shape="linear",#fill='tonexty', 
#                              name="cases >65", 
#                              line=dict(width=1, color='#ff4d4d')), 
#                 row=1, col=1)

#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_18']/SF_cases, #line_shape="linear",#fill='tonexty', 
#                              name="cases >18", 
#                              line=dict(width=1, color='#ff1a1a')), 
#                 row=1, col=1)

                
#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_1']/SF_cases, #line_shape="linear",#fill='tonexty', 
#                              name="cases >1", 
#                              line=dict(width=1, color='#e60000')), 
#                 row=1, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'], #line_shape="linear",
#                                 name="actual cases", 
#                              line=dict({'width': 0.5, 'color': 'black'})), 
#                 #secondary_y=True,
#                 row=2, col=1)

#     # -------------------------------------------------------------------------------------------------------------------
#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_65']/SF_admissions, #line_shape="linear",#fill='tonexty', 
#                              name="admissions >65", 
#                              line=dict(width=1, color='rgb(20, 170, 219)')), 
#                 row=3, col=1)


#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_18']/SF_admissions, #line_shape="linear",#fill='tonexty', 
#                              name="admissions >18", 
#                              line=dict(width=1, color='rgb(20, 120, 219)')), 
#                 row=3, col=1)


#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_1']/SF_admissions, #line_shape="linear",#fill='tonexty',
#                              name="admissions >1", 
#                              line=dict(width=1, color='rgb(20, 20, 219)')), 
#                 row=3, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions']/SF_admissions, #line_shape="linear",
#                              name="actual admissions", 
#                              line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True,
#                 row=4, col=1)

#     # -------------------------------------------------------------------------------------------------------------------
#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_65']/SF_deaths, #line_shape="linear",#fill='tonexty', 
#                              name="deaths >65", 
#                              line=dict(width=1, color='#00cc99')), 
#                 row=5, col=1)

#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_18']/SF_deaths, #line_shape="linear",#fill='tonexty', 
#                              name="deaths >18", 
#                              line=dict(width=1, color='#009973')), 
#                 row=5, col=1)

#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_1']/SF_deaths, #line_shape="linear",#fill='tonexty', 
#                              name="deaths >1", 
#                              line=dict(width=1, color='#008060')), 
#                 row=5, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'], #line_shape="linear",
#                              name="actual deaths", 
#                              line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True,
#                 row=6, col=1)

#     fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
#     fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#     fig.update_layout(showlegend=True, 
#                     autosize=True, 
#                     #width=1000, height=800,
#                     #legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
#                     #legend=dict(orientation="h"),
#                     font=dict(family="Arial", size=11),
#                     )
#     fig.update_xaxes(dtick="M3", tickformat="%b %Y")

#     return fig

def plot_age2_weekly(df):
    df['Date'] = pd.to_datetime(df['date'])
    df['Week_Number'] = df['Date'].dt.isocalendar().week
    df['Year']=(df['Date'] - df['Date'].dt.weekday * timedelta(days=1)).dt.year
    df.set_index('Date')

    df = df.groupby(['Year','Week_Number'], as_index=False).sum()

    df['year_week']=df['Year'].astype(str)+'-W'+df['Week_Number'].astype(str)
    df['year_week2']=df['Year'].astype(int)*100+df['Week_Number'].astype(int)
    df['date'] = pd.to_datetime(df['year_week2'].astype(str) + '0', format='%Y%W%w') # do not use: last week of the year become weird...

    sub_groups = ['Cases by ages', 'Actual cases', #'Cases (>65)', 'Cases (18~65)', 'Cases (1~18)', 
                'Admissions by ages', 'Actual admissions',#'Admissions (>65)', 'Admissions (18~65)', 'Admissions (1~18)', 
                'Deaths by ages', 'Actual deaths',#'Deaths (>65)', 'Deaths (18~65)', 'Deaths (1~18)', 
                ]
    fig = make_subplots(rows=6, cols=1, 
        subplot_titles=sub_groups, 
        shared_xaxes=True,
        shared_yaxes=True, # use same y axis range
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        row_width=[0.1, 0.2, 0.1, 0.4, 0.1, 0.4],
        #column_width=[0.25, 0.25, 0.25]
        )

    msize=4
    msym1='diamond'
    msym2='circle'
    msym3='square'

    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_65']/SF_cases, #line_shape="linear",#fill='tonexty', 
                             name="cases >65", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym1, #marker_color="black",
                             line=dict(width=1, color='#e60000')), 
                row=1, col=1)

    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_18']/SF_cases, #line_shape="linear",#fill='tonexty', 
                             name="cases >18", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym2, #marker_color="black",
                             line=dict(width=1, color='#ff4d4d')), 
                row=1, col=1)

                
    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_1']/SF_cases, #line_shape="linear",#fill='tonexty', 
                             name="cases >1", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym3, #marker_color="black",
                             line=dict(width=1, color='#ff1a1a')), 
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'], #line_shape="linear",
                                name="actual cases", 
                             line=dict({'width': 0.5, 'color': 'black'})), 
                #secondary_y=True,
                row=2, col=1)

    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_65']/SF_admissions, #line_shape="linear",#fill='tonexty', 
                             name="admissions >65", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym1, #marker_color="black",
                             line=dict(width=1, color='rgb(20, 20, 219)')), 
                row=3, col=1)


    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_18']/SF_admissions, #line_shape="linear",#fill='tonexty', 
                             name="admissions >18", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym2, #marker_color="black",
                             line=dict(width=1, color='rgb(20, 170, 219)')), 
                row=3, col=1)


    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_1']/SF_admissions, #line_shape="linear",#fill='tonexty',
                             name="admissions >1", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym3, #marker_color="black",
                             line=dict(width=1, color='rgb(20, 120, 219)')), 
                row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions']/SF_admissions, #line_shape="linear",
                             name="actual admissions", 
                             line=dict({'width': 0.5, 'color': 'black'})),
                #secondary_y=True,
                row=4, col=1)

    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_65']/SF_deaths, #line_shape="linear",#fill='tonexty', 
                             name="deaths >65", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym1, #marker_color="black",
                             line=dict(width=1, color='#00cc99')), 
                row=5, col=1)

    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_18']/SF_deaths, #line_shape="linear",#fill='tonexty', 
                             name="deaths >18", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym2, #marker_color="black",
                             line=dict(width=1, color='#008060')), 
                row=5, col=1)

    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_1']/SF_deaths, #line_shape="linear",#fill='tonexty', 
                             name="deaths >1", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym3, #marker_color="black",
                             line=dict(width=1, color='#009973')), 
                row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'], #line_shape="linear",
                             name="actual deaths", 
                             line=dict({'width': 0.5, 'color': 'black'})),
                #secondary_y=True,
                row=6, col=1)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, )
    fig.update_layout(showlegend=True, 
                    autosize=True, 
                    #width=1000, height=800,
                    #legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
                    #legend=dict(orientation="h"),
                    font=dict(family="Arial", size=11),
                    )
    fig.update_xaxes(dtick="M3", tickformat="%b %Y")

    return fig

# def plot_gender2(df):
#     sub_groups = ['Cases by gender', 'Actual cases',#'Cases (Males)', 'Cases (Females)', #
#                 'Admissions by gender', 'Actual admissions',#'Admissions (Males)', 'Admissions (Feales)', #
#                 'Deaths by gender', 'Actual deaths',#'Deaths (Males)', 'Deaths (Feales)', #
#                 ]

#     fig = make_subplots(rows=6, cols=1, 
#         subplot_titles=sub_groups, 
#         shared_xaxes=True,
#         shared_yaxes=True, # use same y axis range
#         vertical_spacing=0.05,
#         horizontal_spacing=0.05,
#         row_width=[0.1, 0.4, 0.1, 0.4, 0.1, 0.4],
#         #
#         # column_width=[0.25, 0.25]
#         )
#     df['cases_male'] = upper_envelope(df['cases_male'],7)
#     df['cases_female'] = upper_envelope(df['cases_female'], 7)
#     #df['vcases'] = upper_envelope(df['vcases'],7)
#     df['admissions_male'] = upper_envelope(df['admissions_male'], 7)
#     df['admissions_female'] = upper_envelope(df['admissions_female'],7)
#     #df['vadmissions'] = upper_envelope(df['vadmissions'],7)
#     df['deaths_male'] = upper_envelope(df['deaths_male'],7)
#     df['deaths_female'] = upper_envelope(df['deaths_female'],7)
#     #df['vdeaths'] = upper_envelope(df['vdeaths'],7)
    
#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_male']/SF_cases,
#                              name="cases Male", 
#                              line=dict(width=1, color='red')), 
#                 row=1, col=1)

#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_female']/SF_cases,
#                              name="cases Female", 
#                              line=dict(width=1, color='crimson')),
#                 row=1, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases']/SF_cases,
#                              name="actual cases", 
#                              line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True, 
#                 row=2,col=1)
#     # -------------------------------------------------------------------------------------------------------------------
#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_male']/SF_admissions, 
#                              name="admissions Male", 
#                              line=dict(width=1, color='green')), 
#                     row=3, col=1)

#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_female']/SF_admissions, 
#                              name="admissions Female", 
#                              line=dict(width=1, color='seagreen')), 
#                     row=3, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
#                              name="actual admissions", 
#                              line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True, 
#                 row=4,col=1)
#     # -------------------------------------------------------------------------------------------------------------------
#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_male']/SF_deaths, 
#                              name="deaths Male", 
#                              line=dict(width=1, color='slategrey')), 
#                     row=5, col=1)

#     fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_female']/SF_deaths, 
#                              name="deaths Female", 
#                              line=dict(width=1, color='slategrey')), 
#                     row=5, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
#                              name="actual deaths", 
#                              line=dict({'width': 0.5, 'color': 'black'})),
#                 #secondary_y=True, 
#                 row=6,col=1)
#     # -------------------------------------------------------------------------------------------------------------------
#     fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
#     fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#     fig.update_layout(showlegend=True, 
#                     autosize=True, 
#                     font=dict(family="Arial", size=12))
#     fig.update_xaxes(dtick="M2", tickformat="%b %Y")

#     return fig

def plot_gender2_weekly(df):

    df['Date'] = pd.to_datetime(df['date'])
    df['Week_Number'] = df['Date'].dt.isocalendar().week
    df['Year']=(df['Date'] - df['Date'].dt.weekday * timedelta(days=1)).dt.year
    df.set_index('Date')

    df = df.groupby(['Year','Week_Number'], as_index=False).sum()

    df['year_week']=df['Year'].astype(str)+'-W'+df['Week_Number'].astype(str)
    df['year_week2']=df['Year'].astype(int)*100+df['Week_Number'].astype(int)
    df['date'] = pd.to_datetime(df['year_week2'].astype(str) + '0', format='%Y%W%w') # do not use: last week of the year become weird...

    sub_groups = ['Cases by gender', 'Actual cases',#'Cases (Males)', 'Cases (Females)', #
                'Admissions by gender', 'Actual admissions',#'Admissions (Males)', 'Admissions (Feales)', #
                'Deaths by gender', 'Actual deaths',#'Deaths (Males)', 'Deaths (Feales)', #
                ]

    fig = make_subplots(rows=6, cols=1, 
        subplot_titles=sub_groups, 
        shared_xaxes=True,
        shared_yaxes=True, # use same y axis range
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        row_width=[0.1, 0.2, 0.1, 0.4, 0.1, 0.4],
        #
        # column_width=[0.25, 0.25]
        )

    msize=4
    msym1='diamond'
    msym2='circle'
  
    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_male']/SF_cases,
                             name="cases Male", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym1, #marker_color="black",
                             line=dict(width=1, color='red')), 
                row=1, col=1)

    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['cases_female']/SF_cases,
                             name="cases Female", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym2, #marker_color="black",
                             line=dict(width=1, color='pink')),
                row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases']/SF_cases,
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'black', 'dash':'dot'})),
                #secondary_y=True, 
                row=2,col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_male']/SF_admissions, 
                             name="admissions Male", 
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym1, #marker_color="black",
                             line=dict(width=1, color='green')), 
                    row=3, col=1)

    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['admissions_female']/SF_admissions, 
                             name="admissions Female", 
                             mode='lines',
                             marker_size=msize, marker_symbol=msym2, #marker_color="black",
                             line=dict(width=1, color='lightgreen')), 
                    row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                             line=dict({'width': 1, 'color': 'black', 'dash':'dot'})),
                #secondary_y=True, 
                row=4,col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_male']/SF_deaths, 
                             name="deaths Male",  
                             mode='lines', 
                             marker_size=msize, marker_symbol=msym1, #marker_color="black",
                             line=dict(width=1, color='grey')), 
                    row=5, col=1)

    fig.add_trace(go.Scatter(fill='tonexty', x=df['date'], y=df['deaths_female']/SF_deaths, 
                             name="deaths Female",  
                             mode='lines',
                             marker_size=msize, marker_symbol=msym2,  #marker_color="black",
                             line=dict(width=1, color='lightgrey')), 
                    row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", 
                             line=dict({'width': 1, 'color': 'black', 'dash':'dot'})),
                #secondary_y=True, 
                row=6,col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, 
                    autosize=True, 
                    font=dict(family="Arial", size=12))
    fig.update_xaxes(dtick="M2", tickformat="%b %Y")

    return fig

# def plot_race2(df):
#     sub_groups = ['Cases (Whites)', 'Admissions (Whites)','Deaths (Whites)',
#                   'Cases (Blacks)', 'Admissions (Blacks)','Deaths (Blacks)',
#                   'Cases (Asians)', 'Admissions (Asians)','Deaths (Asians)',
#                   'Cases (Other)',  'Admissions (Other)','Deaths (Two)',
#                   'Cases (Two)', 'Admissions (Two)','Deaths (Other)',
#                   'Actual cases', 'Actual admissions', 'Actual deaths'
#                 ]
#     fig = make_subplots(rows=6, cols=3, 
#         subplot_titles=sub_groups, 
#         shared_xaxes=True,
#         shared_yaxes='columns', # use same y axis range
#         vertical_spacing=0.05,
#         horizontal_spacing=0.05,
#         #column_width=[0.25, 0.25,0.25,0.25,0.25],
#         row_width=[0.1, 0.25, 0.25, 0.25, 0.25, 0.25,]
#         )
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_white']/SF_cases,
#                             name="cases (white)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': 'darkcyan'})), 
#                 row=1, col=1)


#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_black']/SF_cases,
#                              name="cases (black)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor2})), 
#                 row=2, col=1)


#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_asian']/SF_cases,
#                                 name="cases (asian)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor3})), 
#                 row=3, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_other']/SF_cases,
#                                  name="cases (other)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor4})), 
#                 row=4, col=1)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_two']/SF_cases,
#                              name="cases (two)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor5})), 
#                 row=5, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
#                              name="actual cases", 
#                              line=dict({'width': 0.5, 'color': 'black'})), 
#                 #secondary_y=True, 
#                 row=6, col=1)
#     # -------------------------------------------------------------------------------------------------------------------
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_white']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (white)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': 'darkcyan'})), 
#                         row=1, col=2)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_black']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (black)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor2})), 
#                         row=2, col=2)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_asian']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (asian)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor3})), 
#                         row=3, col=2)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_other']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (other)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor4})), 
#                         row=4, col=2)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_two']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (two)",  
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor5})), 
#                          row=5, col=2)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
#                              name="actual admissions", 
#                              line=dict({'width': 0.5, 'color':'black'})),
#                         #secondary_y=True, 
#                         row=6, col=2)
#     # -------------------------------------------------------------------------------------------------------------------
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_white']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (white)", 
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': 'darkcyan'})), 
#                     row=1, col=3)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_black']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (black)", 
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor2})), 
#                     row=2, col=3)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_asian']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (asian)", 
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor3})), 
#                     row=3, col=3)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_other']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (other)", 
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor4})), 
#                     row=4, col=3)

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_two']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (two)", 
#                             fill='tozeroy',
#                              line=dict({'width': 1, 'color': fillcolor5})), 
#                     row=5, col=3)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
#                              name="actual deaths", 
#                              line=dict({'width': 0.5, 'color': 'black'})), 
#                     #secondary_y=True, 
#                     row=6,col=3)

#     fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
#     fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#     fig.update_layout(showlegend=True, 
#                     autosize=True, 
#                     #width=1000, height=1200,
#                     #legend=dict(orientation="h",x=0, y=-0.16, traceorder="normal"),
#                     #legend=dict(orientation="h"),
#                     font=dict(family="Arial", size=11))
#     fig.update_xaxes(dtick="M3", tickformat="%b %Y")

#     return fig

def plot_race2_weekly(df):
    df['Date'] = pd.to_datetime(df['date'])
    df['Week_Number'] = df['Date'].dt.isocalendar().week
    df['Year']=(df['Date'] - df['Date'].dt.weekday * timedelta(days=1)).dt.year
    df.set_index('Date')

    df = df.groupby(['Year','Week_Number'], as_index=False).sum()

    df['year_week']=df['Year'].astype(str)+'-W'+df['Week_Number'].astype(str)
    df['year_week2']=df['Year'].astype(int)*100+df['Week_Number'].astype(int)
    df['date'] = pd.to_datetime(df['year_week2'].astype(str) + '0', format='%Y%W%w') # do not use: last week of the year become weird...

    sub_groups = ['Cases (Whites)', 'Admissions (Whites)','Deaths (Whites)',
                  'Cases (Blacks)', 'Admissions (Blacks)','Deaths (Blacks)',
                  'Cases (Asians)', 'Admissions (Asians)','Deaths (Asians)',
                  'Cases (Other)',  'Admissions (Other)','Deaths (Two)',
                  'Cases (Two)', 'Admissions (Two)','Deaths (Other)',
                  'Actual cases', 'Actual admissions', 'Actual deaths'
                ]
    fig = make_subplots(rows=6, cols=3, 
        subplot_titles=sub_groups, 
        shared_xaxes=True,
        shared_yaxes='columns', # use same y axis range
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        #column_width=[0.25, 0.25,0.25,0.25,0.25],
        row_width=[0.1, 0.25, 0.25, 0.25, 0.25, 0.25,]
        )
    msize=4
    msym1='diamond'
    msym2='circle'
    msym3='square'

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_white']/SF_cases,
                            name="cases (white)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': 'darkcyan'})), 
                row=1, col=1)


    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_black']/SF_cases,
                             name="cases (black)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor2})), 
                row=2, col=1)


    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_asian']/SF_cases,
                                name="cases (asian)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor3})), 
                row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_other']/SF_cases,
                                 name="cases (other)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor4})), 
                row=4, col=1)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_two']/SF_cases,
                             name="cases (two)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor5})), 
                row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 0.5, 'color': 'black'})), 
                #secondary_y=True, 
                row=6, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_white']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (white)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': 'darkcyan'})), 
                        row=1, col=2)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_black']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (black)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor2})), 
                        row=2, col=2)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_asian']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (asian)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor3})), 
                        row=3, col=2)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_other']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (other)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor4})), 
                        row=4, col=2)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_two']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (two)",  
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor5})), 
                         row=5, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                             line=dict({'width': 0.5, 'color':'black'})),
                        #secondary_y=True, 
                        row=6, col=2)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_white']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (white)", 
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': 'darkcyan'})), 
                    row=1, col=3)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_black']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (black)", 
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor2})), 
                    row=2, col=3)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_asian']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (asian)", 
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor3})), 
                    row=3, col=3)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_other']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (other)", 
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor4})), 
                    row=4, col=3)

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_two']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (two)", 
                            fill='tozeroy',
                             line=dict({'width': 1, 'color': fillcolor5})), 
                    row=5, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'],
                             name="actual deaths", 
                             line=dict({'width': 0.5, 'color': 'black'})), 
                    #secondary_y=True, 
                    row=6,col=3)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, 
                    autosize=True, 
                    #width=1000, height=1200,
                    #legend=dict(orientation="h",x=0, y=-0.16, traceorder="normal"),
                    #legend=dict(orientation="h"),
                    font=dict(family="Arial", size=11))
    fig.update_xaxes(dtick="M3", tickformat="%b %Y")

    return fig

# def plot_FPL2(df):
#     sub_groups = ['Cases (0-100)', 'Admissions(00-100)', 'Deaths (00-100)',
#                 'Cases (100-150)', 'Admissions(100-150)', 'Deaths (100-150)', 
#                 'Cases (150-175)','Admissions(150-175)',  'Deaths (150-175)',
#                 'Cases (175-200)','Admissions(175-200)','Deaths (175-200)',
#                 'Cases (200-1800)','Admissions(200-1800)', 'Deaths (200-1800)',
#                 'Actual cases', 'Actual admissions', 'Actual deaths',
#                 ]
#     fig = make_subplots(rows=6, cols=3, 
#         subplot_titles=sub_groups, 
#         shared_xaxes=True,
#         shared_yaxes='columns', # use same y axis range
#         vertical_spacing=0.05,
#         horizontal_spacing=0.05,
#        # column_width=[0.25, 0.25,0.25,0.25,0.25],
#         row_width=[0.1, 0.25, 0.25, 0.25, 0.25, 0.25,]
#     )

#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_0-100']/SF_cases,
#                             #stackgroup='one', groupnorm='percent',
#                              name="Cases (0-100)", 
#                             fill='tozeroy',
#                              line=dict({'width':0.5, 'color':'darkcyan'})), 
#                     row=1, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_100-150']/SF_cases,
#                             #stackgroup='one', groupnorm='percent',
#                              name="cases (100-150)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor2})), 
#                     row=2, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_150-175']/SF_cases,
#                             #stackgroup='one', groupnorm='percent',
#                              name="cases (150-175)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor3})), 
#                     row=3, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_175-200']/SF_cases,
#                             #stackgroup='one', groupnorm='percent',
#                              name="cases (175-200)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor4})), 
#                     row=4, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_200-1800']/SF_cases,
#                             #stackgroup='one', groupnorm='percent',
#                              name="cases (200-1800)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor5})), 
#                     row=5, col=1)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
#                              name="actual cases", 
#                              line=dict({'width': 1, 'color': 'black'})), 
#                     #secondary_y=True, 
#                     row=6, col=1)
#     #-------------------------------------------------------------------------------------------------------------------
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_0-100']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (0-100)", 
#                             fill='tozeroy',
#                              line=dict({'width':0.5, 'color':'darkcyan'})), 
#                     row=1, col=2)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_100-150']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (100-150)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor2})), 
#                     row=2, col=2)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_150-175']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (150-175)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor3})), 
#                     row=3, col=2)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_175-200']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (175-200)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor4})), 
#                     row=4, col=2)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_200-1800']/SF_admissions,
#                             #stackgroup='two', groupnorm='percent',
#                              name="admissions (200-1800)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor5})),
#                     row=5, col=2)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
#                              name="actual admissions", 
#                              line=dict({'width': 1, 'color': 'black'})), 
#                     #secondary_y=True, 
#                     row=6, col=2)

#     # -------------------------------------------------------------------------------------------------------------------
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_0-100']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (0-100)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':'darkcyan'})), 
#                     row=1, col=3)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_100-150']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (100-150)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor2})), 
#                     row=2, col=3)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_150-175']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (150-175)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor3})),
#                     row=3, col=3)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_175-200']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (175-200)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor4})), 
#                     row=4, col=3)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_200-1800']/SF_deaths,
#                             #stackgroup='three', groupnorm='percent',
#                              name="deaths (200-1800)", 
#                             fill='tozeroy',
#                              line=dict({'width': 0.5, 'color':fillcolor5})), 
#                     row=5, col=3)
#     fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'], 
#                              name="actual deaths", 
#                              line=dict({'width': 1, 'color': 'black'})), 
#                     #secondary_y=True, 
#                     row=6, col=3)

#     # -------------------------------------------------------------------------------------------------------------------

#     fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
#     fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#     fig.update_layout(showlegend=True, 
#                     autosize=True, 
#                     #width=1000, height=800,
#                     legend=dict(orientation="h",x=0, y=-0.15, traceorder="normal"),
#                     #legend=dict(orientation="h"),
#                     font=dict(family="Arial", size=11))
#     fig.update_xaxes(dtick="M2", tickformat="%b %Y")

#     return fig

def plot_FPL2_weekly(df):
    df['Date'] = pd.to_datetime(df['date'])
    df['Week_Number'] = df['Date'].dt.isocalendar().week
    df['Year']=(df['Date'] - df['Date'].dt.weekday * timedelta(days=1)).dt.year
    df.set_index('Date')

    df = df.groupby(['Year','Week_Number'], as_index=False).sum()

    df['year_week']=df['Year'].astype(str)+'-W'+df['Week_Number'].astype(str)
    df['year_week2']=df['Year'].astype(int)*100+df['Week_Number'].astype(int)
    df['date'] = pd.to_datetime(df['year_week2'].astype(str) + '0', format='%Y%W%w') # do not use: last week of the year become weird...

    sub_groups = ['Cases (0-100)', 'Admissions(00-100)', 'Deaths (00-100)',
                'Cases (100-150)', 'Admissions(100-150)', 'Deaths (100-150)', 
                'Cases (150-175)','Admissions(150-175)',  'Deaths (150-175)',
                'Cases (175-200)','Admissions(175-200)','Deaths (175-200)',
                'Cases (200-1800)','Admissions(200-1800)', 'Deaths (200-1800)',
                'Actual cases', 'Actual admissions', 'Actual deaths',
                ]
    fig = make_subplots(rows=6, cols=3, 
        subplot_titles=sub_groups, 
        shared_xaxes=True,
        shared_yaxes='columns', # use same y axis range
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
       # column_width=[0.25, 0.25,0.25,0.25,0.25],
        row_width=[0.1, 0.25, 0.25, 0.25, 0.25, 0.25,]
    )

    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_0-100']/SF_cases,
                            #stackgroup='one', groupnorm='percent',
                             name="Cases (0-100)", 
                            fill='tozeroy',
                             line=dict({'width':0.5, 'color':'darkcyan'})), 
                    row=1, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_100-150']/SF_cases,
                            #stackgroup='one', groupnorm='percent',
                             name="cases (100-150)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor2})), 
                    row=2, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_150-175']/SF_cases,
                            #stackgroup='one', groupnorm='percent',
                             name="cases (150-175)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor3})), 
                    row=3, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_175-200']/SF_cases,
                            #stackgroup='one', groupnorm='percent',
                             name="cases (175-200)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor4})), 
                    row=4, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['cases_200-1800']/SF_cases,
                            #stackgroup='one', groupnorm='percent',
                             name="cases (200-1800)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor5})), 
                    row=5, col=1)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vcases'],
                             name="actual cases", 
                             line=dict({'width': 1, 'color': 'black'})), 
                    #secondary_y=True, 
                    row=6, col=1)
    #-------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_0-100']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (0-100)", 
                            fill='tozeroy',
                             line=dict({'width':0.5, 'color':'darkcyan'})), 
                    row=1, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_100-150']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (100-150)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor2})), 
                    row=2, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_150-175']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (150-175)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor3})), 
                    row=3, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_175-200']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (175-200)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor4})), 
                    row=4, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['admissions_200-1800']/SF_admissions,
                            #stackgroup='two', groupnorm='percent',
                             name="admissions (200-1800)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor5})),
                    row=5, col=2)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vadmissions'],
                             name="actual admissions", 
                             line=dict({'width': 1, 'color': 'black'})), 
                    #secondary_y=True, 
                    row=6, col=2)

    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_0-100']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (0-100)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':'darkcyan'})), 
                    row=1, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_100-150']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (100-150)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor2})), 
                    row=2, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_150-175']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (150-175)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor3})),
                    row=3, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_175-200']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (175-200)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor4})), 
                    row=4, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['deaths_200-1800']/SF_deaths,
                            #stackgroup='three', groupnorm='percent',
                             name="deaths (200-1800)", 
                            fill='tozeroy',
                             line=dict({'width': 0.5, 'color':fillcolor5})), 
                    row=5, col=3)
    fig.add_trace(go.Scatter(mode='lines', x=df['date'], y=df['vdeaths'], 
                             name="actual deaths", 
                             line=dict({'width': 1, 'color': 'black'})), 
                    #secondary_y=True, 
                    row=6, col=3)

    # -------------------------------------------------------------------------------------------------------------------

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticklabelmode="period", dtick="M1")
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(showlegend=True, 
                    autosize=True, 
                    #width=1000, height=800,
                    legend=dict(orientation="h",x=0, y=-0.15, traceorder="normal"),
                    #legend=dict(orientation="h"),
                    font=dict(family="Arial", size=11))
    fig.update_xaxes(dtick="M2", tickformat="%b %Y")

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
                    dlist.append(d)

        print('All plots-Loading completed!')
        plotdf = pd.concat(dlist, axis=1)
        plotdf.drop('date', axis=1, inplace=True)

        dates = dlist[0]['date'].tolist()
        #####plotdf.drop('date', axis=1, inplace=True)

        max = plotdf.groupby(plotdf.columns, axis=1).max()
        min = plotdf.groupby(plotdf.columns, axis=1).min()
        
        #mean = plotdf.groupby(plotdf.columns, axis=1).mean() # error! return 0s ... use transpose() for fix
        
        df2=plotdf.transpose()
        df2 = df2.groupby(by=df2.index, axis=0).apply(lambda g: g.mean() if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[0])
        mean = df2.transpose()
        
        # mean= plotdf.groupby(plotdf.columns, axis=1).sum() # as of 3/31, use sum only

        max['date'] = dates
        min['date'] = dates
        mean['date'] = dates

        #return plot2(min, mean, max)
        #return plot3(min, mean, max)
        return plot3_weekly(min,mean,max)
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
            #return plot_age(plotdf)
            return plot_age2_weekly(plotdf)
        elif mode == 'By Gender':
            #return plot_gender(plotdf)
            #return plot_gender2(plotdf)
            return plot_gender2_weekly(plotdf)
        elif mode == 'By Race':
            #return plot_race(plotdf)
            return plot_race2_weekly(plotdf)
        elif mode == 'By Federal Poverty Level':
            #return plot_FPL(plotdf)
            return plot_FPL2_weekly(plotdf)

def load_scatter(zipcode, year, sampling, width, height, show_whole_county):
    #return load_scatter_mongodb(zipcode, year)
    return load_scatter_read_parquet(zipcode, year, sampling, width, height, show_whole_county)

def load_heatmap(zipcode,year, sampling, width, height, show_whole_county):
    #return load_heatmap_mongodb(zipcode, year)
    #return load_heatmap_read_parquet(zipcode,year, sampling, width, height, show_whole_county)
    #return draw_heatmap_weekly(width, height)
    return draw_heatmap_bubble(width, height)

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

    #print('scatter memory usage', pdf.info())
    print('scatter memory size(MB)', sys.getsizeof(pdf)/(1024*1024))
    pdf = pdf.sort_values(by='step') # should be run after sampling
    pdf.drop('step',axis=1, inplace=True)

    return draw_scatter(pdf, zipcode, width, height, show_whole_county)

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
                            animation_frame='Date',

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

    fig.update_layout(legend=dict(
        orientation="h",
        xanchor="right",
        yanchor="bottom",
        y=1.02,
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


"""
load_heatmap using_read_parquet
"""
def load_heatmap_read_parquet(zipcode=default_zipcode, year=default_year, sampling=default_sampling, width=graph_width, height=graph_height, show_whole_county=False):

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
 
    print(zipcode, year, min_step, max_step, "show_whole_county=", show_whole_county)
    
    columns_being_used_in_heatmap=[
        'step',
        'x',
        'y',
        'z',
        #'zip',
    ]

    t1=datetime.now()

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

    print('heatmap memory usage(MB)', sys.getsizeof(pdf)/(1024*1024))
    pdf = pdf.sort_values(by='step')
    pdf.drop('step',axis=1, inplace=True)

    return draw_heatmap(pdf, zipcode, width, height, show_whole_county)

def draw_heatmap(pdf, zipcode, width, height, show_whole_county):
    global heatmap_size
    heatmap_size=pdf.size   
    print("heatmap size in draw=", heatmap_size)
    center_lat = 28.03711
    center_lon = -82.46390
    zoom_level=8
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

def draw_heatmap_weekly(width, height):
    riskzips_df2=pd.read_parquet("../risky_zipcodes2.parquet")
    riskzips_df2['date_str']=riskzips_df2['date'].dt.strftime('%Y-%m-%d')
    riskzips_df2['y']=riskzips_df2.apply(
        lambda x: ZIPS_centers[x.zipcode][0], axis=1
    )

    riskzips_df2['x']=riskzips_df2.apply(
        lambda x: ZIPS_centers[x.zipcode][1], axis=1
    )
    center_lat = 28.03711
    center_lon = -82.46390
    fig = px.density_mapbox(riskzips_df2,
                            color_continuous_scale='RdYlGn_r',
                            lat=riskzips_df2['y'],
                            lon=riskzips_df2['x'],
                            z=riskzips_df2['risk'],
                            #animation_frame=pdf['step'],
                            animation_frame='date_str',
                            opacity=0.75,
                            zoom=8,
                            width=width,
                            height=height,
                            center=dict(lat=center_lat, lon=center_lon),
                            # mapbox_style='open-street-map'
                            mapbox_style='stamen-terrain')
    return fig

def draw_heatmap_bubble(width, height):
    px.set_mapbox_access_token("pk.eyJ1Ijoia2ltc29vaWwiLCJhIjoiY2wxa3Byd3A0MDI3dTNibzg0czF3dHd3aCJ9.hrsnzqpk-4MtEvfh_DZRdg")

    riskzips_df2=pd.read_parquet("../risky_zipcodes2.parquet")
    riskzips_df2['date_str']=riskzips_df2['date'].dt.strftime('%Y-%m-%d')
    riskzips_df2['y']=riskzips_df2.apply(
        lambda x: ZIPS_centers[x.zipcode][0], axis=1
    )

    riskzips_df2['x']=riskzips_df2.apply(
        lambda x: ZIPS_centers[x.zipcode][1], axis=1
    )
    center_lat = 28.03711
    center_lon = -82.46390

    rmax=riskzips_df2['risk'].max()
    riskzips_df2['color'] = (riskzips_df2['risk']/rmax).fillna(0).replace(np.inf , 0)
    # Create the figure and feed it all the prepared columns
    fig = px.scatter_mapbox(riskzips_df2, lat="y", lon="x", color="risk", size=riskzips_df2['risk'],
                    animation_frame = 'date_str',
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=50, zoom=9)
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

def draw_risky_zipcodes():
    dfm2=gpd.read_file("hillsborough-zipcodes-boundarymap.geojson")

    riskzips_df2=pd.read_parquet("../risky_zipcodes2.parquet")
    riskzips_df2['date_str']=riskzips_df2['date'].dt.strftime('%Y-%m-%d')
    fig = px.choropleth_mapbox(riskzips_df2, 
                            geojson=dfm2, locations='zipcode', 
                            color='risk', 
                            featureidkey="properties.zipcode",
                            color_continuous_scale="Viridis_r",
                            #range_color=(0, 12),
                            #mapbox_style="carto-positron",
                            #mapbox_style='white-bg',
                            mapbox_style="open-street-map",
                            zoom=8, center = {"lat": 27.91, "lon": -82.4},
                            opacity=0.5,
                            #labels={'zip_area':'random'}
                            #animation_frame='step',
                            #animation_frame='year_week2',
                            animation_frame='date_str',
                            width=graph_width,
                            height=graph_height
                            )     
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

"""
main plotly dash start here.
"""

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
                        'margin-left':'10px',
                        'width':'20%',
                        'text-align':'center',
                        'display':'inline-block'
                        },
                className="nav",
                children=[
                    html.H4("Options for social distancing:", className="control_label", style={'padding': 10, 'flex': 1}),
                    dcc.RadioItems(
                        id="social_distaincing",
                        options=[{'label': i, 'value': i} for i in ['No change in social distancing',
                                                                    'Social distancing (+25%)',
                                                                    'Social distancing (-25%)']
                        ],
                        value="No change",
                        labelStyle={'display': 'block', 'text-align': 'left', 'margin-right': 20},
                        #labelStyle = {'display': 'inline-block', 'margin-right': 10},
                        style={'padding': 10, 'flex': 1}
                    ),
                    html.H4("Vaccination scenarios:", className="control_label", style={'padding': 10, 'flex': 1}),
                    dcc.RadioItems(
                        id="vaccination",
                        options=[{'label': i, 'value': i} for i in ['No change in vaccination',
                                                                    'Vaccination (+5%)',
                                                                    'Vaccination (-5%)']
                        ],
                        value="No intervention",
                        labelStyle={'display': 'block', 'text-align': 'left', 'margin-right': 20},
                        #labelStyle = {'display': 'inline-block', 'margin-right': 10},
                        style={'padding': 10, 'flex': 1}
                    ),
                    #html.P("Interventions: Lockdown(+-25%), Vaccination(+-10%)"),
                    html.Br(),
                    html.Br(),
                    html.A("Contact Info.", href='https://health.usf.edu/publichealth/overviewcoph/faculty/edwin-michael', target="_blank"),
                    html.Br(),
                    html.Img(src=app.get_asset_url('usf-logo-white-bg.jfif'), style={'margin-left': 10, 'width':'200px'}),

                ]
            ),
            html.Div(
                style={
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
                                dcc.Tab(label='All zip codes plots', value='tab1_1', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='Bubble Map of infections', value='tab3', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='Risk by zip codes', value='tab5', style = tab_style, selected_style = tab_selected_style),
                                dcc.Tab(label='Spatial Spread', value='tab2', style = tab_style, selected_style = tab_selected_style),
                                #dcc.Tab(label='Spatial Spread & Heatmap (Hillsborough County)', value='tab4', style = tab_style, selected_style = tab_selected_style),
                            ], style = tabs_styles),
                            html.Div(
                                id='tabs-contentgraph'),
                                html.Div(
                                    className="footer",
                                    children=[
                                        html.P('Paper: "An Agent-based City Scale Digital Twin (EDEN) for Pandemic Analytics and Scenario Planning, Imran Mahmood et al. (in publishing)"'),
                                        html.H3(children='* Simulation results are provided by Dr. Edwin Michael Lab, USF College of Public Health *'),
                                        html.H4('Team members: Edwin Michael (PI), Imran Mahmood, Yilian Alonso Otano, Soo I. Kim, Shakir Bilal'),
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
                                                            'By Federal Poverty Level']],

                value="All cases",
                labelStyle={'display': 'inline-block'}
            ),
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
                        "width": "100%",
                        "height": "100%",
                    }
                )],
                style={
                        "width": "900px",
                        "height": "900px",
                        "display": "inline-block",
                        "overflow": "hidden"
                }
            )
        ])
    elif tab == 'tab1_1':
        #fig_cases, fig_admissions, fig_deaths=load_allzipcodes()
        fig=load_allzipcodes()
        return html.Div([
            html.Br(),
            html.H2("All zip codes"),
            #html.Img(src=app.get_asset_url('cases-allzip.png'), style={'margin-left': 10, 'width':'600px'}),
            html.Div(children=[
                dcc.Graph(
                    id='graph11',
                    #figure=figure1,
                    figure=fig,
                    config={
                        'displayModeBar': False
                    },
                    responsive=True,
                    style={
                        "width": "100%",
                        "height": "100%",
                    }
                )],
                style={
                        "width": "900px",
                        "height": "900px",
                        "display": "inline-block",
                        "overflow": "hidden"
                }
            ),
            html.Br(),
            # html.H2("Admissions (all zip codes)"),
            # html.Img(src=app.get_asset_url('admissions-allzip.png'), style={'margin-left': 10, 'width':'600px'}),            
            # html.Br(),
            # html.H2("Deaths (all zip codes)"),
            # html.Img(src=app.get_asset_url('deaths-allzip.png'), style={'margin-left': 10, 'width':'600px'}),            
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
            draw_legend_table(),   
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
            # html.Div(children=[
            #     html.A("More info of ZIP="+zipcode_for_all, href='https://www.unitedstateszipcodes.org/'+zipcode_for_all+'/', target="_blank"),
            # ]),

        ])        
    elif tab == 'tab3':
        fig=load_heatmap(zipcode_for_all, year_for_all, sampling_for_all, graph_width, graph_height, show_whole_county=False)
        return html.Div(id="tab3", children=[
            html.Br(),
            html.H2("Bubble Map of Weekly infections in zip codes"),
            # html.H2("Heatmap of the density of daily infectious cases"),
            #html.P("(Z values represents the number of cases within the same zipcode area.)"),
            #html.P("(Steps equals Days starting March 1, 2020)"),
            # html.P(heat_map_explain),
            # html.Div(children=[
            #     html.H4("Year:", className="control_label", style={'display': 'inline-block'}),
            #     dcc.Dropdown(
            #         id="year_heatmap",
            #         options=[{'label': i, 'value': i} for i in ['2020','2021']],
            #         #value=default_year,
            #         value=year_for_all,
            #        style={'width':'100px', 'display':'inline-block', 'verticalAlign':'middle'}
            #     ),
            #     html.H4(", Zip Code: ", className="control_label", style={'display': 'inline-block'}),
            #     dcc.Dropdown(
            #         id="zipcode_heatmap",
            #         options=[{'label': i, 'value': i} for i in ZIPS],
            #         #value=default_zipcode,
            #         value=zipcode_for_all,
            #         style={'width':'120px', 'display':'inline-block', 'verticalAlign':'middle'}
            #     ),
            #     html.H4("Sampling rate: ", className="control_label", style={'display': 'inline-block'}),
            #     dcc.Dropdown(
            #         id="sampling_heatmap",
            #         options=[{'label': '10 %', 'value':0.1},
            #                 {'label': '25 %', 'value':0.25},
            #                 {'label': '50 %', 'value': 0.5},
            #                 {'label': '75 %', 'value':0.75},
            #                 {'label': "100 %", 'value':1.0}],
            #         #value=0.25,
            #         value=sampling_for_all,
            #         style={'width':'100px', 'display':'inline-block', 'verticalAlign':'middle'}
            #     ),
            # ], style={'width': '100%', 'display': 'inline-block'}),
            # #html.P("Note: For the fast web response, only a fraction of data is being used here. This page uses "+str(SAMPLING_PERCENT_2*100)+" %", style={'textAlign': 'center', 'color':'orange'}),
            # html.Div(id="heatmap_size_num", children=[
            #     html.P("(Data size="+str(heatmap_size)+"), Sampling rate="+str(sampling_for_all), style={'textAlign': 'center', 'color':'orange'}),
            # ]),
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
            # html.Div(children=[
            #     html.A("More info of ZIP="+zipcode_for_all, href='https://www.unitedstateszipcodes.org/'+zipcode_for_all+'/', target="_blank"),
            # ]),

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
               html.H4(", Sampling rate: ", className="control_label", style={'display': 'inline-block'}),
                dcc.Dropdown(
                    id="sampling_for_all",
                    options=[{'label': '1 %', 'value':0.01},
                            {'label': '5 %', 'value':0.05},
                            {'label': '10 %', 'value': 0.1},
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
    elif tab == 'tab5':
        fig=draw_risky_zipcodes()
        return  html.Div(children=[
                    html.Br(),
                    html.H2("Risk by zip codes"),
                    html.Div(
                        dcc.Graph(
                            id='graph5',
                            figure=fig,
                            config={
                                'displayModeBar': False
                            }                
                        ), style={'display': 'inline-block'})
                ], style={'width': '100%', 'display': 'inline-block'})        


@app.callback(Output("graph1", 'figure'), Input("filter_type", "value"),
            prevent_initial_call=True)
@cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
def update_SEIR(filter_type1):
    global filter_type
    filter_type=filter_type1
    return load_SEIR(filter_type1)
    #return load_SEIR_by_intervention(filter_type, "No intervention")

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
