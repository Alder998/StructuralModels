import pandas as pd

import processes as pr
import GraphicalInterface as gr
import MarketDataScraper as mkt
import time
import dash
import numpy as np
from dash import dcc
from dash import html
from dash import Input
from dash import Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import MertonModel as mm
import matplotlib.pyplot as plt
import processes as pr
from plotly.subplots import make_subplots
import JumpDiffusionModelClosedForm as jp
import montecarloJumpDiffisionDaO as mdao
import statsmodels.api as sm
import yfinance as yf
import modelComparison as model
import DownAndOutCall as dao
import MarketDataScraper as mkt

from sklearn.preprocessing import PolynomialFeatures
import time

# Creiamo la Dashboard

# Andiamo a definire la nostra app

app = dash.Dash(__name__)

# definiamo il layout

app.layout = html.Div([

    # inseriamo il Dropdown come elemento grafico

    dcc.Input(
        placeholder='Enter a value...',
        type='text',
        value='',
        id='Ticker',
        loading_state = {'is_loading' : True, 'component_name' : 'Ticker'},
        style={'height': '100%', 'width': '100%', 'display': 'flex', 'justifyContent' : 'center'}
    ),

    html.Br(),

    dcc.RadioItems(['quarterly', 'yearly'], 'yearly', id = 'freq', style={'height': '100%', 'width': '100%', 'display': 'flex', 'justifyContent' : 'center'}),

    dcc.Graph(id='PD_Behavior'),
           ], style={'height': '100%', 'width': '100%', 'display': 'inline-block', 'justifyContent' : 'center'})

# Ora bisogna inserire gli elementi che ci faranno interagire con il nostro grafico, quindi mettiamo le mani sulle callbaks

@app.callback(

    # Definiamo Input e Output, cioè cosa comunica con cosa
    #Output("loading-output-1", 'children'),
    Output('PD_Behavior', 'figure'),
    Input('freq', 'value'),
    Input('Ticker', 'value'))

def update_graph(freq, Ticker):

    # definiamo il filtro adatto, per entrambi i grafici che abbiamo in Output
    dfM = gr.MertonProbabilityOfDefaultRealStocksForGraphicalInterface(Ticker, 1, freq)
    dfJD = gr.JDProbabilityOfDefaultRealStocksForGraphicalInterface(Ticker, 1, freq)
    dfDao = gr.MertonDownAndOutRealStocksForGraphicalInterface(Ticker, 1, freq)
    final = pd.concat([dfM, dfJD, dfDao], axis=1).set_axis(['Merton', 'Jump-Diffusion', 'Down-And-Out'], axis = 1)
    TS = yf.Ticker(Ticker).history(start = pd.to_datetime(final.index)[0], end = pd.to_datetime(final.index)[len(final['Merton'])-1])

    # Grafico

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(y = final['Merton'], x= final.index, name="Merton Model"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(name="Jump-Diffusion Model", y = final['Jump-Diffusion'], x= final.index),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(y = final['Down-And-Out'], x= final.index, name="Down-And-Out"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(name="Time Series", y=(TS['Close'].pct_change() * 100).dropna().cumsum(), x=TS.index, mode='lines'),
        secondary_y=True,
    )

    fig.update_layout(margin=dict(t=60, b=0, l=10, r=0))
    fig.update_layout(template='plotly_white', title = Ticker + ' ' + freq + ' PD Evolution')
    fig.update_layout(transition_duration=1000)
    fig.update_xaxes(
        title_text="Time",
        title_standoff=25)

    fig.update_yaxes(
        title_text="PD (%)", secondary_y=False)

    fig.update_yaxes(
        title_text="Stock % Return", secondary_y=True)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 8060)

