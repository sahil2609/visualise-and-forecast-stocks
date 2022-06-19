import dash
from dash import dcc
from dash import html
from datetime import datetime as dt
import yfinance as yf
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
# model
from model import prediction
from sklearn.svm import SVR

def get_stock_price_fig(df):
    fig = px.line(df, x = "Date", y = ["Close", "Open"], title="Closing and Opening Price vs Date")
    return fig


#Exponential weighted functions in Pandas
#The ewm() function is used to provide exponential weighted functions.
def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span = 20, adjust = False).mean()
    fig = px.scatter(df, x = "Date",y = "EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode = 'lines+markers')
    return fig

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Roboto&display=swap"
    ])
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.P(
                    "Welcome to StockDash App!",
                    className="start"
                ),
                html.Div(
                    [
                        #stock code input
                        html.P("Input stock code: "),
                        html.Div([
                            dcc.Input(
                                id = "stock-code",
                                placeholder='Enter a value...',
                                type='text'
                            ),
                            html.Button("Submit", id='submit')
                        ], className="stock-code-form"
                        ),
                        
                    ], className="input"
                ),
                html.Div(
                    [
                        #Date range picker input
                        dcc.DatePickerRange(
                            id='date-picker-range',
                            min_date_allowed=dt(1995, 8, 5),
                            max_date_allowed=dt.now(),
                            initial_visible_month=dt.now(),
                            end_date=dt.now().date()
                            
                    )
                    ], className="date"
                ),
                html.Div(
                    [
                        # Stock price button
                        html.Button("Stock Price", className="stock-btn", id = "stock"),
                        # Indicators button
                        html.Button("Indicators", className="indicator-btn", id = "indicator"),
                        # Number of days of forecast input
                        dcc.Input(
                            id = "n-days",
                            placeholder='Number of days',
                            type='text'
                        ),
                        # Forecast button
                        html.Button("Forecast Button", className="forecast-btn", id="forecast")
                    ], className="buttons"
                )
            ], className="navbar"
        ),
        #content
        html.Div(
            [
                html.Div(
                    [
                        #Logo
                        html.Img(id = "logo"),
                        #Company name
                        html.P(id = "ticker")
                    ], className="header"
                ),
                html.Div(
                    #Description
                    id = "description",
                    className="description_ticker"
                ),
                html.Div(
                    [
                        #Graph Content
                    ],
                    id="graph-content"
                ),
                html.Div(
                    [
                        #Indicator plot
                    ],
                    id = "main-content"
                ),
                html.Div(
                    [
                        #Forecast-plot
                    ],
                    id = "forecast-content"
                )
            ],className="content"
        )
        
    ], className="container"
)
#callback for company info
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("ticker", "children"),
        Output("stock", "n_clicks"),
        Output("indicator", "n_clicks"),
        Output("forecast", "n_clicks")
    ],
    [
        Input("submit", "n_clicks")
    ],
    [
        State("stock-code", "value")
    ]
    
    
)
def update_data(n, val):
    if n == None:
        return "Hey there! Please enter a legitimate stock code to get details.", "https://m.foolcdn.com/media/dubs/original_images/Intro_slide_-_digital_stock_chart_going_up_-_source_getty.jpg", "Stonks", None, None, None
    else:
        if val == None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info 
            df = pd.DataFrame().from_dict(inf, orient="index").T
            df[['logo_url', 'shortName', 'longBusinessSummary']]
            print (df['longBusinessSummary'].values[0])
            return df['longBusinessSummary'].values[0], df['logo_url'].values[0], df[ 'shortName'].values[0], None, None, None

#State allows you to pass along extra values without firing the callbacks.

#callback for stocks graphs
@app.callback(
    [
        Output("graph-content","children")
    ],
    [
        Input("stock", "n_clicks"),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')
    ],
    [
        State("stock-code", "value")
    ]
)
def stock_price(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    else:
        if start_date != None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    df.reset_index(inplace = True)
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure = fig)]


#callback for indicators

@app.callback(
    [
        Output("main-content","children")
    ],
    [
        Input("indicator", "n_clicks"),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')
    ],
    [
        State("stock-code", "value")
    ]
)
def indicators(n , start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]
    else:
        if start_date != None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    df.reset_index(inplace = True)
    fig = get_more(df)
    return [dcc.Graph(figure = fig)]

#callback for forecast
@app.callback(
    [
        Output("forecast-content","children")
    ],
    [
        Input("forecast", "n_clicks")
    ],
    [
        State("n-days", "value"),
        State("stock-code", "value")
    ]
)
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    return [dcc.Graph(figure = fig)]


if __name__ == '__main__':
    app.run_server(debug=True)
