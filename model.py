from pickle import TRUE
from tkinter.tix import X_REGION
from typing import List


def prediction(stock, n_days):
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

    #model
    import numpy as np
    from datetime import date, timedelta
    from model import prediction
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR


    #load the data
    df = yf.download(stock, period='60d')
    df.reset_index(inplace = True)
    df['Day'] = df.index

    days = list()
    for i in range(len(df.Day)):
        days.append([i])

    #Splitting the dataset
    
    X = days
    Y = df[['Close']]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        
    y_train = y_train.values.ravel()
    y_train
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf',C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], max_iter=-1)
    best_svr.fit(x_train, y_train)
    svm_confidence = best_svr.score(x_train, y_train)
    print(svm_confidence)

    # Support Vector Regression Models

    # RBF model
    #rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=4.0)

    rbf_svr = best_svr.fit(x_train, y_train)
    output_days = list()
    for i in range(1, n_days):
        output_days.append([i + x_test[-1][0]])

    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days = 1)
        dates.append(current)


    # plot Results
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(x=np.array(x_test).flatten(),
    #                y=y_test.values.flatten(),
    #                mode='markers',
    #                name='data'))
    # fig.add_trace(
    #     go.Scatter(x=np.array(x_test).flatten(),
    #                y=rbf_svr.predict(x_test),
    #                mode='lines+markers',
    #                name='test'))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = dates,
            y = rbf_svr.predict(output_days),
            mode='lines+markers',
            name='data'
        )
    )

    fig.update_layout(
        title="Predicted Close Price of next " + str(n_days - 1) + " days",
        xaxis_title="Date",
        yaxis_title="Closed Price"
    )

    return fig
