import requests
from api_key import apikey
import pandas as pd
import glob
import os
from sklearn import linear_model

from flask import Flask, render_template, request, redirect

app = Flask(__name__, static_url_path='/static')

symbol = ""


@app.route("/")
@app.route("/step1")
def step1():
    print(f"step 1")

    files = glob.glob('static/data/*.csv')
    for filename in files:
        os.unlink(filename)

    return render_template("step1.html")


@app.route('/ticker_handler', methods=['POST'])
def ticker_handler():
    print(f"ticker_handler")

    symbol = request.form['tickerID']
    print(f"symbol= {symbol}\n")

    function = 'TIME_SERIES_DAILY'
    outputsize = 'compact'

    # https: // www.alphavantage.co/documentation/
    PARAMS = {'function': function,
              'symbol': symbol,
              'outputsize': outputsize,
              'apikey': apikey}

    URL = 'https://www.alphavantage.co/query'

    request_data = requests.get(url=URL, params=PARAMS)
    request_json = request_data.json()
    print(f"symbol= {request_json}\n")

    stock_data = request_json['Time Series (Daily)']
    stock_df = pd.DataFrame(stock_data)
    print(stock_df.head())

    # Transpose index and columns
    df = stock_df.T
    print(df.head())
    print()

    df = df.rename(columns={'1. open': 'Open', '2. high': 'High',
                            '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volumn'})

    df.index.names = ['Date']
    print(df.head())
    print()

    csv_file = "static/data/" + symbol + ".csv"

    df.to_csv(csv_file)

    return render_template("step1.html", message="Please click on Step 2 on the menu bar.", symbol=symbol)


@app.route("/step2/")
def step2_no_ticker():
    print(f"step2_no_ticker")
    return render_template("step2.html", symbol="")


@app.route("/step3/")
def step3_no_ticker():
    print(f"step3_no_ticker")
    return render_template("step1.html")


@app.route("/step2/<ticker>")
def step2(ticker=None):
    print(f"step2")
    symbol = ticker
    return render_template("step2.html", symbol=symbol)


@app.route("/step3/<ticker>")
def step3(ticker=None):
    print(f"step 3")
    symbol = ticker

    filename = "static/data/" + symbol + ".csv"

    df = pd.read_csv(filename)

    html_df = df.loc[0:2, ["Date", "Open", "High", "Low", "Close"]]
    html = html_df.to_html(index=False)

    return render_template("step3.html", symbol=symbol, price_table=html)


@app.route("/prediction", methods=['POST'])
def prediction():
    print(f"prediction")

    symbol = request.form['ticker']
    openprice = float(request.form['open_price'])
    highprice = float(request.form['high_price'])
    lowprice = float(request.form['low_price'])

    filename = "static/data/" + symbol + ".csv"

    df = pd.read_csv(filename)
    print(df.head())

    html_df = df.loc[0:2, ["Date", "Open", "High", "Low", "Close"]]
    html = html_df.to_html(index=False)

    X = df[['Open', 'High', 'Low']].astype(float)
    Y = df[['Close']].astype(float)

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    prediction_close = regr.predict([[openprice, highprice, lowprice]])

    return render_template("step3.html", message=prediction_close, symbol=symbol, price_table=html)


if __name__ == "__main__":
    app.run(debug=True)
