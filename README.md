# Predicting Stock Prices

### Using Yahoo Finance historical data, we predicted the stock price of several auto companies by creating several different models using SciKit-Learn Machine Learning in Python.

We processed the API data using Pandas and stored the output data in SQLiteDB.

Pandas_datareader module was used to fetch the stock data from Yahoo Finance API.

Data set (CSV) used was 2017 to 2020 daily General Motors Company (GM) stock price data.

The Adjusted Closing Price is the dependent (predicted) variable while the independent variables are price data related to Open, Close, Low and High.

Used Scikit-learn Module - train_test_split to split data into training data for training the model and test data to check the accuracy of the model.

Autocorrelation analysis and plot were performed comparing GM to other auto companies.

Used Plotly for visualizations of the model results.

Additionally we fetched data from the website Alpha Vantage, using API key, to get latest prediction of the company stock price.

We created a website using HTML, Flask and designed it using Bootstrap.

We type the Open, Highest and Lowest Price to get the Close Price prediction using linear regression formula.
