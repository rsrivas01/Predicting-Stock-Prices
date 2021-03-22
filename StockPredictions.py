import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import plotly.express as px
import plotly
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import plotly.express as px
import plotly
from sklearn import metrics
import sqlite3 as db

print('Script Start')
StockSymbol = "GM"
StockStart = datetime.datetime(2017, 1, 1)
StockEnd = datetime.datetime(2020, 12, 12)
df_auto = web.DataReader(StockSymbol, 'yahoo', StockStart, StockEnd)
FileName = "StockData_" + StockSymbol + ".csv"
df_auto['StockSymbol'] = StockSymbol
df_auto.to_csv(FileName)
print('Saved GM stock data ')

closingPrice = df_auto['Adj Close']
movingavg = closingPrice.rolling(window=50).mean() # 200 day moving Average
df_closingPrice = closingPrice.reset_index()
df_movingavg = movingavg.reset_index()

df_closingPrice['ValType'] = 'Adj Close'
df_movingavg['ValType'] = 'Moving Avg Adj Close'
df_dataViz = pd.concat([df_closingPrice,df_movingavg ]).fillna(0)
df_dataViz.to_csv('GM_AdjClose_MovingAvgAdjClose.csv', index=False)

fig = px.line(df_dataViz, x="Date", y="Adj Close", color='ValType', title='GM Adj Close & Moving Avg Adj Close')
plotly.offline.plot(fig, filename='AdjClose_MovingAvg.html')

print('GM_AdjClose_MovingAvgAdjClose plot generated')

# Get Stock Prices for companies in same business
df_auto_TM= web.DataReader('TM','yahoo',start=StockStart,end=StockEnd)['Adj Close'].reset_index()
df_auto_F= web.DataReader('F','yahoo',start=StockStart,end=StockEnd)['Adj Close'].reset_index()
df_auto_FCAU= web.DataReader('FCAU','yahoo',start=StockStart,end=StockEnd)['Adj Close'].reset_index()
df_auto_TM['Stock'] = 'Toyota'
df_auto_F['Stock'] = 'Ford'
df_auto_FCAU['Stock'] = 'FCA'
df_auto_GM = df_auto.reset_index()
df_auto_GM['Stock'] = 'General Motors'

df_auto_all = pd.concat([df_auto_TM, df_auto_F, df_auto_FCAU, df_auto_GM])
df_auto_all = df_auto_all.fillna(0)
df_auto_all_csv = df_auto_all.copy()
df_auto_all_csv.drop('StockSymbol', axis=1, inplace=True)
df_auto_all_csv.to_csv('StockPriceComparision.csv', index = False)

fig = px.line(df_auto_all, x="Date", y="Adj Close", color='Stock', title = 'Stock Price Comparision')
plotly.offline.plot(fig, filename='AdjClose_StocksPriceCompare.html')

print('Stock Price Comparision completed')

# Get Correlation between stock prices of similar companies
All_auto_stocks = web.DataReader(['GM', 'F', 'TM', 'FCAU', 'HMC'],'yahoo',start=StockStart,end=StockEnd)['Adj Close']

All_auto_stocksPctChange = All_auto_stocks.pct_change()
All_auto_stocksCorr = All_auto_stocksPctChange.corr()
All_auto_stocksCorr.to_csv('CorrelationWithStockPricesOtherCompanies.csv')

sns.set_theme(style="dark")


# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(All_auto_stocksCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(260, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
correlationFig = sns.heatmap(All_auto_stocksCorr, mask=mask, cmap=cmap, vmax=1, center= -1,
            square=True, linewidths=.8)
correlationFig.get_figure().savefig('Correlation.png', dpi=400)




# Start Model Building preparation
df_auto3 = df_auto.reset_index()

df_auto.to_csv('GMStockData.csv', index=False)
df_auto3.drop(['Date','StockSymbol'], axis=1, inplace=True)

print('Start Model Building preparation')
### Fill nulls with average value:
for col in df_auto3.columns:
    if col in df_auto3.select_dtypes(include=np.number).columns:
        avg = df_auto3[col].mean()
        df_auto3[col] = df_auto3[col].fillna(avg)
        
df_train, df_test = train_test_split(df_auto3, test_size=0.3)
TargetVariable = 'Adj Close'
x_train = df_train.drop(TargetVariable, axis=1)
x_test = df_test.drop(TargetVariable, axis=1)

y_train = df_train[TargetVariable]
y_test = df_test[TargetVariable]

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Linear regression
lnrReg = LinearRegression()
lnrReg.fit(x_train, y_train)

y_predict_linear = lnrReg.predict(x_test)

x_test_linear = x_test.copy()
x_test_linear['Adj Close'] = y_test
x_test_linear['ValueType'] = 'Actual' 
x_test_linear = x_test_linear.reset_index()
x_test_predicted = x_test_linear.copy()
x_test_predicted['ValueType'] = 'Predicted'
x_test_predicted['Adj Close'] = y_predict_linear
predictedValues = pd.concat([x_test_linear, x_test_predicted], ignore_index=True)
df_predictedVal = df_auto.reset_index().reset_index()[['index', 'Date']].merge(predictedValues, suffixes=('left','right'), on = 'index').fillna(0)

MSE_linear = metrics.mean_squared_error(y_test, y_predict_linear)
print('MSE :: Linear Regression ' + str(MSE_linear))
score_linear = lnrReg.score(x_test,y_test)
print('Accuracy Score:: Linear Regression: ' + str(score_linear))

fig = px.line(df_predictedVal, x="Date", y="Adj Close", color='ValueType', title = "Linear Regression")
plotly.offline.plot(fig, filename='LinearReg.html')

df_predictedVal.to_csv('PredictedValLinearReg.csv')

# Quadratic Regression 2 with Ridge
quadReg2 = make_pipeline(PolynomialFeatures(2), Ridge(normalize=True))
quadReg2.fit(np.array(x_train), y_train)

y_predict_quadReg2 = quadReg2.predict(x_test)
x_test_quadReg2 = x_test.copy()
x_test_quadReg2['Adj Close'] = y_test
x_test_quadReg2['ValueType'] = 'Actual' 
x_test_quadReg2 = x_test_quadReg2.reset_index()
x_test_predicted_quadReg2 = x_test_quadReg2.copy()
x_test_predicted_quadReg2['ValueType'] = 'Predicted'
x_test_predicted_quadReg2['Adj Close'] = y_predict_quadReg2
predictedValues_quadReg2 = pd.concat([x_test_quadReg2, x_test_predicted_quadReg2], ignore_index=True)
df_predictedVal_quadReg2 = df_auto.reset_index().reset_index()[['index', 'Date']].merge(predictedValues_quadReg2, suffixes=('left','right'), on = 'index').fillna(0)

MSE_quadReg2 = metrics.mean_squared_error(y_test, y_predict_quadReg2)
print('MSE :: Polynomial Degree 2 Ridge Regression ' + str(MSE_quadReg2))
score_quadReg2 = quadReg2.score(x_test,y_test)
print('Accuracy Score:: Polynomial Degree 2 Ridge Regression: ' + str(score_quadReg2))

fig = px.line(df_predictedVal_quadReg2, x="Date", y="Adj Close", color='ValueType', title = 'Degree2 Polynomial Ridge')
plotly.offline.plot(fig, filename='Deg2PolynomialRidge.html')
predictedValues_quadReg2.to_csv('PredictedDeg2PolyRidge.csv')

# Quadratic Regression 2 with Linear Regression

quadLin2 = make_pipeline(PolynomialFeatures(2), LinearRegression(normalize=True))
quadLin2.fit(np.array(x_train), y_train)

y_predict_quadLin2 = quadLin2.predict(x_test)
x_test_quadLin2 = x_test.copy()
x_test_quadLin2['Adj Close'] = y_test
x_test_quadLin2['ValueType'] = 'Actual' 
x_test_quadLin2 = x_test_quadLin2.reset_index()
x_test_predicted_quadLin2 = x_test_quadLin2.copy()
x_test_predicted_quadLin2['ValueType'] = 'Predicted'
x_test_predicted_quadLin2['Adj Close'] = y_predict_quadReg2
predictedValues_quadLin2 = pd.concat([x_test_quadLin2, x_test_predicted_quadLin2], ignore_index=True)
df_predictedVal_quadLin2 = df_auto.reset_index().reset_index()[['index', 'Date']].merge(predictedValues_quadLin2, suffixes=('left','right'), on = 'index').fillna(0)

MSE_quadLin2 = metrics.mean_squared_error(y_test, y_predict_quadLin2)
print('MSE :: Polynomial Degree 2 Linear Regression ' + str(MSE_quadLin2))
score_quadLin2 = quadLin2.score(x_test,y_test)
print('Accuracy Score :: Polynomial Degree 2 Linear Regression  ' + str(score_quadLin2))

fig = px.line(df_predictedVal_quadLin2, x="Date", y="Adj Close", color='ValueType', title = 'Degree2 Polynomial Linear Regression')
plotly.offline.plot(fig, filename='Deg2PolynomialLinearReg.html')
predictedValues_quadLin2.to_csv('PredictedDeg2PolyLinearReg.csv')

# Create Model Summary with Score and MSE of each Model

Summary = pd.DataFrame({'Model': ['Linear Regression', 'Degree2 Polynomial Ridge', 'Degree2 Polynomial Linear Regression'], 
              'Score': [score_linear,score_quadReg2, score_quadLin2 ],
             'MSE': [MSE_linear,MSE_quadReg2, MSE_quadLin2]})
con = db.connect('ML_Project.db')
df_predictedVal_quadLin2.to_sql(name='PredVal_QuadLin', con=con,if_exists='replace')
df_predictedVal_quadReg2.to_sql(name='PredVal_QuadRidge', con=con,if_exists='replace')
df_predictedVal.to_sql(name='PredVal_LinearReg', con=con,if_exists='replace')

Summary.to_sql(name='Summary', con=con,if_exists='replace')
Summary.to_csv('ModelEvaluationSummary.csv', index=False)

print('*************')
print(Summary)
print('-------------')

print('Script Completed')

