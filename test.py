import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
from datetime import  date
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("price prediction App (deep learning Embadded)")
coins = ("BTC-USD", "ETH-USD","LTC-USD")
selected_coin= st.selectbox("Select Cryptocurrency ",coins)
#day=st.slider("No of days prediction",1,15)
#algos=("Machine Learning -Random Forest","Deep Learning- LSTM")
#selected_algo= st.selectbox("Select Prediction technique ",algos)

def load_data(ticker):
    data=yf.download(ticker,START,TODAY,interval='1d')
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load data...")
df=load_data(selected_coin)



df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.set_index('Date')
data_load_state.text("Loding data.....Done")


st.subheader("Raw Data")
st.table(df.head())

df=df.dropna()

st.line_chart(df.Close) 

# fig=yf.Ticker(selected_coin)
# fig1=fig.history(period="1d",start=START,end=TODAY)
# st.line_chart(fig1.Close)

