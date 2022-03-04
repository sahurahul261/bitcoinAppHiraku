import base64
from locale import currency
from turtle import color
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
from PIL import Image
import time
import json
import requests
import base64
from bs4 import BeautifulSoup
import seaborn as sns


st.set_page_config(layout="wide")

image = Image.open('priceapp.jpg')
st.image(image,width=300)
st.title("Crypto Price App")
st.markdown("""
This app retrive cryptocurrency prices for the top 100 cryptocurrency from the **CoinMarketCap** !""")
expander_bar=st.expander("About")
expander_bar.markdown(""" 
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib
* **Data source:** [coinMarketCap]
""")

col1 =st.sidebar
col2, col3 = st.columns((2,1))

col1.header('Input Options')

currency_price_unit =col1.selectbox('Select currency for price',('USD','ETC'))

@st.cache
def load_data():
    cmc=requests.get('https://coinmarketcap.com')
    soup= BeautifulSoup(cmc.content, 'html.parser')

    data = soup.find('script',id='__NEXT_DATA__',type= 'application/json')
    coins={}
    coin_data =json.loads(data.contents[0])
    listings=coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
     
    attributes = listings[0]["keysArr"]
    index_of_id = attributes.index("id")
    index_of_slug = attributes.index("slug")

    for i in listings[1:]:
        coins[str(i[index_of_id])] = i[index_of_slug]

    coin_name=[]
    coin_symbol=[]
    market_cap=[]
    percent_change_1h=[]
    percent_change_24h=[]
    percent_change_7d=[]
    price=[]
    volume_24h=[]

    index_of_slug = attributes.index("slug")
    index_of_symbol = attributes.index("symbol")


   
    index_of_quote_currency_price = attributes.index(
        f"quote.{currency_price_unit}.price"
    )
    index_of_quote_currency_percent_change_1h = attributes.index(
        f"quote.{currency_price_unit}.percentChange1h"
    )
    index_of_quote_currency_percent_change_24h = attributes.index(
        f"quote.{currency_price_unit}.percentChange24h"
    )
    index_of_quote_currency_percent_change_7d = attributes.index(
        f"quote.{currency_price_unit}.percentChange7d"
    )
    index_of_quote_currency_market_cap = attributes.index(
        f"quote.{currency_price_unit}.marketCap"
    )
    index_of_quote_currency_volume_24h = attributes.index(
        f"quote.{currency_price_unit}.volume24h"
    )

    for i in listings[1:]:
        coin_name.append(i[index_of_slug])
        coin_symbol.append(i[index_of_symbol])

        price.append(i[index_of_quote_currency_price])
        percent_change_1h.append(i[index_of_quote_currency_percent_change_1h])
        percent_change_24h.append(i[index_of_quote_currency_percent_change_24h])
        percent_change_7d.append(i[index_of_quote_currency_percent_change_7d])
        market_cap.append(i[index_of_quote_currency_market_cap])
        volume_24h.append(i[index_of_quote_currency_volume_24h])


    df=pd.DataFrame(columns=['coin_name', 'coin_symbol', 'market_cap', 'percent_change_1h','percent_change_24h','percent_change_7d','price','volume_24h'])
    df['coin_name']=coin_name
    df['coin_symbol']=coin_symbol
    df['price']=price
    df['percent_change_1h']=percent_change_1h
    df['percent_change_24h']=percent_change_24h
    df['percent_change_7d']=percent_change_7d
    df['market_cap']=market_cap
    df['volume_24h']=volume_24h
    return df

df= load_data()

sorted_coin=sorted (df['coin_symbol'])
selected_coin=col1.multiselect('Cryptocurrency',sorted_coin,sorted_coin)


df_selected_coin=df[(df['coin_symbol'].isin(selected_coin))]

num_coin=col1.slider('Display top N coins',1,100,100)
df_coins=df_selected_coin[:num_coin]

percent_timeframe=col1._selectbox('percent change time frame',['7d','24h','1h'])
percent_dict={"7d":'percent_change_7d',"24h":'percent_change_24h'}
selected_percent_timeframe=percent_dict[percent_timeframe]

sort_values=col1.selectbox('Sort values?',['Yes','No'])

col2.subheader('Price Data of Selected Cryptocurrency')
col2.write('Data dimension :'+str(df_selected_coin.shape[0])+"rows and "+str(df_selected_coin.shape[1])+"columns.") 
col2.dataframe(df_coins)

df_change=pd.concat([df_coins.coin_symbol, df_coins.percent_change_1h, df_coins.percent_change_24h, df_coins.percent_change_7d],axis=1)
df_change=df_change.set_index('coin_symbol')
df_change['positive_percent_change_1h']=df_change['percent_change_1h'] > 0
df_change['positive_percent_change_24h']=df_change['percent_change_24h'] > 0
df_change['positive_percent_change_7d']=df_change['percent_change_7d'] > 0
col2.dataframe(df_change)

col3.subheader('Bar plot of % price change')

if percent_timeframe == '7d':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_7d'])
    col3.write('*7 days period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['percent_change_7d'].plot(kind='barh', color=df_change.positive_percent_change_7d.map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
elif percent_timeframe == '24h':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_24h'])
    col3.write('*24 hour period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['percent_change_24h'].plot(kind='barh', color=df_change.positive_percent_change_24h.map({True: 'o', False: 'r'}))
    col3.pyplot(plt)
    df.plot(color)
else:
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_1h'])
    col3.write('*1 hour period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['percent_change_1h'].plot(kind='barh', color=df_change.positive_percent_change_1h.map({True: 'p', False: 'r'}))
    col3.pyplot(plt)






