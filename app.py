import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
from datetime import  date, datetime,timedelta
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

#sevendaydf=pd.DataFrame(columns=['lin_svr','poly_svr','rbf_svr','lstm','rf'])





START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
image = Image.open('crypto1.jpg')
st.image(image,use_column_width=5,) 
st.title("Bitcoin price prediction App(Random Forest Embedded)")






coins = ("BTC-USD", "ETH-USD","DOGE-USD","DOT-USD","SOL-USD")
selected_coin= st.selectbox("Select Cryptocurrency",coins)
day=   st.slider("No of days prediction",1,15)


def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state=st.text("Load data...")
df=load_data(selected_coin)

for j in range(len(df)):
    df['Date'][j]= datetime.strptime(str(df['Date'][j]),"%Y-%m-%d %H:%M:%S").date()

data_load_state.text("Loding data.....Done")


st.subheader("Raw Data")

st.write(df.head())
#st.write(df.tail())
df=df.dropna()

df['Volume']=df['Volume'].astype(float)
print(df.dtypes)
print(df.info())
#df.set_index('Date')
import plotly.offline as py           # create table
import plotly.graph_objs as go        #create candlestick charts          

fig=yf.Ticker(selected_coin)
fig1=fig.history(period="1d",start=START,end=TODAY)
st.write("### Data Visualization ")
st.write(""" Bar chart for **Close price** """)
st.bar_chart(fig1.Close)
st.write(" Line chart for **Close price** ")
st.line_chart(fig1.Close)
st.write(" Line chart for **High** ")
st.line_chart(fig1.High)
st.write(" Area chart for **Close price** ")
st.area_chart(fig1.Close)


#create sorrelation matrix
import plotly as py
import seaborn as sns
if st.checkbox("**InterCorrelated Heatmap**"):
    st.header('Intercorrelation matrix heatmap')
    corr=df.corr()
    mask=np.zeros_like(corr)
    mask[np.triu_indices_from(mask)]=True
    with sns.axes_style("white"):
        f,ax=plt.subplots(figsize=(3,2))
        ax=sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    
forecast_out = day
df['prediction'] = df[['Close']].shift(-forecast_out)
X = np.array(df['Close']).reshape(-1,1)
X = X[:-forecast_out]
y = np.array(df['prediction'])
y = y[:-forecast_out]


reg = RandomForestRegressor(n_estimators = 30, max_depth =300, random_state = 17)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 7)
reg.fit(X_train, y_train)



df.drop(df.tail(day).index,inplace = True)
#st.write(df.tail(day))

result=df.tail(day)
#st.write(result)
new = pd.DataFrame(columns=['Date','Actual_price','Predicted_price'])


new['Date']=result['Date']
new['Actual_price']=result['Close']
new['Predicted_price']=result['prediction']
st.subheader(" Your Result")
st.table(new)

st.write("**(R^2):**", reg.score(X_test, y_test))

plt.figure(figsize=(25,9))
plt.plot(new['Date'], new['Predicted_price'], color="red", linewidth="3", label="predicted price")
plt.plot(new['Date'], new['Actual_price'], color="green",linewidth="3", label="Actual price")
plt.legend()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


st.subheader("For this prediction Random Forest Technique is Used")
