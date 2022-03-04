from turtle import color
from  sklearn.svm import SVR
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from PIL import Image
from datetime import  date,datetime,timedelta
import yfinance as yf
import pickle
import plotly.offline as py 
from plotly.figure_factory import create_table 
import plotly.express as px   
import time

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
image = Image.open('crypto1.jpg')
st.image(image,use_column_width=5,) 
st.title("Bitcoin price prediction App (SVR Embedded)")

coins = ("BTC-USD", "ETH-USD","DOGE-USD","DOT-USD","SOL-USD")
selected_coin= st.selectbox("**Select Cryptocurrency**",coins)
day=st.slider("No of days prediction",1,15) 
myday=day
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load data...")
df=load_data(selected_coin)


#sevendaydf=pd.DataFrame(columns=['lin_svr','ploy_svr','rbf_svr','lstm','rf'])
#fifteendaydf=pd.DataFrame(columns=['lin_svr','ploy_svr','rbf_svr','lstm','rf'])




for j in range(len(df)):
    df['Date'][j]= datetime.strptime(str(df['Date'][j]),"%Y-%m-%d %H:%M:%S").date()

data_load_state.text("Loding data.....Done")


st.subheader("Raw Data")
st.write(df.head())

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


#st.write(df.tail(day))
df1=df.tail(day)
df.insert(0, 'Day', range(1, 1 + len(df)))
df=df.dropna()


actual_data=df.tail(day)
df= df.head(len(df)-day)

days=list()
adj_close_prices=list()

df_days=df.loc[:,'Day']
df_adj_close=df.loc[:,'Adj Close']


#create the independent set
for day in df_days:
  days.append([(day)])

#create the dependent set
for adj_close_price in df_adj_close:
  adj_close_prices.append(float(adj_close_price))


future_days= list()
for day in actual_data['Day']:
  future_days.append([int(day)])  

st.subheader("Select The Kernel Function")
kernel=st.radio("Pick one Kernel Function",["Linear Kernel","Polynomial Kernel","RBF Kernel"] )

#create 3 support vector model using kernal function

lin_svr_model = pickle.load(open('lin_svr.sav', 'rb'))
rbf_svr_model = pickle.load(open('rbf_svr.sav', 'rb'))
poly_svr_model = pickle.load(open('poly_svr.sav', 'rb'))



if kernel=="Linear Kernel":
    

    st.spinner(text="In progress...")
    with st.spinner('Creating plot...'):
      time.sleep(4)

    st.success('**Here is your plot !**')
    plt.figure(figsize=(20,9))
    plt.scatter(days,adj_close_prices,color='red',label='Data')
    plt.plot(days, lin_svr_model.predict(days),color ='blue', label= 'linear model', linewidth="4")
  
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    lin_predicted=pd.DataFrame(columns=['Date','predicted price'])
    lin_predicted["predicted price"]=lin_svr_model.predict(future_days)
    lin_predicted.insert(0, 'Day', range(1, 1 + len(lin_predicted)))

    actualPrice = list(df1.Close)

    
    today=date.today()
    #print(myday)
    for j in range(myday):
      lin_predicted['Date'][j]   =today-timedelta(days=(myday-j))
      
    
    lin_predicted["Actual_price"] = actualPrice
    st.subheader('The result of prediction using  Linear kernel function:')
    

    my_bar = st.progress(0)

    for percent_complete in range(100):
     time.sleep(0.05)
     my_bar.progress(percent_complete + 1)

    st.spinner(text="In progress...")
    with st.spinner('Wait for it...'):
      time.sleep(7)

    st.success('**Here is your result!**')
    st.table(lin_predicted)
     
    def mean_absolute_percentage_error(y_true,y_pred):
        y_true, y_pred= np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true-y_pred)/y_true))*100

    mape_baseline =mean_absolute_percentage_error(lin_predicted["Actual_price"],lin_predicted["predicted price"]) 
    st.write("**Mean Absolute Percent :**",mape_baseline)
    st.subheader("Result Visualization")
    plt.figure(figsize=(25,9))
    plt.plot(lin_predicted['Date'],lin_predicted['Actual_price'], color="red",linewidth="3", label= "Actual price")
    plt.plot(lin_predicted['Date'],lin_predicted['predicted price'], color="green", linewidth="3", label="Predicted price")
    plt.legend()
    st.pyplot()
    
    plt.bar(lin_predicted['Date'],lin_predicted['predicted price'], color="green")
    st.pyplot()
    
    #sevendaydf['lin_svr']=lin_predicted['predicted price']

   

   

elif kernel=="Polynomial Kernel":
      
    st.spinner(text="In progress...")
    with st.spinner('Creating plot...'):
      time.sleep(4)
    plt.figure(figsize=(20,9))
    plt.scatter(days,adj_close_prices,color='red',label='Data')
    plt.plot(days, poly_svr_model.predict(days),color ='orange', label= 'Polynomial model', linewidth="4")
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    poly_predicted=pd.DataFrame(columns=['Date','predicted price','Actual_price'])
    poly_predicted["predicted price"]=poly_svr_model.predict(future_days)
    poly_predicted.insert(0, 'Day', range(1, 1 + len(poly_predicted)))
    actualPrice = list(df1.Close)
    
    poly_predicted["Actual_price"] = actualPrice


    today=date.today()
    #print(myday)
    for j in range(myday):
      poly_predicted['Date'][j]   =today-timedelta(days=(myday-j))

    print(type(poly_predicted['Date']))


     

    st.subheader('**The result of prediction using  polynomial kernel function:**')


    my_bar = st.progress(0)

    for percent_complete in range(100):
     time.sleep(0.005)
     my_bar.progress(percent_complete + 1)

    st.spinner(text="In progress...")
    with st.spinner('Wait for it...'):
      time.sleep(4)

    st.success('**Here is your result!**')
    st.table(poly_predicted)
    st.subheader("Result Visualization")
    plt.figure(figsize=(25,9))
    plt.plot(poly_predicted['Date'],poly_predicted['Actual_price'], color="red",linewidth="3", label="Actual price")
    plt.plot(poly_predicted['Date'],poly_predicted['predicted price'], color="green", linewidth="3", label="Predicted Price")
    plt.legend()
    st.pyplot()
    
    plt.bar(poly_predicted['Date'],poly_predicted['predicted price'], color="green")
    st.pyplot()

    def mean_absolute_percentage_error(y_true,y_pred):
        y_true, y_pred= np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true-y_pred)/y_true))*100

    mape_baseline =mean_absolute_percentage_error(poly_predicted["Actual_price"],poly_predicted["predicted price"]) 
    st.write("**Mean Absolute Percent :**",mape_baseline)


    
   


    #sevendaydf['poly_svr']=poly_predicted['predicted price']

else:
    st.spinner(text="In progress...")
    with st.spinner('Creating plot...'):
      time.sleep(4)
    plt.figure(figsize=(20,9))
    plt.scatter(days,adj_close_prices,color='red',label='Data')
    plt.plot(days, rbf_svr_model.predict(days),color ='green',label= 'RBF Model', linewidth="4")
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    rbf_predicted=pd.DataFrame(columns=['Date','predicted price'])
    rbf_predicted["predicted price"]=rbf_svr_model.predict(future_days)
    rbf_predicted.insert(0, 'Day', range(1, 1 + len(rbf_predicted)))

    actualPrice = list(df1.Close)
    
    rbf_predicted["Actual_price"] = actualPrice

    today=date.today()
    #print(myday)
    for j in range(myday):
      rbf_predicted['Date'][j]   =today-timedelta(days=(myday-j))

    import time
    my_bar = st.progress(0)
    for percent_complete in range(100):
      time.sleep(0.005)
      my_bar.progress(percent_complete + 1)

    st.spinner(text="In progress...")
    with st.spinner('Wait for it...'):
      time.sleep(7)

    st.success('**Here is your result!**')
    

    st.subheader('**The result of prediction using RBF kernel function:**')
    st.table(rbf_predicted)
    def mean_absolute_percentage_error(y_true,y_pred):
      y_true, y_pred= np.array(y_true), np.array(y_pred)
      return np.mean(np.abs((y_true-y_pred)/y_true))*100
    st.subheader("Result Visualization")
    plt.figure(figsize=(25,9))
    plt.plot(rbf_predicted['Date'],rbf_predicted['Actual_price'], color="red",linewidth="3", label = "predicted price")
    plt.plot(rbf_predicted['Date'],rbf_predicted['predicted price'], color="green", linewidth="3", label="predicted price")
    st.pyplot()

    plt.bar(rbf_predicted['Date'],rbf_predicted['predicted price'], color="green")
    st.pyplot()
    mape_baseline =mean_absolute_percentage_error(rbf_predicted["Actual_price"],rbf_predicted["predicted price"]) 
    st.write("**Mean Absolute Percent :**",mape_baseline)

    #sevendaydf['rbf_svr']=rbf_predicted['predicted price']
    
    

st.subheader("For this prediction Support vector Regression Technique is Used")


#print(sevendaydf)



#sevendaydf.to_csv("sevendayeth.csv")


#fifteendaydf.to_csv("fifteendaybitcoin")














