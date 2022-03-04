import streamlit as st
import pandas as pd
from datetime import  date, datetime,timedelta
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
from datetime import  date
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
import time

START = "2015-01-01"

TODAY = date.today().strftime("%Y-%m-%d")

st.title("Price Prediction App (deep learning Embedded)")
coins = ("BTC-USD", "ETH-USD","DOGE-USD","DOT-USD","SOL-USD")
selected_coin= st.selectbox(" Select *Cryptocurrency*",coins)
day=st.slider("No of days prediction",1,15)
algos=("Machine Learning -Random Forest","Deep Learning- LSTM")
#selected_algo= st.selectbox("Select Prediction technique ",algos)

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



st.subheader("Data Visualizaton")

fig=yf.Ticker(selected_coin)

fig1=fig.history(period="1d",start=START,end=TODAY)

st.write(" Line chart for **Close price** ")
st.line_chart(fig1.Close)
st.write(" Line chart for **High** ")
st.line_chart(fig1.High)
st.write(" Area chart for **Close price** ")
st.area_chart(fig1.Close)



df1=df['Close'].head(len(df)-day)
actual_close=df['Close'].tail(day)

sc=MinMaxScaler(feature_range= (0,1))  #feature scaling 
df1= sc.fit_transform(np.array(df1).reshape(-1,1))


train_size=int (len(df1)*0.85)

test_size=(len(df1)- train_size )
train_data, test_data= df1[0:train_size,:], df1[train_size:len(df1),:1]  # test train split



import numpy

def create_dataset(dataset,time_step=1):
  datax, datay= [],[]
  for i in range (len(dataset)-time_step-1 ):
    a= dataset[i:(i+time_step),0] 
    datax.append(a)
    datay.append(dataset[i+time_step,0])
  return numpy.array(datax),numpy.array(datay)



  #reshape into x=t,t+1,t+2,t+3 and y =t+4
time_step=100
X_train, y_train = create_dataset(train_data,time_step)
X_test,y_test =create_dataset(test_data,time_step)


# reshape input so that we can feed into lstm 
 
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# part 2 - building the RNN
#importing the keras library and packages

import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialising the RNN


from numpy import loadtxt
from keras.models import load_model
model = load_model('model.h5')
# summarize model.


#print(test_data.shape)
x_input=test_data[len(test_data)-100:].reshape(1,-1)
#print(x_input.shape)



temp_input=list(x_input)
temp_input=temp_input[0].tolist()



from numpy import array

#predicting next 30 days price suing the current data
#it will predict in slidnig window manner (algorithm) with slide

lst_output=[]
n_steps=100
i=0
while(i<day):
  if(len(temp_input)>100): 
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
  else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
print("{} day output {}".format(i,yhat))
print(lst_output)


df3= sc.inverse_transform(lst_output) 

lstm_result=pd.DataFrame(columns=['Date','Predicted_Price','Actual_Price'])
lstm_result.insert(0, 'Day', range(1, 1 + len(df3)))




print(type(df['Date'][0]))

lstm_result['Actual_Price']=list(actual_close)
lstm_result['Predicted_Price']=df3

#Sid code





#End code



current_time= date.today()

for i in range(day):
      lstm_result['Date'][i]=current_time-timedelta(days=(day-i))

      
my_bar = st.progress(0)
for percent_complete in range(100):
  time.sleep(0.05)
  my_bar.progress(percent_complete + 1)

st.subheader("Results")
st.table(lstm_result)


def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred= np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


st.spinner(text="In progress...")

with st.spinner('Wait for it...'):
   time.sleep(5)

st.success('**Here is your result!**')
st.subheader("Result Visualization")
plt.figure(figsize=(25,9))
plt.plot(lstm_result['Date'], lstm_result['Predicted_Price'], color="red", linewidth="3", label="predicted price")
plt.plot(lstm_result['Date'], lstm_result['Actual_Price'], color="green",linewidth="3", label="Actual price")
plt.legend()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

plt.bar(lstm_result['Date'],lstm_result['Predicted_Price'], color="green")
st.pyplot()


mape_baseline =mean_absolute_percentage_error(lstm_result['Actual_Price'], lstm_result['Predicted_Price']) 
mape_baseline

st.write("**Mean Absolute Percent Error**",mape_baseline)
st.subheader("For this  prediction Long Short Term  Memory Technique of Deep Learning is Used")

df=pd.read_csv("sevendaybitcoin.csv")
df["lstm"]=lstm_result["Predicted_Price"]
df.to_csv("sevendaybitcoin.csv")