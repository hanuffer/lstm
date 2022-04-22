import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import plotly.graph_objs as go

import plotly.figure_factory as ff



#start = '2010-01-01'
#end ='2021-12-31'

st.title('Stock ForeCasting')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start_input = st.text_input('Enter Start Date', '2010-01-01')
end_input = st.text_input('Enter End Date', '2021-12-31')
df = data.DataReader(user_input,'yahoo',start_input,end_input)
option = st.selectbox('Select Graph you want to see',('Data','Closing Vs Opening', '100MA', '200MA', 'Predicted'))


#Data Representation
if option=='Data':
    st.subheader('Data from 2010-2021')
    st.write(df.describe())

#Graphs 
elif option=='Closing Vs Opening':
    st.subheader('Closing Price Vs Time chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Open,'r',label='Opening Price')
    plt.plot(df.Close,'g',label='Closing Price')
    plt.legend()
    st.pyplot(fig)

elif option=='100MA':
    st.subheader('Closing Price Vs Time chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

elif option=='200MA':
    st.subheader('Closing Price Vs Time chart with 200MA')
    ma200 = df.Close.rolling(200).mean()
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig)

#Splitting Data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


    
#load our LSTM model
model= load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test,y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler= scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

if option=='Predicted':
    #forecasting Graph
    st.subheader('Actual VS Predicted Graph')
    fig2= plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_predicted,'r',label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.plotly_chart(fig2,use_container_width=True)