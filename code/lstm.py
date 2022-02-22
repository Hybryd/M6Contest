#!/usr/bin/env python
# coding: utf-8

# # Model : LTSM

# In[2]:


import pandas as pd
import numpy as np
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

pd.options.mode.chained_assignment = None 


# In[5]:


# Load data
df = pd.read_csv("../data/ohlcv_m6.csv")
print(df[df["Symbol"]=="CARR"])


# In[10]:


def make_predictions(df):
    # PARAMETERS
    split_percent = 1 # Split ratio for the train/test split. Set to 1 if we do not test the model
    look_back = 15 # Look back for the LTSM
    num_epochs = 50 # Number of epochs for the LTSM
    num_prediction = 30 # Make forecast for the next month
    
    # Forecast dates
    last_date = df['Date'].values[-1]
    forecast_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    
    # This dataframe stores the forecasts of the next month for every symbols
    res = pd.DataFrame()
    res["Date"] = forecast_dates
    
    # Get list of assets
    assets = pd.unique(df["Symbol"].values.ravel())

    for symbol in assets:
        tf.keras.backend.clear_session()
        print(symbol)
        
        # Get data. Keep only Close values
        dataframe = df[df["Symbol"]==symbol]
        #print("##############",len(dataframe))
        dataframe['Date'] = pd.to_datetime(dataframe['Date'], format="%Y-%m-%d")
        dataframe.set_axis(dataframe['Date'], inplace=True)
        dataframe.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
        close_data = dataframe["Close"].values.reshape(-1,1)
        
        #print("##############",len(close_data))
        
        # Split
        split = int(split_percent*len(close_data))
        close_train = close_data[:split]
        #close_test = close_data[split:]
        date_train = dataframe['Date'][:split]
        #date_test = dataframe['Date'][split:]
        
        
        train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
        #test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

        # Create the model (LTSM)
        model = Sequential()
        model.add(
            LSTM(10,
                activation='relu',
                input_shape=(look_back,1))
        )
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        #dot_img_file = symbol+"_model.png"
        #tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

        
        # Fit the model
        model.fit(train_generator, epochs=num_epochs, verbose=2)
        
        # Predictions
        #prediction = model.predict(test_generator)

        close_train = close_train.reshape((-1))
        #close_test = close_test.reshape((-1))
        #prediction = prediction.reshape((-1))
                
        # Make prediction for the next month
        close_data = close_data.reshape((-1))

       
        prediction_list = close_data[-look_back:]

        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        forecast = prediction_list[look_back-1:]

        
        # Transform results as dataframes
        df_train = pd.DataFrame({"Date":date_train,"Train" : close_train})
        #df_test = pd.DataFrame({"Date":date_test,"Test" : close_test})
        #df_prediction = pd.DataFrame({"Date":date_test[look_back:],"Prediction" : prediction})
        df_forecast = pd.DataFrame({"Date":forecast_dates,"Forecast" : forecast})
        res[symbol] = forecast
        
        
    return res#df_train, df_test, df_prediction, df_forecast
        
        
# In[11]:


#df_train, df_test, df_prediction, df_forecast = make_predictions(df)
forecasts = make_predictions(df)


# In[ ]:


#forecasts.plot(x="Date", y=list(forecasts.columns[1:]), label=list(forecasts.columns[1:]))
#plt.show()
forecasts.to_csv("forecasts.csv")


# In[ ]:





# In[ ]:




