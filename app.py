import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from streamlit_option_menu import option_menu
import time





#Title section

st.title("Oil Drive Forex")


#sidebar menu

with st.sidebar:
    col1, mid, col2 = st.columns([1,9,20])
    with col1:
        st.image('streamlit_logo.png', width=100)
    with col2:
        st.header('Oil Drive Forex')
        
    selected = option_menu(
        menu_title=None, 
        options=["Home", "Tool", "Contact"],
        #orientation="horizontal", 
    )
    
    
    
        
if selected == "Home":
    with st.expander("About the Project"):
        st.markdown("""
        This project focuses on the development of a financial data analysis application, specifically designed to forecast oil prices using historical financial data. The application leverages the power of Python and its extensive libraries to fetch, preprocess, and analyze data, resulting in predictions using an LSTM (Long Short-Term Memory) model.
        """)
        
    with st.expander("How it Works?"):
        st.markdown("""
            1. Data Retrieval: The application retrieves historical Brent oil prices and Euro to USD exchange rates using the FRED (Federal Reserve Economic Data) API. The user provides an API key, start date, and end date to fetch the required data.

        2. Data Preprocessing: The application processes the retrieved data to create a clean and structured dataset suitable for analysis. It merges the Brent oil prices and Euro to USD exchange rates based on the date, converting them to a pandas DataFrame.

        3. Feature Engineering: The application generates input and output data for the LSTM model using a sliding window approach. It scales the input and output data using the MinMaxScaler to improve the model's training performance.

        4. Data Splitting: The application splits the data into training, validation, and testing sets, with an 80-20 split for both the training-testing and training-validation sets.

        5. Model Training: The application trains an LSTM model using the Keras deep learning library. It employs the Adam optimizer and mean squared error as the loss function. The model is trained using the training data and validated using the validation data.

        6. Model Evaluation: The application evaluates the trained LSTM model on the testing data, calculating the loss function to measure the model's performance.

        7. Prediction: The application uses the trained LSTM model to make predictions on new input data, providing insights into future oil price trends based on historical data.
        """)
    with st.expander("Importance of this Project"):
            st.markdown("""
            The importance of this financial data analysis application lies in its ability to provide accurate and timely forecasts of oil prices, a critical factor in the global economy. Oil prices directly impact various industries, including transportation, manufacturing, and energy production. Accurate forecasting of oil prices enables businesses, investors, and policymakers to make informed decisions, effectively manage risks, and identify potential opportunities.
    """)
    
    #st.write(f"Predict the future")
    st.header("Latest EUR/USD rates")
    def get_euro_to_usd(api_key, outputsize="compact"):
        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&apikey={api_key}&outputsize={outputsize}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "Time Series FX (Daily)" in data:
                df = pd.DataFrame(data["Time Series FX (Daily)"]).T
                df.reset_index(inplace=True)
                df.columns = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]
                df["DATE"] = pd.to_datetime(df["DATE"])
                df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
                return df
            else:
                print("Error: Unable to fetch data. Please check the API key or the API limits.")
                return None
        else:
            print("Error: Unable to fetch data.")
            return None

    api_key = "CYT9KLKTLECSR3X2"  # Replace with your Alpha Vantage API key

    df = get_euro_to_usd(api_key)

    if df is not None:
        df_last_five_days = df.head(5)
        fig = px.line(df_last_five_days, x="DATE", y="CLOSE", title="EUR/USD Exchange Rate (Past 5 Days)", labels={"DATE": "Date", "CLOSE": "Exchange Rate"})
        st.plotly_chart(fig)

        # Print the latest available exchange rate
        latest_date = df["DATE"].max()
        latest_rate = df.loc[df["DATE"] == latest_date, "CLOSE"].values[0]
        st.write(f"Latest available EUR/USD exchange rate ({latest_date.date()}): {latest_rate:.4f}")
    else:
        st.error("Failed to fetch data.")





    st.header("Latest Brent Oil rates")
    # Function to fetch Brent Oil price data
    def get_brent_oil_prices(api_key, start_date, end_date):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=968&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DCOILBRENTEU&scale=left&cosd={start_date}&coed={end_date}&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date={end_date}&revision_date={end_date}&nd=1987-05-20"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text.splitlines()
            data = [row.split(",") for row in data]
            df = pd.DataFrame(data[1:], columns=data[0])
            df["DATE"] = pd.to_datetime(df["DATE"])
            df["DCOILBRENTEU"] = pd.to_numeric(df["DCOILBRENTEU"], errors="coerce")
            return df
        else:
            print("Error: Unable to fetch data.")
            return None
    api_key = "8551ea1d6053004447ea8a0dc7d580b6"  # Replace with your FRED API key
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(days=5)).strftime("%Y-%m-%d")

    df = get_brent_oil_prices(api_key, start_date, end_date)

    if df is not None:
        fig = px.line(df, x="DATE", y="DCOILBRENTEU", title="Brent Oil Prices (Past 5 Days)", labels={"DATE": "Date", "DCOILBRENTEU": "Price (USD)"})
        st.plotly_chart(fig)

        # Print the latest available Brent Oil price
        latest_date = df["DATE"].max()
        latest_rate = df.loc[df["DATE"] == latest_date, "DCOILBRENTEU"].values[0]
        st.write(f"Latest available Brent Oil price ({latest_date.date()}): USD {latest_rate:.2f}")
    else:
        st.error("Failed to fetch data.")
    
    
if selected == "Tool":    
    st.header("LSTM Model Prediction")  


    # Add a checkbox to the Streamlit app
    show_lstm_predictions = st.checkbox("Launch model")

    # If the checkbox is ticked, execute the code inside the if statement
    if show_lstm_predictions:
        with st.spinner("Loading..."):

            import requests
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from datetime import datetime
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            from keras.metrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
            from sklearn.metrics import r2_score



            api_key = "8551ea1d6053004447ea8a0dc7d580b6" 
            start_date = "2000-01-01"
            end_date = datetime.today().strftime('%Y-%m-%d')

            def get_brent_oil_prices(api_key, start_date, end_date):
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DCOILBRENTEU&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"

                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()["observations"]
                    df = pd.DataFrame(data)
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.set_index("date")
                    df = df.drop(columns=["realtime_start", "realtime_end"])
                    print(df.columns)
                    return df
                else:
                    print("Failed to retrieve data.")
                    return None

            brent_oil_prices = get_brent_oil_prices(api_key, start_date, end_date)

            def get_euro_to_usd_exchange_rate(api_key, start_date, end_date):
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DEXUSEU&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"

                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()["observations"]
                    df = pd.DataFrame(data)
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.set_index("date")
                    df = df.drop(columns=["realtime_start", "realtime_end"])
                    return df
                else:
                    print("Failed to retrieve data.")
                    return None

            euro_to_usd = get_euro_to_usd_exchange_rate(api_key, start_date, end_date)

            def join_dataframes_by_date(df1, df2, on="date", suffixes=("_brent", "_euro_usd")):
                return pd.merge(df1.reset_index(), df2.reset_index(), on=on, suffixes=suffixes)

            if brent_oil_prices is not None and euro_to_usd is not None:
                merged_data = join_dataframes_by_date(brent_oil_prices, euro_to_usd)
                merged_data = merged_data.dropna()
                print(merged_data)

            def prepare_data(df, n_lag):
                X, Y = [], []
                for i in range(n_lag, len(df)):
                    X.append(df.iloc[i-n_lag:i, 1:].values)
                    Y.append(df.iloc[i, 2])
                return np.array(X), np.array(Y)

            X_data, Y_data = prepare_data(merged_data, 7)

            input_scaler = MinMaxScaler()
            output_scaler = MinMaxScaler()

            X = input_scaler.fit_transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)

            Y = output_scaler.fit_transform(Y_data.reshape(-1, 1)).reshape(-1)

            first_split = int(len(X) * 0.8)
            X_train_full, X_test = X[:first_split], X[first_split:]
            Y_train_full, Y_test = Y[:first_split], Y[first_split:]

            second_split = int(len(X_train_full) * 0.8)
            X_train, X_val = X_train_full[:second_split], X_train_full[second_split:]
            Y_train, Y_val = Y_train_full[:second_split], Y_train_full[second_split:]

            print(Y_train_full.shape, Y_val.shape, Y_test.shape)
            print(X_train_full.shape, X_val.shape, X_test.shape)

            def train_lstm_model(x_train, y_train, x_val, y_val, n_epochs=50, batch_size=32):
                # Create the LSTM model
                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Training the model
                model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_val, y_val))

                return model

            model = train_lstm_model(X_train,Y_train, X_val, Y_val)

            predict = model.predict(X_test)

            plt.plot(predict)
            plt.plot(Y_test)
            

            test_loss = model.evaluate(X_test, Y_test)

            print(test_loss/0.6*100)

            model.compile(optimizer='adam', loss='mean_squared_error',
                          metrics=[MeanAbsoluteError(), MeanSquaredError(), MeanAbsolutePercentageError()])## Get Date

            # Prepare the input data for the model
            last_7_days_data = merged_data.iloc[-7:, 1:].values
            last_7_days_data_scaled = input_scaler.transform(last_7_days_data)

            # Use the model to predict the EUR/USD value for tomorrow
            tomorrow_prediction_scaled = model.predict(last_7_days_data_scaled[np.newaxis, :, :])

            # Inverse transform the prediction to get the value in the original scale
            tomorrow_prediction = output_scaler.inverse_transform(tomorrow_prediction_scaled)

            # Print the predicted EUR/USD value for tomorrow
            st.subheader(f"Predicted EUR/USD value for tomorrow: {tomorrow_prediction[0][0]:.4f}")
            st.text("")
            st.text("")
            st.subheader("Forecast vs Truth")
            def plot_prediction_vs_truth(predict, Y_test):
                data = pd.DataFrame({'Predicted': predict.flatten(), 'True': Y_test})
                return data
            data = plot_prediction_vs_truth(predict, Y_test)
            st.line_chart(data)


            
  

            # Assuming you have the 'predict' and 'Y_test' arrays from your model

            

            st.text("")
            
            

            import altair as alt


            def plot_prediction_graph(dates, exchange_rates):
                data = pd.DataFrame({'Date': dates, 'Exchange Rate': exchange_rates})
                data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime
                data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')  # Convert dates to strings in the "YYYY-MM-DD" format
                chart = alt.Chart(data).mark_line(point=True).encode(
                    x=alt.X('Date:T', axis=alt.Axis(title='Date', labelAngle=-45)),
                    y=alt.Y('Exchange Rate:Q', axis=alt.Axis(title='Exchange Rate')),
                    tooltip=['Date:T', 'Exchange Rate:Q']
                ).properties(
                    title='EUR/USD Exchange Rate Prediction Overtime',
                    width=800,
                    height=400
                )
                return chart


            # Prepare the input data for the model
            last_7_days_data = merged_data.iloc[-7:, 1:].values
            last_7_days_data_scaled = input_scaler.transform(last_7_days_data)

            # Use the model to predict the EUR/USD value for tomorrow
            tomorrow_prediction_scaled = model.predict(last_7_days_data_scaled[np.newaxis, :, :])

            # Inverse transform the prediction to get the value in the original scale
            tomorrow_prediction = output_scaler.inverse_transform(tomorrow_prediction_scaled)

            # Create a list of dates for the past 7 days and tomorrow
            past_7_days_dates = merged_data.index[-7:].tolist()
            tomorrow_date = (pd.Timestamp.today() + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            past_7_days_dates.append(tomorrow_date)

            # Create a list of exchange rates for the past 7 days and the predicted value for tomorrow
            past_7_days_exchange_rates = merged_data["value_euro_usd"].iloc[-7:].tolist()
            past_7_days_exchange_rates.append(tomorrow_prediction[0][0])

            # Create the chart and display it using Streamlit
            chart = plot_prediction_graph(past_7_days_dates, past_7_days_exchange_rates)
            st.altair_chart(chart)

            st.header("Evaluation Metrics")

            def evaluate_lstm_model(model, x_test, y_test):
                # Make predictions on the test set
                y_pred = model.predict(x_test)

                # Calculate evaluation metrics
                mae = MeanAbsoluteError()(y_test, y_pred).numpy()
                mse = MeanSquaredError()(y_test, y_pred).numpy()
                rmse = np.sqrt(mse)
                mape = MeanAbsolutePercentageError()(y_test, y_pred).numpy()
                r2 = r2_score(y_test, y_pred)

                return {"Mean absolute Error": mae, "Mean Squared Error": mse, "rmse": rmse, "Mean Absolute Percentage Error": mape, "r2 score": r2}

            evaluation_metrics = evaluate_lstm_model(model,X_test, Y_test)
            st.write(evaluation_metrics)

            # Assuming you have already loaded the model, X_test and Y_test
            evaluation_metrics = evaluate_lstm_model(model, X_test, Y_test)

            # Create a pandas DataFrame from the evaluation_metrics dictionary
            metrics_df = pd.DataFrame.from_dict(evaluation_metrics, orient="index", columns=["Value"])

            # Plot the evaluation metrics as a bar chart
            st.bar_chart(metrics_df)
            
if selected == "Contact":  
    st.subheader("We'd love to hear more")
    contact_form= """
    <form action="https://formsubmit.co/w1839045@my.westminster.ac.uk" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder=" Your name" required>
        <input type="email" name="email" placeholder=" Your email" required>
        <textarea name="message" placeholder="Your message"></textarea>
        <button type="submit">Send</button>
    </form>
"""
    st.markdown(contact_form, unsafe_allow_html=True)
    #Using local CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    local_css("style.css")
 
    
