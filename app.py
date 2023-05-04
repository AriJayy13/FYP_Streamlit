import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from streamlit_option_menu import option_menu



#st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

#Header section

st.title("OilDrivenForex")

#sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title=None, 
        options=["Home", "Tool", "About", "Contact"],
        #orientation="horizontal",
    )
    if selected == "Home":
        st.title(f"Welcome")
    if selected == "Tool":
        st.title(f"Predict the future")
    if selected == "About":
        st.title(f"What we do")
    if selected == "Contact":
        st.header(f"We'd love to hear more")
        
    

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
