
#  Market Insight AI

## Description

### StockPredictProphet.py
This script utilizes the Prophet forecasting model by Meta (formerly Facebook) to predict future stock prices. It provides an interactive web interface using Streamlit, allowing users to input stock ticker symbols, select historical data ranges, and forecast stock prices for specified future periods.

### StockPredictPytorch.py
This script leverages PyTorch for building and training an LSTM (Long Short-Term Memory) neural network model. It fetches historical stock data from the Alpha Vantage API and is capable of predicting future stock prices.

## Installation

1. **Clone the repository or download the scripts:**
   ```
   git clone https://github.com/DevSakkaravarthi/MarketInsightAI.git
   ```
   or download the scripts directly.

2. **Install required Python packages:**
   Navigate to the directory containing the `requirements.txt` file and run:
   ```bash
   pip install -r POC/requirements.txt
   ```

## How to Run

### Running StockPredictProphet.py
1. **Start the Streamlit web interface:**
   ```bash
   streamlit run POC/StockPredictProphet.py
   ```
2. **Use the web interface:**
   Open your web browser to interact with the Streamlit application.

### Running StockPredictPytorch.py
1. **Obtain an Alpha Vantage API key:**
   Visit [Alpha Vantage](https://www.alphavantage.co/) to get your API key. and add key in script 

2. **Execute the script:**
    Add stock symbol in code
   ```bash
   python POC/StockPredictPytorch.py
   ```
   The script will output the predicted stock prices for the upcoming 7 days.

## Requirements

- Python 3.x
- Streamlit (for `StockPredictProphet.py`)
- PyTorch (for `StockPredictPytorch.py`)
- Additional libraries: Pandas, NumPy, requests, yfinance, sklearn, prophet

Refer to `requirements.txt` for a complete list of dependencies.

## Support

For queries or issues, please open an issue in the repository or contact the maintainer.

