
#  Market Insight AI

## Description

`Market Insight AI` is a comprehensive tool for stock market prediction, utilizing advanced machine learning techniques. It includes two main scripts:


## Installation

1. **Clone the repository or download the scripts:**
   ```
   git clone https://github.com/DevSakkaravarthi/MarketInsightAI.git
   ```
   or download the scripts directly.

2. **Install required Python packages:**
   Navigate to the directory containing the `requirements.txt` file and run:
   ```bash
   pip install -r requirements.txt
   ```
4. **Obtain an polygon API key:**
   Visit [Polygon](https://polygon.io/) to get your API key. and add key to environment  
   ```
   export POLIGON_API_KEY='your_api_key_here'
   ```
   On Windows:


   ```
   set POLIGON_API_KEY=your_api_key_here
   ```

## How to Run

### Streamlit Web Interface (app_streamlit.py)

1. **Start the Streamlit web interface:**
   ```bash
   streamlit run app_streamlit.py
   ```
2. **Use the web interface:**
   Open your web browser to interact with the Streamlit application.

### Flask API Server (app_api.py)



1. **Execute the script:**
    Start the Flask server
   ```bash
   python app_api.py
   ```
2.  Use the endpoint /predict with a POST request to get stock    price predictions. The request should include JSON data with the stock symbol and number of days for prediction.

      Example POST request data:


      ```json
      {
      "symbol": "AAPL",
      "days": 7
      }
      ```
  
## Requirements

- Python 3.x
- Streamlit (for StockPredictProphet.py and app_streamlit.py)
- Flask (for app_api.py)
- PyTorch (for AI model)
- Additional libraries: Pandas, NumPy, requests, yfinance, sklearn, prophet

Refer to `requirements.txt` for a complete list of dependencies.

## Support

For queries or issues, please open an issue in the repository or contact the maintainer.

