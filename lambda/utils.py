import datetime as dt
from nsepy import get_history
import numpy as np

def get_stock_prices(company_symbol, start_date, end_date):
    # stock price data from nsepy library (closing prices)
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    stock_prices = get_history(symbol=company_symbol, start=start_date, end=end_date)
    # pandas dataframe to numpy array
    stock_prices = stock_prices.values
    # return closing prices
    return stock_prices[:,7]

def get_price_movements(stock_prices):
	price_change = stock_prices[1:] - stock_prices[:-1]
	price_movement = np.array(list(map((lambda x: 1 if x>0 else 0), price_change)))
	return price_movement