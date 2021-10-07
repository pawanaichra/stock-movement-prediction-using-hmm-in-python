import json
import HMM
from utils import *

def lambda_handler(event, context):
    hmm = HMM()
    stock_prices = get_stock_prices('SBIN', '2017-01-01', '2017-04-01')
    price_movement = get_price_movements(stock_prices)
    optimal_states = hmm.get_optimal_states(price_movement)
    prediction = hmm.predict(price_movement, optimal_states)
    result = ""
    if prediction==1:
        result = "You should buy stock."
    else:
        result = "Sell stock if you have."
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
