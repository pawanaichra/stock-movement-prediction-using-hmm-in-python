from flask import Flask
from flask import request
from app.stock_prediction import stock_prediction_func
app = Flask(__name__) 
@app.route("/", methods=['GET'])
def root():
  try:
    stock = request.args.get('stock', '')
    return stock_prediction_func(stock)
  except Exception as err:
    print(err)
