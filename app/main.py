from flask import Flask, render_template
from flask import request
from app.stock_prediction import stock_prediction_func
app = Flask(__name__) 
@app.route("/", methods=['GET'])
def home():
  try:
    return render_template('home.html')
  except Exception as err:
    print(err)

@app.route("/predict", methods=['GET'])
def predict():
  try:
    stock = request.args.get('stock', '')
    if not stock:
      return render_template('home.html')
    return stock_prediction_func(stock)
  except Exception as err:
    print(err)
