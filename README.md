# stock-movement-prediction-using-hmm-in-python
This python script predicts stock movement for next day.
Changes in the stock prices is really a difficult task otherwise there is no need.
So we tried only to predict directional movement of stock prices movement using 1st order discrete Hidden Markov Model in Python & implemented EM hill climbing algorithm.
We have implemented forward-backward and Baum-welch algorithms to find unknown parameters and to predict future states of stock prices.
Model predicted near future price movements in SBI stock with about 80% accuracy for the year 2016 using two hidden states.

## Deployment
    You can modify the original script and use however you want to.
    But I am going to create another replica which could be used directly for AWS lamba and also going to create a react app for this
