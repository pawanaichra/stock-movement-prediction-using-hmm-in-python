# stock-movement-prediction-using-hmm-in-python
This python script predicts stock movement for next day.
Changes in the stock prices is really a difficult task otherwise there is no need.
So we tried only to predict directional movement of stock prices movement using 1st order discrete Hidden Markov Model in Python & implemented EM hill climbing algorithm.
We have implemented forward-backward and Baum-welch algorithms to find unknown parameters and to predict future states of stock prices.
Model predicted near future price movements in SBI stock with about 80% accuracy for the year 2016 using two hidden states.

## Deployment
    You can modify the original script and use however you want to.
    But I am going to create another replica which could be used directly for AWS lamba and also going to create a react app for this.

### Deploy on AWS Lambda
  - Go to lambda folder
  - Change permission for build.sh (```chmod 777 build.sh```)
  - Run build.sh (```./build.sh```)
  - Upload lambda_deployment.zip file on AWS Lambda
  - Add layer with these two arn: ```arn:aws:lambda:us-east-2:770693421928:layer:Klayers-python38-numpy:20``` and ```arn:aws:lambda:us-east-2:770693421928:layer:Klayers-python38-numpy:20```
  - Reference: https://github.com/keithrozario/Klayers/blob/master/deployments/python3.8/arns/us-east-2.csv
  - Change timeout and memory as required
