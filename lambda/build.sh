pip3 install --target ./packages -r requirements.txt
cd packages
rm -rf numpy* pandas*
zip -r ../lambda_deployment.zip .
cd ..
zip -g lambda_deployment.zip HMM.py lambda_function.py utils.py