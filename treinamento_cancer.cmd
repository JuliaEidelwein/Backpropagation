REM python configure_network.py cancer 0.25 31 5 2
REM python main.py cancer_rede.txt cancer_pesos.txt datasets/wdbc.txt >> datasetCancer.txt

REM python configure_network.py cancer2 0.25 31 10 2
REM python main.py cancer2_rede.txt cancer2_pesos.txt datasets/wdbc.txt >> datasetCancer2.txt

REM python configure_network.py cancer3 0.25 31 15 2
REM python main.py cancer3_rede.txt cancer3_pesos.txt datasets/wdbc.txt >> datasetCancer3.txt

REM python configure_network.py cancer4 0.25 31 5 5 2
REM python main.py cancer4_rede.txt cancer4_pesos.txt datasets/wdbc.txt >> datasetCancer4.txt

REM python configure_network.py cancer5 0.25 31 10 10 2
REM python main.py cancer5_rede.txt cancer5_pesos.txt datasets/wdbc.txt >> datasetCancer5.txt

REM python configure_network.py cancer6 0.25 31 15 15 2
REM python main.py cancer6_rede.txt cancer6_pesos.txt datasets/wdbc.txt >> datasetCancer6.txt

REM python configure_network.py cancer7 0.25 31 5 5 5 2
REM python main.py cancer7_rede.txt cancer7_pesos.txt datasets/wdbc.txt >> datasetCancer7.txt

REM python configure_network.py cancer8 0.25 31 10 10 10 2
REM python main.py cancer8_rede.txt cancer8_pesos.txt datasets/wdbc.txt >> datasetCancer8.txt

REM python configure_network.py cancer9 0.25 31 15 15 15 2
REM python main.py cancer9_rede.txt cancer9_pesos.txt datasets/wdbc.txt >> datasetCancer9.txt

REM lambdas 0, 0.25, 0.5, 1, 2, 5

python configure_network.py vinho_lambda_0 0 13 5 3
python main.py vinho_lambda_0_rede.txt vinho_lambda_0_pesos.txt datasets/wine.txt >> datasetVinho_lambda0.txt

python configure_network.py vinho_lambda_025 0.25 13 5 3
python main.py vinho_lambda_025_rede.txt vinho_lambda_025_pesos.txt datasets/wine.txt >> datasetVinho_lambda025.txt

python configure_network.py vinho_lambda_05 0.5 13 5 3
python main.py vinho_lambda_05_rede.txt vinho_lambda_05_pesos.txt datasets/wine.txt >> datasetVinho_lambda05.txt

python configure_network.py vinho_lambda_1 1 13 5 3
python main.py vinho_lambda_1_rede.txt vinho_lambda_1_pesos.txt datasets/wine.txt >> datasetVinho_lambda1.txt

python configure_network.py vinho_lambda_2 2 13 5 3
python main.py vinho_lambda_2_rede.txt vinho_lambda_2_pesos.txt datasets/wine.txt >> datasetVinho_lambda2.txt

python configure_network.py vinho_lambda_5 5 13 5 3
python main.py vinho_lambda_5_rede.txt vinho_lambda_5_pesos.txt datasets/wine.txt >> datasetVinho_lambda5.txt