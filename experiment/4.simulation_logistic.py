import os
### QAOA_realestate가 있는 폴더로 지정
dir_path = "/Users/minhyeong-gyu/Documents/GitHub/QuantumVariableSelection"
os.chdir(dir_path)
import sys
module_path = dir_path + "/Module"
if module_path not in sys.path:
    sys.path.append(module_path)

from optimizer import optimizer as opt
from optimizer import basefunctions as bf


import numpy as np
import pandas as pd
import argparse
import random
import json

import warnings

def fxn():
    warnings.warn("deprecated", RuntimeWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

parser = argparse.ArgumentParser(description='Process simulation parameters.')
parser.add_argument('-n','--number_samples', default = 1000, type = int, help='Number of Samples')
parser.add_argument('-p','--number_features', default = 100, type = int, help='Number of Features')
parser.add_argument('-q','--number_influentials', default = 20, type = int, help='Number of Infulential Features')
parser.add_argument('-e','--epsilon', default = None, type = float, help='Noise of Samples')
parser.add_argument('-cov','--covariance_parameter', default = 1, type = float, help='Magnitude of Covariance')
parser.add_argument('-lmbd','--lmbd', default = 0.1, type = float, help='Hyperparameter Controlling Convariance Mitigation')
parser.add_argument('-iter','--number_of_test', default = 100, type = int, help='a number of test')

if __name__ == "__main__":
    args = parser.parse_args()
    
    AIC_list_total =[]
    QUBO_list_total = []
    MSPE_list_total = []
    R2_list_total = []
    CN_list_total = []

    for ite in range(args.number_of_test):
        beta_coef = np.concatenate([np.random.normal(5,2,args.number_influentials),np.zeros(args.number_features-args.number_influentials)])
        if args.epsilon == None :
            epsilon = args.number_influentials*5/3
        else :
            epsilon = args.epsilon
        X,y = bf.generate_dependent_sample_logistic(args.number_samples,args.number_features,beta_coef,covariance_parameter=args.covariance_parameter,epsilon=epsilon)
        lmbd = args.lmbd
        y_type = "binary"

        sa_aic = opt.SimulatedAnnealing(mode = "AIC",y_type=y_type)
        sa_aic_result = sa_aic.optimize(X,y,lmbd,reps=10)
        ga_aic = opt.GeneticAlgorithm(mode = "AIC",y_type=y_type)
        ga_aic_result = ga_aic.optimize(X,y,lmbd)
        sa_qubo = opt.SimulatedAnnealing(mode = "QUBO",y_type=y_type)
        sa_qubo_result = sa_qubo.optimize(X,y,lmbd,reps=10)
        ga_qubo = opt.GeneticAlgorithm(mode = "QUBO",y_type=y_type)
        ga_qubo_result = ga_qubo.optimize(X,y,lmbd)

        X_sa_qubo = X[:,sa_qubo_result.astype(bool)]
        X_ga_qubo = X[:,ga_qubo_result.astype(bool)]
        X_sa_aic = X[:,sa_aic_result.astype(bool)]
        X_ga_aic = X[:,ga_aic_result.astype(bool)]
        datasets = [X,X_sa_aic,X_ga_aic,X_sa_qubo,X_ga_qubo]
        thetasets = [[1 for i in range(args.number_features)],sa_aic_result,ga_aic_result,sa_qubo_result,ga_qubo_result]

        AIC_list = []
        QUBO_list = []
        MSPE_list = []
        R2_list = []
        CN_list = []
        for i in range(5) :
            dataset = datasets[i]
            theta_temp = thetasets[i]
            AIC_list += [bf.get_aic(dataset,y,y_type=y_type)]
            Q,beta = bf.get_selecting_qubo(X,y,y_type=y_type)
            QUBO_list += [bf.get_QB(theta_temp,Q,beta,lmbd)]
            MSPE_list += [bf.get_accuracy(dataset,y,0.8)]
            R2_list += [bf.get_prediction_R2(dataset,y,0.8,y_type = y_type)]
            CN_list += [bf.get_CN(dataset)]


        AIC_list_total += [AIC_list]
        QUBO_list_total += [QUBO_list]
        MSPE_list_total += [MSPE_list]
        R2_list_total += [R2_list]
        CN_list_total += [CN_list]
        
    target = AIC_list_total
    AIC_result = pd.DataFrame(target).apply(lambda x: str(round(np.mean(x),2)))+"("+pd.DataFrame(target).apply(lambda x: str(round(np.std(x),2)))+")"
    target = QUBO_list_total
    QUBO_result = pd.DataFrame(target).apply(lambda x: str(round(np.mean(x),2)))+"("+pd.DataFrame(target).apply(lambda x: str(round(np.std(x),2)))+")"
    target = MSPE_list_total
    MSPE_result = pd.DataFrame(target).apply(lambda x: str(round(np.mean(x),2)))+"("+pd.DataFrame(target).apply(lambda x: str(round(np.std(x),2)))+")"
    target = R2_list_total
    R2_result = pd.DataFrame(target).apply(lambda x: str(round(np.mean(x),2)))+"("+pd.DataFrame(target).apply(lambda x: str(round(np.std(x),2)))+")"
    target = CN_list_total
    CN_result = pd.DataFrame(target).apply(lambda x: str(round(np.mean(x),2)))+"("+pd.DataFrame(target).apply(lambda x: str(round(np.std(x),2)))+")"


    result_table = pd.DataFrame([AIC_result,QUBO_result,MSPE_result,R2_result,CN_result])
    result_table.columns = ["Original","SA_AIC","GA_AIC","SA_QUBO","GA_QUBO"]
    result_table.index = ["AIC_list","QUBO_list","Accuracy","R2","CN"]


    random_index = str(random.randrange(0,99999)).zfill(5)
    
    result_table.to_csv("result/simulation_logistic/linear_samples_with_intercept"+random_index+".csv")

    with open("result/simulation/test_argument"+random_index+".txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    
    print(result_table)
    print(args.__dict__)
