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

import argparse
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import font_manager, rc
import warnings
# 윈도우 한글 폰트
# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

# Mac 한글 폰트
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


def fxn():
    warnings.warn("deprecated", RuntimeWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

from time import gmtime, strftime
time = strftime("%yy%mm%dd%H:%M", gmtime())
parent = "./result/"
child = "simulation_heatmap/test_"+str(time)
path = os.path.join(parent,child)
os.makedirs(path)

parser = argparse.ArgumentParser(description='Process simulation parameters.')
parser.add_argument('-n','--number_samples', default = 1000, type = int, help='Number of Samples')
parser.add_argument('-p','--number_features', default = 100, type = int, help='Number of Features')
parser.add_argument('-q','--number_influentials', default = 20, type = int, help='Number of Infulential Features')
parser.add_argument('-e','--epsilon', default = None, type = float, help='Noise of Samples')
parser.add_argument('-cov','--covariance_parameter', default = 1, type = float, help='Magnitude of Covariance')
parser.add_argument('-lmbd','--lmbd', default = 0.1, type = float, help='Hyperparameter Controlling Convariance Mitigation')
parser.add_argument('-iter','--number_of_test', default = 100, type = int, help='a number of test')
parser.add_argument('-sp','--selected_features', default = 12, type = int, help='a number of selected features')

num_test = 2

number_samples = 1000
number_features = 40
number_influentials = 8

covariance_parameter = 50
epsilon = 0

n_features=16
lmbd = 0.01

if __name__ == "__main__":
    args = parser.parse_args()
    num_test = args.number_of_test

    number_samples = args.number_samples
    number_features = args.number_features
    number_influentials = args.number_influentials

    covariance_parameter = args.covariance_parameter
    epsilon = args.epsilon

    n_features = args.selected_features
    lmbd = args.lmbd

    sa_mi_heatmap = []
    sa_full_heatmap = []
    sa_partial_heatmap = []
    Accuracy_list_total = []
    CN_list_total = []

    for i in range(num_test):
        beta_coef = np.concatenate([np.random.normal(10,1,number_influentials),np.zeros(number_features-number_influentials)])
        X,y = bf.generate_dependent_sample_logistic(number_samples,number_features,beta_coef,covariance_parameter=covariance_parameter,epsilon=epsilon)
        y_type="binary"
        sa_mi = opt.SimulatedAnnealing("QUBO",y_type=y_type,measure="mi")
        sa_mi_result = sa_mi.optimize(X,y,lmbd,reps=10,n_features=n_features)
        sa_full = opt.SimulatedAnnealing("QUBO",y_type=y_type,measure="full")
        sa_full_result = sa_full.optimize(X,y,lmbd,reps=10,n_features=n_features)
        sa_partial = opt.SimulatedAnnealing("QUBO",y_type=y_type,measure="partial")
        sa_partial_result = sa_partial.optimize(X,y,lmbd,reps=10,n_features=n_features)
        
        sa_mi_heatmap += [sa_mi_result.tolist()]
        sa_full_heatmap += [sa_full_result.tolist()]
        sa_partial_heatmap += [sa_partial_result.tolist()]

        X_mi = X[:,sa_mi_result.astype(bool)]
        X_full = X[:,sa_full_result.astype(bool)]
        X_partial = X[:,sa_partial_result.astype(bool)]
        datasets = [X,X_mi,X_full,X_partial]
        Accuracy_list = []
        CN_list = []
        for i in range(4) :
            dataset = datasets[i]
            Accuracy_list += [bf.get_accuracy(dataset,y,0.9)]
            CN_list += [bf.get_CN(dataset)]
        Accuracy_list_total += [Accuracy_list]
        CN_list_total += [CN_list]
    heatmap = pd.DataFrame(sa_mi_heatmap+sa_full_heatmap+sa_partial_heatmap)

    target = Accuracy_list_total
    Accuracy_result = pd.DataFrame(target).apply(lambda x: str(round(np.mean(x),2)))+"("+pd.DataFrame(target).apply(lambda x: str(round(np.std(x),2)))+")"
    target = CN_list_total
    CN_result = pd.DataFrame(target).apply(lambda x: str(round(np.mean(x),2)))+"("+pd.DataFrame(target).apply(lambda x: str(round(np.std(x),2)))+")"
    result = pd.DataFrame([Accuracy_result,CN_result])
    result.index = ["Accuracy","CN"]
    result.columns = ["Full","MIC","Ord R2","Partial R2"]

    f1, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(heatmap,  linewidths=.5, ax=ax)
    plt.axvline(number_influentials, 0,3,color="red",linewidth=2.5)
    plt.axvline(0, 0, 3,color="red",linewidth=2.5)
    plt.axhline(num_test*3, 0,number_influentials/40,color="red",linewidth=5)
    plt.axhline(num_test*2, 0,number_influentials/40,color="red",linewidth=2.5)
    plt.axhline(num_test*1, 0,number_influentials/40,color="red",linewidth=2.5)
    plt.axhline(0, 0,number_influentials/40,color="red",linewidth=5)

    f2, ax = plt.subplots(figsize=(9, 6))
    plot_df = pd.DataFrame(np.array([np.array([["Full","MIC","Ord R2","Partial R2"] for i in range(num_test)]).reshape(-1),np.array(Accuracy_list_total).reshape(-1)]).T)
    plot_df.columns = ["Methods","Accuracy"]
    plot_df.iloc[:,1] = plot_df.iloc[:,1:2].applymap(float)
    sns.boxplot(data=plot_df,x="Methods",y="Accuracy")
    plt.title("Accuracy of Each meathod")

    f3, ax = plt.subplots(figsize=(9, 6))
    plot_df = pd.DataFrame(np.array([np.array([["Full","MIC","Ord R2","Partial R2"] for i in range(num_test)]).reshape(-1),np.array(CN_list_total).reshape(-1)]).T)
    plot_df.columns = ["Methods","CN"]
    plot_df.iloc[:,1] = plot_df.iloc[:,1:2].applymap(float)
    sns.boxplot(data=plot_df,x="Methods",y="CN")
    plt.title("Conditional Number score of Each meathod")

    acc_result = pd.DataFrame(Accuracy_list_total)
    acc_result.columns = ["Full","MIC","Ord R2","Partial R2"]
    acc_result.index = ["test_" + str(i) for i in range(num_test)]

    cn_result = pd.DataFrame(CN_list_total)
    cn_result.columns = ["Full","MIC","Ord R2","Partial R2"]
    cn_result.index = ["test_" + str(i) for i in range(num_test)]

    result.to_csv(path+"/result.csv")
    acc_result.to_csv(path+"/acc_result.csv")
    cn_result.to_csv(path+"/cn_result.csv")
    heatmap.to_csv(path + "/heatmap.csv")
    f1.savefig(path+"/heatmap.png")
    f2.savefig(path+"/acc_figure.png")
    f3.savefig(path+"/cn_figure.png")

    with open(path+"test_argument.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    
    print(result)
    print(args.__dict__)

