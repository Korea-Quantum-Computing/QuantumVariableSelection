'''
    구현함수에서 dataframe의 연산등이 셖여서 class로도 최적화가 어렵다.
    modele로 일괄실행 이후에 generate 함수에서 묶어주자

    GA 의 실행결과 가져지는 3가지의 값을 저장해주는 class 만 생성해 주자. 
'''
# GA 실행 시 필요한 패키지
import kqc_custom
from qiskit import Aer,IBMQ
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.visualization import plot_histogram
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate initial population
def generate_initial_pop(n_sol, n_var): # n_sol: 해의수, 이후 변수 수도 지정가능하게 변경 가능
    initial_pop = np.random.randint(0, 2, (n_sol, n_var)) # 랜덤으로 0과 1 둘중 하나 배정, 각 해는 변수들의 조합
    return initial_pop

# Objective function 
def get_aic(x, y):
    reg = LinearRegression().fit(x, y)
    prediction = reg.predict(x)
    residual = y - prediction

    N = len(x)
    s = len(x.columns)
    AIC = N*np.log(sum(residual**2)/N) + 2*(s + 2)

    return AIC

# Selection (roulette wheel version)
def roulette_wheel_selection(fitness_value_list): # finess_value
    
    fitness_sum = sum(fitness_value_list)
    prob_list = list(map(lambda x: x / fitness_sum, fitness_value_list))
    cum_prob_list = np.cumsum(prob_list)
    rand_value = np.random.rand()
    index = np.argwhere(rand_value <= cum_prob_list)
    
    return index[0][0]

# Crossover
def crossover(parents1, parents2, n_var):
    crossover_point = np.random.randint(0,n_var) # 처음과 마지막은 바뀌는 의미가 없으므로 제외
    
    child = np.empty(n_var)
    
    child[:crossover_point] = parents1[:crossover_point]
    child[crossover_point:] = parents2[crossover_point:]
    
    
    return child

# Nutation (p의 확률로 0과 1 뒤집기)
def mutation(child, p, n_var):
    for index in range(n_var):
        if np.random.random() < p :
            child[index] = 1 - child[index]
    return child

# Genetic Algorithm
def GA(X_data, y_data, n_gen=30, n_sol=20, n_par=10, mutation_sol_p=0.02, mutation_gene_p=0.02):
    
    n_var = len(X_data.columns)   # 변수 수
#     n_gen = 30    # 세대 수
#     n_sol = 20    # 한 세대에 포함되는 해의 개수
#     n_par = 10    # 부모개수
#     mutation_sol_p = 0.02   # 유전자가 돌연변이일 확률
#     mutation_gene_p = 0.02  # 유전 개체가 돌연변이일 확률

    # 초기해 생성
    current_population = generate_initial_pop(n_sol, n_var)

    # scroe 초기화
    best_score = np.inf

    # 최적해 리스트
    best_solution_list = []

    # 평균 fitness_value 리스트
    solution_mean_list = []

    # 전체 로그 기록할 목록
    result_log = []
    
    # 세대 수 조건만큼 반복
    for generation in range(n_gen):
        str_current_population = ', '.join(map(str,current_population))
        result_log.append(str_current_population)


        fitness_value_list = np.array([get_aic( X_data.loc[:, chromosome.astype(bool)], y_data) for chromosome in current_population])

        solution_mean_list.append(np.mean(fitness_value_list))

        if fitness_value_list.min() < best_score:
            best_score = fitness_value_list.min()
            best_solution = current_population[fitness_value_list.argmin()]

        best_solution_list.append(best_score)

        # 적합 크기대로 해들 나열
        parents = current_population[np.argsort(fitness_value_list)]
        
        # 이후 세대에 그대로 보낼 부모 조합 정하기
        new_population = parents[:n_par]

        # 선택, 교차, 돌연변이
        childs = []

        for _ in range(n_sol - n_par):
            parent_1 = current_population[roulette_wheel_selection(fitness_value_list)]
            parent_2 = current_population[roulette_wheel_selection(fitness_value_list)]

            child = crossover(parent_1, parent_2, n_var)

            if np.random.random() < mutation_sol_p:

                child = mutation(child, mutation_gene_p, n_var)

            childs.append(child)

        current_population = np.vstack([new_population, childs])

    # print("best_score:", best_score, "best_solution: ", best_solution)
    
    # 그래프 출력하기

    # plt.plot(solution_mean_list, c='red')
    # plt.plot(best_solution_list, c = 'blue')
    # plt.xlabel('Generation')
    # plt.ylabel('fitness value')
    # plt.legend(['solution_mean', 'best_solution'])
    # plt.show()
    
    return best_score, best_solution, result_log # 최적 해 반환하기
