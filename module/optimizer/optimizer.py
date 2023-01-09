import sys

module_path = "/Users/minhyeong-gyu/Documents/GitHub/QAOA_realestate/Python"
if module_path not in sys.path:
    sys.path.append(module_path)


from dimod import ConstrainedQuadraticModel, Integer, QuadraticModel
import pandas as pd
import numpy as np
from optimizer import basefunctions as bf

import warnings

class QuantumAnnealing:
    def __init__(self,sampler):
        self.sampler = sampler

    def optimize(self,
                X,y,lamda=0.5
                ):
        X = np.asarray(X); y= np.asarray(y)
        Q,beta = bf.get_selecting_qubo(X,y)
        
        p = len(Q)
        integer_list = []
        for i in range(p) :
            integer_list += [Integer(str("x")+str(i).zfill(3), upper_bound=1,lower_bound=0)]
        linear_qubo = QuadraticModel()
        for i in range(p): 
            linear_qubo += (1-lamda)*beta[i]*integer_list[i]    
        quadratic_qubo = QuadraticModel()
        for j in range(p):
            for i in range(p):
                quadratic_qubo += lamda*Q[i][j]*integer_list[i]*integer_list[j]
        
        Qubo = linear_qubo + quadratic_qubo
        cqm = ConstrainedQuadraticModel()
        cqm.set_objective(Qubo)
        
        self.sampleset = self.sampler.sample_cqm(cqm).record
        sample_true = self.sampleset[[self.sampleset[i][4] for i in range(len(self.sampleset))]]
        self.result = sample_true[[sample_true[i][1] for i in range(len(sample_true))] == np.min([sample_true[i][1] for i in range(len(sample_true))])][0][0].tolist()
        return self.result

class SimulatedAnnealing:
    def __init__(self,mode="AIC",
                schedule_list = [100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 300, 400, 400, 400, 400, 400, 400],
                k_flip=2,
                alpha=0.9,
                tau=1,
                reps=1
                ):
        self.schedule_list, self.k_flip, self.alpha, self.tau, self.reps = schedule_list, k_flip, alpha, tau, reps
        self.mode=mode
    
    def _optimize_aic(self,
                X,y,lamda
                ):
        schedule_list, k_flip, alpha, tau = self.schedule_list, self.k_flip, self.alpha, self.tau
        theta_list = []
        X = np.asarray(X);y=np.asarray(y)
        n = X.shape[0];p = X.shape[1]

        theta_temp = np.random.randint(2,size=p)
        for j in schedule_list:
            for m in range(j):
                theta_star = bf.flip(k_flip, theta_temp, p)
                X_star = X[:,theta_star.astype(bool)]
                X_temp = X[:,theta_temp.astype(bool)]
                comparison = (bf.get_aic(X_temp,y)-bf.get_aic(X_star,y))/tau/n
                if comparison > 100 : 
                    theta_temp = theta_star
                else :
                    if np.random.rand(1) <= min(1, np.exp(comparison)):
                        theta_temp = theta_star
                theta_list += [theta_temp]
                tau = alpha * tau
        
        result = theta_temp
        self.result = (1.0*np.asarray(result)).tolist()
        self.theta_list = theta_list
        return self.result

    def _optimize_qubo(self,
                X,y,lamda
                ):
        schedule_list, k_flip, alpha, tau = self.schedule_list, self.k_flip, self.alpha, self.tau
        theta_list = []
        X = np.asarray(X);y=np.asarray(y)
        p = X.shape[1]
        Q,beta = bf.get_selecting_qubo(X,y)

        theta_temp = np.random.randint(2,size=p)
        for j in schedule_list:
            for m in range(j):
                theta_star = bf.flip(k_flip, theta_temp, p)
                comparison = (bf.get_QB(theta_temp,Q,-1*beta,lamda)-bf.get_QB(theta_star,Q,-1*beta,lamda))/tau
                if comparison > 100 :
                    theta_temp = theta_star
                else:
                    if np.random.rand(1) <= min(1, np.exp(comparison)):
                        theta_temp = theta_star
                theta_list += [theta_temp]
                tau = alpha * tau
        
        result = theta_temp
        self.result = (1.0*np.asarray(result)).tolist()
        self.theta_list = theta_list
        return self.result

    def optimize(self,
                X,y,lamda
                ,reps=10):
        if self.mode == "AIC" :
            optimizer = self._optimize_aic
            obj = bf.get_aic
        elif self.mode == "QUBO" :
            optimizer = self._optimize_qubo
            obj = bf.get_QB
    
        X = np.asarray(X);y=np.asarray(y)
        theta_list = []
        for i in range(reps):
            theta_list += [optimizer(X,y,lamda)]
        if self.mode == "AIC":
            score_list = [obj(X[:,np.array(theta_list[i]).astype(bool)],y) for i in range(reps)]
        if self.mode == "QUBO" :
            Q,beta = bf.get_selecting_qubo(X,y)
            score_list = [obj(theta_list[i],Q,beta,lamda) for i in range(reps)]
        sampleset = pd.DataFrame(theta_list)
        sampleset["score"] = score_list
        self.sampleset = sampleset
        self.result = np.asarray(theta_list)[score_list == np.min(score_list)][0]
        return self.result
    


class GeneticAlgorithm:
    def __init__(self,
                mode = "AIC",
                n_gen=100, 
                n_sol=40, 
                n_par=20, 
                mutation_sol_p=0.02, 
                mutation_gene_p=0.02
                ):
        self.n_gen = n_gen
        self.n_sol = n_sol
        self.n_par = n_par
        self.mutation_sol_p = 0.02
        self.muation_gene_p = 0.02
        self.mode = mode
    def optimize(self,
                X,y,lamda
                ):
        n_gen,n_sol,n_par,mutation_sol_p,mutation_gene_p = self.n_gen,self.n_sol,self.n_par,self.mutation_sol_p,self.muation_gene_p
        X = np.asarray(X);y = np.asarray(y)
        n_var = X.shape[1]
        current_population = bf.generate_initial_pop(n_sol, n_var)
        best_score = np.inf
        
        for generation in range(n_gen):
            if self.mode == "AIC":
                fitness_value_list = np.array([bf.get_aic( X[:, chromosome.astype(bool)], y) for chromosome in current_population])
            if self.mode == "QUBO":
                Q,beta = bf.get_selecting_qubo(X,y)
                fitness_value_list = np.array([bf.get_QB(chromosome,Q,-1*beta,lamda) for chromosome in current_population])                
            
            if fitness_value_list.min() < best_score:
                best_score = fitness_value_list.min()
                best_solution = current_population[fitness_value_list.argmin()]
            parents = current_population[np.argsort(fitness_value_list)]
            new_population = parents[:n_par]

            childs = []
            for _ in range(n_sol - n_par):
                parent_1 = current_population[bf.roulette_wheel_selection(fitness_value_list)]
                parent_2 = current_population[bf.roulette_wheel_selection(fitness_value_list)]
                child = bf.crossover(parent_1, parent_2, n_var)
                if np.random.random() < mutation_sol_p:
                    child = bf.mutation(child, mutation_gene_p, n_var)
                childs.append(child)
            current_population = np.vstack([new_population, childs])
            return best_solution