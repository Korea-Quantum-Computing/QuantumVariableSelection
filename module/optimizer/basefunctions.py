import numpy as np
from sklearn.utils import check_random_state
import random
import sys

def projection(X):
    X = np.asarray(X)
    return X@np.linalg.inv(X.T@X)@X.T

def get_aic(X, y):
    X = np.asarray(X);y = np.asarray(y)
    n = X.shape[0];p = X.shape[1]
    aic = n*np.log(y.T@(np.identity(n) - projection(X))@y/n) + 2*p
    return aic

def get_bic(X, y):
    X = np.asarray(X);y = np.asarray(y)
    n = X.shape[0];p = X.shape[1]
    bic = n*np.log(y.T@(np.identity(n) - projection(X))@y/n) + p*np.log(n)
    return bic

def get_QB(theta_temp,Q, beta, lmbd):
    Q = np.asarray(Q)
    theta_temp = np.asarray(theta_temp)
    beta = np.asarray(beta)
    res = lmbd*theta_temp.T @ Q @ theta_temp+(1-lmbd)* beta.T @ theta_temp
    return res

def get_QUBO(X: list,y: list, lmbd: float) -> float:
    X = np.asarray(X);y = np.asarray(y)
    Q = np.corrcoef(X.T)
    if type(Q) != np.ndarray : Q = np.array([[Q]])
    beta = get_partial_r2(X,y)
    theta_temp = np.ones((X.shape[1],1))
    res = (lmbd * theta_temp.T @ Q @ theta_temp + (1 - lmbd) * beta.T @ theta_temp)[0][0]
    # try: 
    #     assert type(res) == 'float'
    # except Exception as e:
    #     print('return type should be float!')
    #     return None
    return res

def get_CN(X):
    if X.shape[1] <= 1 : return 1
    X = np.asarray(X)
    eig_list = np.linalg.eig(np.corrcoef(X.T))[0]
    return np.sqrt(np.max(eig_list)/np.min(eig_list))

def R2(y_true,y_pred):
    y_true = np.asarray(y_true);y_pred = np.asarray(y_pred)
    y_mean = np.mean(y_true)
    return 1-np.linalg.norm(y_true-y_pred)/np.linalg.norm(y_true-y_mean)

def get_prediction_R2(X,y,ratio=0.8):
    if ratio == 1.0 :
        beta_coef = np.linalg.inv(X.T@X)@X.T@y
        y_pred = X@beta_coef
        return R2(y,y_pred)  
    n = X.shape[0]
    train_index = random.sample(range(n),int(n*ratio))
    test_index = list(filter(None,np.array([None if i in train_index else i for i in range(n)])))
    X_train = X[train_index,:];y_train = y[train_index].reshape((-1,1))
    X_test = X[test_index,:] ; y_test = y[test_index].reshape((-1,1))
    beta_coef = np.linalg.inv(X_train.T@X_train)@X_train.T@y_train
    y_pred = X_test@beta_coef
    return R2(y_test,y_pred)

def MSE(y_true,y_pred):
    y_true = np.asarray(y_true);y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    return np.linalg.norm(y_true-y_pred)/n

def get_MSPE(X,y,ratio=0.8):
    if ratio == 1.0 :
        beta_coef = np.linalg.inv(X.T@X)@X.T@y
        y_pred = X@beta_coef
        return MSE(y,y_pred)  
    n = X.shape[0]
    train_index = random.sample(range(n),int(n*ratio))
    test_index = list(filter(None,np.array([None if i in train_index else i for i in range(n)])))
    X_train = X[train_index,:];y_train = y[train_index].reshape((-1,1))
    X_test = X[test_index,:] ; y_test = y[test_index].reshape((-1,1))
    beta_coef = np.linalg.inv(X_train.T@X_train)@X_train.T@y_train
    y_pred = X_test@beta_coef
    return MSE(y_test,y_pred)

def get_partial_r2(X,y):
    X = np.asarray(X);y = np.asarray(y).reshape(-1)
    n = X.shape[0];p = X.shape[1]+1
    X_pre = np.concatenate([np.ones((n,1)),X],axis=1)
    SSRF = y.T@(np.identity(n) - projection(X_pre))@y/np.cov(y)
    SSRP = []
    for i in range(1,p):
        index = [j for j in range(p)]
        index = index[:i]+index[i+1:]
        X_temp = X_pre[:,index]
        SSRP += [y.T@(np.identity(n) - projection(X_temp))@y/np.cov(y)]
    result = np.array(SSRP)**(-1)*SSRF
    return 1-result

def get_selecting_qubo(X,y) :
    X = np.asarray(X);y = np.asarray(y)
    Q = np.corrcoef(X.T)
    beta = get_partial_r2(X,y)
    return Q,beta

def generate_dependent_sample(n_samples=500, n_features=10, beta_coef =[4,3,2,2],epsilon=4,covariance_parameter=1, random_state=None):
    rng = check_random_state(random_state)
    if n_features < 4:
        raise ValueError("`n_features` must be >= 4. "
                            "Got n_features={0}".format(n_features))
    v = rng.normal(0, 0.4, (n_features, n_features))
    mean = np.zeros(n_features)
    cov = v @ v.T*covariance_parameter + 0.1 * np.identity(n_features)
    X = rng.multivariate_normal(mean, cov, n_samples)
    n_informative = len(beta_coef)
    beta = np.hstack((
        beta_coef, np.zeros(n_features - n_informative)))
    y = np.dot(X, beta)
    y += epsilon * rng.randn(n_samples)
    return X, y

def get_bin(x, p):
    '''
        선택된 변수의 정수 index를 [01100110..] 방식으로 변환해주는 함수
        input: 정수 index array, 총 변수 개수 p
        output: binary 방식 변수 선택 결과
    '''
    zeros = np.zeros(p, dtype=int)
    zeros[x] = 1
    return zeros

def get_index(theta):
    return(np.where(theta)[0])

def flip(k, x, p):
    '''
        기존 선택된 변수들에서 k개만큼 flip해주는 함수
        input: flip할 횟수 k, 정수 index array, 총 변수 개수 p
        output: 새롭게 선택된 변수 결과
    '''
    zeros = np.zeros(p, dtype=int)
    idx = np.random.choice(p, size = k, replace = False)
    zeros[idx] = 1
    new = abs(x - zeros)
    return new

def flip2(k, x, p):
    '''
    k : 몇 개 뒤집을 것인지, 2의 배수여야 함
    x : 뒤집을 대상
    p : 총 변수 개수
    '''
    x_array = np.asarray(x)
    one = get_index(x_array==1)
    zero = get_index(x_array==0)
    idx_onetozero = np.random.choice(one, size = int(k/2), replace = False).tolist()
    idx_zerotoone = np.random.choice(zero, size = int(k/2), replace = False).tolist()
    x_array[idx_onetozero] = 0
    x_array[idx_zerotoone] = 1
    return(x_array.tolist())


# Generate initial population
def generate_initial_pop(n_sol, n_var): # n_sol: 해의수, 이후 변수 수도 지정가능하게 변경 가능
    initial_pop = np.random.randint(0, 2, (n_sol, n_var)) # 랜덤으로 0과 1 둘중 하나 배정, 각 해는 변수들의 조합
    return initial_pop

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

# Mutation (p의 확률로 0과 1 뒤집기)
def mutation(child, p, n_var):
    for index in range(n_var):
        if np.random.random() < p :
            child[index] = 1 - child[index]
    return child

