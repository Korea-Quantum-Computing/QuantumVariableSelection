➜  QuantumVariableSelection git:(linear_simulation) python3 experiment/2.simulat
ion.py -p 20 -q 8 -iter 10 -lmbd 0.15 -cov 12
                 Original          SA_AIC           GA_AIC          SA_QUBO         GA_QUBO
AIC_list   5212.62(35.85)  5196.25(36.24)  5851.58(412.82)  6479.33(479.72)  6798.33(541.4)
QUBO_list       4.41(0.6)      6.07(0.24)       5.34(0.85)        3.3(0.61)      3.25(0.94)
MSPE           0.96(0.04)      0.94(0.05)       1.34(0.34)       1.83(0.43)      2.25(0.53)
R2             0.84(0.02)      0.84(0.02)       0.77(0.03)        0.7(0.06)       0.62(0.1)
CN            27.49(7.13)      2.94(0.53)       6.25(2.65)       3.64(1.07)      3.41(0.87)
{'number_samples': 1000, 'number_features': 20, 'number_influentials': 8, 'epsilon': None, 'covariance_parameter': 12.0, 'lmbd': 0.15, 'number_of_test': 10}
➜  QuantumVariableSelection git:(linear_simulation) ✗ python3 experiment/2.simulation.py -p 20 -q 8 -iter 10 -lmbd 0.10 -cov 12
                Original          SA_AIC          GA_AIC         SA_QUBO          GA_QUBO
AIC_list   5185.34(46.8)  5169.97(45.83)  6046.43(522.9)  5978.21(719.3)  6398.95(434.37)
QUBO_list     4.02(0.81)      5.99(0.45)       4.85(0.6)      4.19(1.03)        3.9(0.68)
MSPE          0.95(0.06)      0.96(0.04)      1.53(0.39)      1.51(0.59)       1.79(0.39)
R2            0.86(0.02)      0.86(0.02)      0.78(0.05)      0.77(0.09)       0.74(0.08)
CN           27.97(7.74)       3.63(1.0)       5.59(1.9)      3.85(1.22)       4.39(0.78)
{'number_samples': 1000, 'number_features': 20, 'number_influentials': 8, 'epsilon': None, 'covariance_parameter': 12.0, 'lmbd': 0.1, 'number_of_test': 10}
➜  QuantumVariableSelection git:(linear_simulation) ✗ python3 experiment/2.simulation.py -p 20 -q 8 -iter 10 -lmbd 0.05 -cov 12
                 Original          SA_AIC           GA_AIC         SA_QUBO         GA_QUBO
AIC_list   5213.82(31.95)  5199.09(31.85)  5580.62(438.36)  5369.25(202.7)  5838.2(451.53)
QUBO_list       3.15(0.7)      5.41(0.67)       4.34(1.02)      4.85(0.74)      4.31(0.87)
MSPE           0.98(0.04)      0.96(0.04)        1.19(0.3)      1.05(0.13)      1.34(0.29)
R2             0.86(0.02)      0.86(0.02)       0.81(0.05)      0.84(0.03)      0.79(0.06)
CN             26.81(5.0)      3.81(0.95)       7.54(2.53)      4.37(0.94)      5.47(1.93)
{'number_samples': 1000, 'number_features': 20, 'number_influentials': 8, 'epsilon': None, 'covariance_parameter': 12.0, 'lmbd': 0.05, 'number_of_test': 10}
➜  QuantumVariableSelection git:(linear_simulation) ✗ python3 experiment/2.simulation.py -p 20 -q 8 -iter 10 -lmbd 0.15 -cov 40
                Original         SA_AIC          GA_AIC          SA_QUBO           GA_QUBO
AIC_list   5203.5(26.82)  5188.34(26.0)  6474.3(714.16)  5942.48(898.23)  7901.53(1108.54)
QUBO_list     5.29(1.04)     6.61(0.47)       5.5(1.15)       5.11(1.17)        2.99(1.74)
MSPE          0.94(0.02)     0.97(0.04)       1.9(0.69)        1.57(1.0)         4.2(1.67)
R2             0.9(0.02)      0.9(0.02)       0.8(0.06)       0.84(0.09)        0.55(0.21)
CN          37.67(14.47)     4.01(0.88)       5.34(2.1)       3.81(0.93)        3.71(1.06)
{'number_samples': 1000, 'number_features': 20, 'number_influentials': 8, 'epsilon': None, 'covariance_parameter': 40.0, 'lmbd': 0.15, 'number_of_test': 10}
➜  QuantumVariableSelection git:(linear_simulation) ✗ python3 experiment/2.simulation.py -p 20 -q 8 -iter 10 -lmbd 0.10 -cov 40
                 Original          SA_AIC          GA_AIC          SA_QUBO         GA_QUBO
AIC_list   5205.91(39.62)  5190.43(39.69)  6263.1(972.32)  5974.7(1001.32)  7613.77(630.2)
QUBO_list      5.34(0.83)      7.05(0.39)      5.97(1.02)        5.6(1.37)      3.67(1.07)
MSPE           0.96(0.04)      0.96(0.04)       1.9(1.13)       1.61(1.08)      3.33(1.23)
R2             0.91(0.02)      0.91(0.02)      0.83(0.09)       0.86(0.07)       0.69(0.1)
CN           36.53(13.42)      3.73(1.46)      6.54(2.16)       3.65(1.03)      4.18(1.49)
{'number_samples': 1000, 'number_features': 20, 'number_influentials': 8, 'epsilon': None, 'covariance_parameter': 40.0, 'lmbd': 0.1, 'number_of_test': 10}
➜  QuantumVariableSelection git:(linear_simulation) ✗ python3 experiment/2.simulation.py -p 20 -q 8 -iter 10 -lmbd 0.05 -cov 40
                 Original          SA_AIC           GA_AIC          SA_QUBO          GA_QUBO
AIC_list   5231.75(34.55)  5215.54(34.32)  6046.42(728.69)  5716.88(764.26)  7300.68(982.74)
QUBO_list      3.36(0.79)      6.48(0.48)       5.41(1.33)       5.63(1.07)       3.47(1.52)
MSPE           0.98(0.05)      0.97(0.06)        1.56(0.6)       1.34(0.67)       3.01(1.25)
R2             0.91(0.01)      0.91(0.01)       0.85(0.04)       0.87(0.07)       0.71(0.12)
CN           52.25(13.29)      4.79(3.11)        7.16(3.3)       4.74(1.15)       4.96(2.28)
{'number_samples': 1000, 'number_features': 20, 'number_influentials': 8, 'epsilon': None, 'covariance_parameter': 40.0, 'lmbd': 0.05, 'number_of_test': 10}