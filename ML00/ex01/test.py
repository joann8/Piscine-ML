from TinyStatistician import TinyStatistician
import numpy as np

a = [1, 42, 300, 10, 59]
stat = TinyStatistician()

print(" List = ", a)
print("***** MEAN :" , stat.mean(a))
print("***** MEDIAN :" , stat.median(a))
print("***** QUARTILE :" , stat.quartile(a))
print("***** PERC 10 :" , stat.percentile(a,10))
print("***** PERC 28 :" , stat.percentile(a, 28))
print("***** PERC 83 :" , stat.percentile(a, 83))
print("***** VAR :" , stat.var(a))
print("***** STD :" , stat.std(a))

# pas sure a mettre dans tests finaux
b = np.array((1, 2, 3, 4, 5, 6 , 7 , 8))
print(" NP array = b = ", b)
print("***** MEAN :" , stat.mean(b))
print("***** MEDIAN :" , stat.median(b))
print("***** QUARTILE :" , stat.quartile(b))
print("***** PERC 10 :" , stat.percentile(b,10))
print("***** PERC 28 :" , stat.percentile(b, 28))
print("***** PERC 83 :" , stat.percentile(b, 83))
print("***** VAR :" , stat.var(b))
print("***** STD :" , stat.std(b))
