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
print("***** VAR :" , stat.var(a))
print("***** STD :" , stat.std(a))

b = np.array(a)
print(" NP array = b = ", b)
print("***** MEAN :" , stat.mean(b))
print("***** MEDIAN :" , stat.median(b))
print("***** QUARTILE :" , stat.quartile(b))
print("***** PERC 10 :" , stat.percentile(b,10))
print("***** PERC 28 :" , stat.percentile(b, 28))
print("***** VAR :" , stat.var(b))
print("***** STD :" , stat.std(b))
