import numpy as np
import math

def return_list(x):
    if not isinstance(x, (list, np.ndarray)):
        print("MEAN: inputs type error")
        return None
    if isinstance(x, np.ndarray):
        mylist = list(x)
    else:
        mylist = x
    if len(mylist) == 0:
        return None
    return mylist

class TinyStatistician(object):

    @staticmethod
    def mean(x):
        if not isinstance(x, (list, np.ndarray)):
            print("MEAN: inputs type error")
            return None
        if isinstance(x, np.ndarray):
            mynp = x
        else:
            mynp = np.array(x)
        return np.mean(mynp)                   

    @staticmethod
    def median(x):
        return TinyStatistician.percentile(x, 50)
               
    @staticmethod
    def quartile(x):
        return tuple((TinyStatistician.percentile(x, 25), TinyStatistician.percentile(x, 75)))

    @staticmethod
    def percentile(x, p): #pas calculer pareil que dans numpy
        if not isinstance(x, (list, np.ndarray)):
            print("PERCENILE ", p, " - inputs type error")
            return None
        if isinstance(x, list):
            mylist = x
        else:
            mylist = list(x)
        mylist.sort()
        index = len(mylist) * (p / 100) - 1
        iceil = math.ceil(index)    
        return mylist[iceil]        
    
    @staticmethod
    def var(x):
        if not isinstance(x, (list, np.ndarray)):
            print("VAR: inputs type error")
            return None
        if isinstance(x, np.ndarray):
            mynp = x
        else:
            mynp = np.array(x)
        return np.var(mynp)       
    
    @staticmethod
    def std(x):
        if not isinstance(x, (list, np.ndarray)):
            print("STD: inputs type error")
            return None
        if isinstance(x, np.ndarray):
            mynp = x
        else:
            mynp = np.array(x)
        return np.std(mynp)     

