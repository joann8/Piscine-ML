import numpy as np

class TinyStatistician(object):

    @staticmethod
    def mean(x):
        if isinstance(x, np.ndarray):
            mean = 0.0
            for i in x:
                if isinstance(i, (float, int)):
                    mean += i
                else:
                    print(i)
                    print("MEAN - NP: inputs type error")
                    return None
            return mean / len(x)     
        elif isinstance(x, list):
            mean = 0.0
            for i in x:
                if isinstance(i, (float, int)):
                    mean += i
                else:
                    print("MEAN - LIST: inputs type error")
                    return None
            return mean / len(x)               
        else:
            print("MEAN: inputs type error")
            return None

    @staticmethod
    def median(x):
        pass
        
    @staticmethod
    def quartile(x):
        pass

    @staticmethod
    def percentile(x, p):
        pass
    
    @staticmethod
    def var(x):
        pass

    @staticmethod
    def std(x):
        pass

