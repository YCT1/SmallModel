import numpy as np
from scipy.stats import gaussian_kde
class KLPrinter():
    def __init__(self) -> None:
        pass

    def __getDist(self, data_org, epsilon = 10**-14):
        """
        Scipy version for finding PDF
        """
        data = data_org[0].flatten()

        for i in range(1, len(data_org)):
            data = np.concatenate([data,data_org[i].flatten()])

        data_abs = np.abs(data)+epsilon
        data_log = np.log(data_abs)
        x_grid_space = np.linspace(-15,15,200)
        density = gaussian_kde(data_log)
        return density(x_grid_space)
    
    def __kl(self,p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def __kl_divergence(self,p, q,epsilon=10**-20):
        p += epsilon
        q += epsilon
        return (self.__kl(p,q)+self.__kl(q,p))/2

    def printKLDivergences(self, weights, ids):
        weightDistrubutionList = []
        for data in weights:
            weight = data
            weightDistrubution = self.__getDist(weight)
            weightDistrubutionList.append(weightDistrubution)
        
        for i in range(len(weightDistrubutionList)):
            for k in range(i+1,len(weightDistrubutionList)):
                line = "KL: " + "<" + ids[i] +" : "+ ids[k] + ">" + str(self.__kl_divergence(weightDistrubutionList[i]  , weightDistrubutionList[k] ))
                print(line)
