import pickle
from typing_extensions import Self
from unittest import result
import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch.nn import Linear
from torch.nn.functional import kl_div
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import copy
class Aligner(torch.nn.Module):
    def __init__(self,N_SOURCE_NODES):
        
        super(Aligner, self).__init__()
        self.layer1 = Linear(N_SOURCE_NODES,N_SOURCE_NODES)
        self.layer2 = Linear(N_SOURCE_NODES,N_SOURCE_NODES//2)
        self.layer3 = Linear(N_SOURCE_NODES//2,N_SOURCE_NODES)

    def forward(self, data):
        x = self.layer1(data)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class AlignerFramework():
    def __init__(self) -> None:
        weights = None
        ids = None
        device = None
        self.epoch = 4000
        self.N_EPOCHS = 1024
        self.originalShapes = []
    def loadData(self, address, ids):
        """
        Loading data from a sended weights objects
        For the debug purpose it can load a data from a snapshoot
        """
        file = open(address, 'rb')
        self.weights = pickle.load(file)
        file.close()
        self.ids = ids
    
    def loadWeightsandIds(self, weights, ids):

        self.weights = weights
        self.ids = ids

    def getKLlistForEachLayer(self, weights, ids=["H1","H2","H3"]):
        """
        It gets
        """
        
        # Parse the data for better readibilty
        # This array will store each hospitals layer-wise weights
        models = [weights[i][0]for i in range(len(weights))]
        
        layerPairs = []
        for k in range(len(models[0])):
            layerWise = []
            for i in range(len(models)):
                pass
                layerWise.append(models[i][k].astype(np.float64))
            layerPairs.append(layerWise)

        
        
        results = []
        # Traverse through each layer
        for layerPair in layerPairs:
            # A function that get layer-wise KL
            result = self.getLayerWiseKL(layerPair)
            results.append(result)
            pass
        
        # It will create a data where it stores the id combination
        clientPair = []
        for i in range(len(ids)):
            for k in range(i+1,len(ids)):
                clientPair.append([ids[i],ids[k]])
        
        return clientPair,np.array(results),layerPairs


    def getLayerWiseKL(self,weights):
        # Run the combination
        results = []
        for i in range(len(weights)):
            for k in range(i+1,len(weights)):
                # we need to run kl
                # Firstly let's check size
                if weights[i].shape != tuple() and weights[k].shape != tuple() and weights[i].shape != (1,) and weights[k].shape !=  (1,) and (weights[k].size < 10000  and weights[i].size < 10000):
                    results.append( self.kl_divergence( self.getDist(weights[i]), self.getDist(weights[k])) )
                    #results.append(self.torchKL(weights[i],weights[k]))
                else:
                    results.append(0)
        return np.array(results)

    def kl(self,p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def torchKL(self,p,q):
        p_torch = torch.from_numpy(p)
        q_torch = torch.from_numpy(q)
        return kl_div(p_torch,q_torch,reduction="batchmean")
    
    def kl_divergence(self,p, q,epsilon=10**-20):
        p += epsilon
        q += epsilon
        return (self.kl(p,q)+self.kl(q,p))/2


    def getDist(self, data_org, epsilon = 10**-14):
        """
        Scipy version for finding PDF
        """
        
        data = data_org.flatten()
        data_abs = np.abs(data)+epsilon
        data_log = np.log(data_abs)
        x_grid_space = np.linspace(-15,15,200)
        density = gaussian_kde(data_log)
        return density(x_grid_space)

    def getDistALL (self, data_org, epsilon = 10**-14):
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
    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def getLayerForKLOLD(self, weights, hospitalPair, topN=5):
        """
        This function will return the array of which layer's KL is higher as index
        Returns [LayerNumber]
        """
        layerNumber = weights.shape[0]
        results = np.zeros((layerNumber,2))
        variance = np.zeros(layerNumber)
        for i in range(layerNumber):
            if weights[i].all() != 0:
                results[i] = (-weights[i]).argsort()[:2]
            else:
                results[i] = np.array([-1,-1])
            variance[i] = np.var(weights[i])
        
        
        # Now let's get 
        topNIndexes = (-variance).argsort()[:topN]
        hospitalPairID = results[topNIndexes].astype(np.int8)

        hospital = []
        for i in hospitalPairID:
            hospital.append(self.intersection(hospitalPair[i[0]], hospitalPair[i[1]]))
        return topNIndexes, hospitalPairID,hospital
    

    def getLayerForKL(self, weights,klList, hospitalPair, topN=5):
        # Create big matrix
        models = [weights[i][0]for i in range(len(weights))]

        layerNumber = klList.shape[0]
        results = np.zeros((layerNumber,2))
        variance = np.zeros(layerNumber)
        for i in range(layerNumber):
            if klList[i].all() != 0:
                results[i] = (-klList[i]).argsort()[:2]
            else:
                results[i] = np.array([-1,-1])
            variance[i] = np.var(klList[i])

        # Now let's get 
        topNIndexes = (-variance).argsort()[:topN]

        distance = []
        for i, hospitalWeights in enumerate(models):
            distance.append(
                self.getDistALL(hospitalWeights)
            )
            pass
        
        results = []
        for i in range(len(distance)):
            for k in range(i+1, len(distance)):
                results.append(
                    self.kl_divergence(distance[i],distance[k])
                )
        
        firstMax = hospitalPair[results.index(max(results))]
        results[results.index(max(results))] = -1
        secondMax = hospitalPair[results.index(max(results))]

        hospital = self.intersection(firstMax,secondMax)
        hospitals = []
        for i in range(len(topNIndexes)):
            hospitals.append(hospital)
        return topNIndexes,None , hospitals

    def getLayersForAligment(self, weights, topNIndexes ,ids, hospitalList):
        
        result = []
        target = []
        
        for i in range(len(topNIndexes)):
            hospitalNumber = len(weights[topNIndexes[i]])
            index = ids.index(hospitalList[i][0])
            result.append(weights[topNIndexes[i]][index])
            
            targetLayer = np.zeros(shape=weights[topNIndexes[i]][0].shape)
            for k in range(hospitalNumber):
                if k != index:
                    targetLayer += weights[topNIndexes[i]][k]
            targetLayer /= (hospitalNumber-1)
            target.append(targetLayer)
        return result,target
    
    
    def klTorch(self, p_32, q_32):
        p = p_32.double()
        q = q_32.double()
        return torch.sum(torch.where(p != 0, p * torch.log(p / q), 0.))
        #return kl_div(p,q,reduction="batchmean")
    def data_log(self, data_org, epsilon = 10**-15):
        data = data_org.flatten()
        data_abs = torch.abs(data)+epsilon
        data_log = torch.log(data_abs)
        return data_log
    
    def kl_divergenceLoss(self,p, q,epsilon=10**-15):
        p = self.data_log(p)
        q = self.data_log(q)
        p += epsilon
        q += epsilon

        
        # Distance
        return (self.klTorch(p,q)+self.klTorch(q,p))/2
    
    def getDistanceTorchVersion(self, p):
        return torch.histc(p,bins=10)


    def plotter(self, output, target, beforeAligment):
        plt.hist(output.detach().numpy().flatten(),bins=10, alpha=0.5)
        plt.hist(target.flatten(),bins=10, alpha=0.5)
        plt.hist(beforeAligment.flatten(),bins=10, alpha=0.5)
        plt.show()

    def trainForSingleLayer(self,source, target):
        
        if torch.cuda.is_available() and 1:
            self.device = torch.device("cuda:0")
            print("Running on cuda:0")
        else:
            self.device = torch.device("cpu")
            print("Running on cpu")
        
        # Create the model for this
        aligner = Aligner(source.flatten().shape[0]).to(self.device)

        # Get store the original shape
        self.originalShapes.append(source.shape)

        # Set train mode
        aligner.train()

        # Optimizer for aligment
        optimizer = torch.optim.SGD(aligner.parameters(), lr=1e-4)
        #optimizer = torch.optim.Adam(aligner.parameters(), lr=0.001, betas=(0.5, 0.999))


        # Data pre-processing before go into traning
        data = source.astype(np.float32).flatten()
        dataSet = np.array([data])
        dataSource = torch.from_numpy(dataSet).to(self.device)
        target = torch.from_numpy(target.astype(np.float32).flatten()).to(self.device)

        # Model list
        model_list = []
        with tqdm(total = self.N_EPOCHS) as progressbar:
            for epochs in range(self.N_EPOCHS):
                with torch.autograd.set_detect_anomaly(True):
                    # Get the result from model
                    A_output = aligner(dataSource)

                    # Copy the current model into a list 
                    tempModel = copy.deepcopy(aligner)
                    model_list.append(tempModel)

                    # If list is bigger than 3 delete the old one
                    if len(model_list) > 3:
                        del model_list[0]
                        
                    loss = torch.abs(self.kl_divergenceLoss(A_output,target))*10000

                    #print(loss.item())
                    # Loss Traning
                    if torch.isnan(loss) == True:
                        if len(model_list) < 2 or 1:
                            A_output = dataSource
                            print("Loss produced Nan in the first step-", "Passed")
                        else:

                            A_output = model_list[1](dataSource)
                            print("Nan detected using previously trained model")
                            print("Prev Loss: ", torch.abs(self.kl_divergenceLoss(A_output,target)))
                        break
                    else:
                        # Loss Traning
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    progressbar.update(1)
        
        print("Loss: ", torch.abs(self.kl_divergenceLoss(A_output,target)))
        
        return A_output

    def reintegrateWeights(self, results, topNIndexes,hospital):
        for i in range(len(results)):
            hospitalId = self.ids.index(hospital[i][0])
            self.weights[hospitalId][0][topNIndexes[i]] = np.reshape(np.squeeze(results[i].detach().cpu().numpy()), newshape=self.originalShapes[i])
        pass

    def reintegrateWeightsNEW(self, results, topNIndexes,hospital):
        for i in range(len(results)):
            hospitalId = self.ids.index(hospital[i][0])
            self.weights[hospitalId][0][topNIndexes[i]] = (np.reshape(np.squeeze(results[i].detach().cpu().numpy()), newshape=self.originalShapes[i]) - self.weights[hospitalId][0][topNIndexes[i]])*0.3 + self.weights[hospitalId][0][topNIndexes[i]]
        pass

    def saveResults(self, output, target, before, fileID):
        item = (output, target,before)
        fileName = f"Results/id_{fileID}.obj"
        filehandler = open(fileName, 'wb') 
        pickle.dump(item, filehandler)
        pass

    def run(self, topN=5):
        "Allaha emanet bir fonksiyon"
        
        print("Calculating KLs for each layer")
        hospitalPair, klList,layerList  = self.getKLlistForEachLayer(self.weights, self.ids)
        topNIndexes,hospitalPairID,  hospital = self.getLayerForKL(self.weights, klList,topN=topN, hospitalPair=hospitalPair)
        print("Detecded Hospital: ", hospital)
        print("Layer Indexes: ", topNIndexes)
        aligmentReadyLayers, targetLayers  = self.getLayersForAligment(layerList,topNIndexes,self.ids, hospital)
        
        print("Traning")
        results = []
        self.originalShapes = []
        for i in range(topN):
            A_output = self.trainForSingleLayer(aligmentReadyLayers[i],targetLayers[i])
            results.append(A_output)
            #self.plotter(A_output,targetLayers[i],aligmentReadyLayers[i])
            self.saveResults(A_output,targetLayers[i],aligmentReadyLayers[i], fileID=i)
        pass

        print("Results are submitted")
        # Reintegrate weights
        exit()
        self.reintegrateWeights(results, topNIndexes,hospital)
        return self.weights
    

class AlignerV2Original(AlignerFramework):
    def kl_divergenceLoss(self, p_, q_):

        p = p_.flatten().clone()
        q = q_.flatten().clone()

        minValue = 3
        p += minValue
        q += minValue 


        p = self.data_log(p)
        q = self.data_log(q)

                
        return (self.klTorch(p,q)+self.klTorch(q,p))/2

    def data_log(self, data_org):
        data_log = torch.log(data_org)
        return data_log

    
    
    
