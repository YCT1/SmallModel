from multiprocessing import Process
from model import SuperReso
import numpy as np
from sklearn.model_selection import KFold
import torch
import pandas as pd
import os
from aligner import AlignerV2Original
import copy
from KLPrinter import KLPrinter
class Hosptital():
    def __init__(self, GPUID, id) -> None:
        self.model = SuperReso(GPUID=GPUID)
        self.id = id
        pass

    def train(self,train_source, train_target, epoch):
        self.model.train(train_source, train_target, epoch)
        pass

    def eval(self,expName, test_source, test_target):
        result, all_results = self.model.test(test_source, test_target)
        print(f"Hospital ID: {self.id}, Result: {result}")
        
        # Experiment Name, Hospital ID, Result
        with open(f'Results/{expName}/h_{self.id}.txt', 'w') as f:
            f.write( str(result))
        
        df = pd.DataFrame(all_results)
        df.to_csv(f'Results/{expName}/h_{self.id}.csv',index=None, header=None)


    def get_parameters(self):
        return self.model.get_parameters()

    def set_parameters(self,parameters):
        self.model.set_parameters(parameters)


class FedL():
    def __init__(self,name) -> None:
        self.expName = name


        if not os.path.exists(f"Results/{name}"):
        
        # Create a new directory because it does not exist 
            os.makedirs(f"Results/{name}")
        pass

    def createHospital(self, numberOfHospital):
        
        hospitals = []
        for i in range(numberOfHospital):
            # Set CUDA value
            newHosptital = Hosptital(i+1, i+1)
            hospitals.append(newHosptital)
        
        return hospitals
        
    
    def trainEachHospital(self, traningData, hospitalList,epoch):
        print("Traning Each Hospital")
        threads = list()
        for i, hospital in enumerate(hospitalList):
            x = Process(target=hospital.train,args=(traningData[i][0], traningData[i][1], epoch))
            threads.append(x)
            x.start()
        for thread in threads:
            thread.join()

    def getWeights(self, hospitalList, forAligner=False):
        weights = []
        for hospital in hospitalList:
            if forAligner:
                weights.append( [hospital.get_parameters(),50])
            else:
                weights.append(hospital.get_parameters())
        return weights
   
    def tranferWeightsStructure(self, weights):
        models = [weights[i][0]for i in range(len(weights))]
        return models
    def aggregation(self, weights):
        print("Aggregation")
        newModel = []
        for i in range(len(weights[0])):
            a = np.zeros(shape=weights[0][i].shape)


            for hospital in weights:
                a += hospital[i]

            a = a / len(weights)
            newModel.append(a)
            pass

        return newModel

    def setWeights(self,hospitalList,newWeights):
        for hospital in hospitalList:
            hospital.set_parameters(newWeights)

    def eval(self, hospitalList,test_source,test_target):
        print("Evaluation...")
        threads = list()
        for i, hospital in enumerate(hospitalList):
            x = Process(target=hospital.eval,args=(self.expName,test_source,test_target))
            threads.append(x)
            x.start()
        
        for i, thread in enumerate(threads):
            thread.join()

    def run(self, training, testing, epoch=20, cycle=5):


        # Create Hospitals
        hospitalList = self.createHospital(3)

        for i in range(cycle):
            # Train hospital
            print(f"CYCLE : {i}")
            self.trainEachHospital(training,hospitalList,epoch)
            
            # Get weights
            weights = self.getWeights(hospitalList)
            newWeights = self.aggregation(weights)

            # Set weights
            self.setWeights(hospitalList, newWeights)


        # Eval
        self.eval(hospitalList, testing[0], testing[1])
        pass
    
    def woAggRun(self, training, testing, epoch):
        # Create Hospitals
        hospitalList = self.createHospital(3)
        self.trainEachHospital(training,hospitalList,epoch)
        self.eval(hospitalList, testing[0], testing[1])
        pass
    
    def runWithAlignment(self, training, testing, epoch=20, cycle=5, topN=1):
        # Create Hospitals
        hospitalList = self.createHospital(3)
        aligner = AlignerV2Original()
        for i in range(cycle):
            # Train hospital
            print(f"CYCLE : {i}")
            self.trainEachHospital(training,hospitalList,epoch)
            
            # Get weights
            weights = self.getWeights(hospitalList, forAligner=True)

            # Aligner Operations
            
            original_weights = self.tranferWeightsStructure(copy.deepcopy(weights))
            aligner.loadWeightsandIds(weights=weights, ids=["H1","H2","H3"])
            newWeights = aligner.run(topN=topN)
            newWeights = self.tranferWeightsStructure(newWeights)
            newWeightsAggregated = self.aggregation(newWeights)
            self.setWeights(hospitalList, newWeightsAggregated)

            kl = KLPrinter()
            print("Before Aligment")
            kl.printKLDivergences(original_weights,ids=["H1","H2","H3"])

            print("After Aligment")
            kl.printKLDivergences(newWeights,ids=["H1","H2","H3"])

        # Eval
        self.eval(hospitalList, testing[0], testing[1])

def createData():
    # Set seed
    np.random.seed(10)

    # Creating source and target not OOD
    source_cluster1 = np.random.normal(0,0.5, (500,35)).astype(np.float32)
    target_cluster1 = np.random.normal(0,0.5, (500,160)).astype(np.float32)

    # Creating source and target OOD
    source_cluster2 = np.random.normal(1,0.5,(500,35)).astype(np.float32)
    target_cluster2 = np.random.normal(1,0.5,(500,160)).astype(np.float32)

    # Concanate them
    source_all = np.concatenate((source_cluster1,source_cluster2))
    target_all = np.concatenate((target_cluster1,target_cluster2))

    # Prepara the data
    data_index = np.arange(0,source_cluster1.shape[0])

    normal_index = np.random.choice(data_index,400,replace=False)
    h1_index, h2_index = normal_index[0:200],normal_index[200:400]

    h1_source,h1_target = source_cluster1[h1_index], target_cluster1[h1_index]
    h2_source,h2_target = source_cluster1[h2_index], target_cluster1[h2_index]

    ood_index = np.random.choice(data_index,200,replace=False)
    h3_source, h3_target = source_cluster2[ood_index], target_cluster2[ood_index]

    all_data_index = np.arange(0,source_all.shape[0])
    testing_index = np.random.choice(all_data_index,100,replace=False)
    testing_source, testing_target = source_all[testing_index], target_all[testing_index]

    testing_fold1 = (testing_source, testing_target )
    traning_fold1 = [
        [h1_source,h1_target],
        [h2_source,h2_target],
        [h3_source,h3_target]
    ]


    all_data_index = np.arange(0,source_all.shape[0])
    testing_index = np.random.choice(all_data_index,100,replace=False)
    testing_source, testing_target = source_all[testing_index], target_all[testing_index]
    testing_fold2 = (testing_source, testing_target )

    # Create fold 2 Traning Data
    data_index = np.arange(0,source_cluster2.shape[0])

    normal_index = np.random.choice(data_index,400,replace=False)
    h1_index, h2_index = normal_index[0:200],normal_index[200:400]

    h1_source,h1_target = source_cluster2[h1_index], target_cluster2[h1_index]
    h2_source,h2_target = source_cluster2[h2_index], target_cluster2[h2_index]

    ood_index = np.random.choice(data_index,200,replace=False)
    h3_source, h3_target = source_cluster1[ood_index], target_cluster1[ood_index]

    traning_fold2 = [
        [h1_source,h1_target],
        [h2_source,h2_target],
        [h3_source,h3_target]
    ]
    # Fold 1, Fold 2
    return ((traning_fold1, testing_fold1), (traning_fold2, testing_fold2))



def main1():
    fold1, fold2 = createData()
    fold1_traning, fold1_testing = fold1[0], fold1[1]
    fold2_traning, fold2_testing = fold2[0], fold2[1]

            
    fedl= FedL("Bursa3_fold1")
    fedl.runWithAlignment(fold1_traning, fold1_testing,epoch = 20, cycle=10, topN=2)
    
    fedl= FedL("Bursa3_fold2")
    fedl.runWithAlignment(fold2_traning, fold2_testing,epoch = 20, cycle=10,topN=2)


def main2():
    fold1, fold2 = createData()
    fold1_traning, fold1_testing = fold1[0], fold1[1]
    fold2_traning, fold2_testing = fold2[0], fold2[1]

            
    #fedl= FedL("Bursa4_fold1")
    #fedl.runWithAlignment(fold1_traning, fold1_testing,epoch = 20, cycle=10, topN=10)
    
    fedl= FedL("Bursa4_fold2")
    fedl.runWithAlignment(fold2_traning, fold2_testing,epoch = 20, cycle=10,topN=10)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    
    #main1()
    main2()
