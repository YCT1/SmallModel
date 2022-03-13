from multiprocessing import Process
from model import SuperReso
import numpy as np
from sklearn.model_selection import KFold
import torch
class Hosptital():
    def __init__(self, GPUID, id) -> None:
        self.model = SuperReso(GPUID=GPUID)
        self.id = id
        pass

    def train(self,train_source, train_target, epoch):
        self.model.train(train_source, train_target, epoch)
        pass

    def eval(self,test_source, test_target):
        result = self.model.test(test_source, test_target)
        print(f"Hospital ID: {self.id}, Result: {result}")

    def get_parameters(self):
        return self.model.get_parameters()

    def set_parameters(self,parameters):
        self.model.set_parameters(parameters)


class FedL():
    def __init__(self) -> None:
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

    def getWeights(self, hospitalList):
        weights = []
        for hospital in hospitalList:
            weights.append(
                hospital.get_parameters()
            )
        return weights
    
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
            x = Process(target=hospital.eval,args=(test_source,test_target))
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

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    # Create data
    source = np.random.normal(0,0.5, (500,35)).astype(np.float32)
    target = np.random.normal(0,0.5, (500,160)).astype(np.float32)

    source_ood = np.random.normal(10,0.5,(500,35)).astype(np.float32)
    target_ood = np.random.normal(10,0.5,(500,160)).astype(np.float32)

    source_all = np.concatenate((source,source_ood))
    target_all = np.concatenate((target,target_ood))

    # Prepara the data
    data_index = np.arange(0,source.shape[0])

    normal_index = np.random.choice(data_index,400,replace=False)
    h1_index, h2_index = normal_index[0:200],normal_index[200:400]

    h1_source,h1_target = source[h1_index], target[h1_index]
    h2_source,h2_target = source[h2_index], target[h2_index]

    ood_index = np.random.choice(data_index,200,replace=False)
    h3_source, h3_target = source_ood[ood_index], target_ood[ood_index]

    all_data_index = np.arange(0,source_all.shape[0])
    testing_index = np.random.choice(all_data_index,100,replace=False)
    testing_source, testing_target = source_all[testing_index], target_all[testing_index]


    fedl= FedL()

    traning = [
        [h1_source,h1_target],
        [h2_source,h2_target],
        [h3_source,h3_target]
    ]
    fedl.run(traning, (testing_source,testing_target),epoch = 20, cycle = 10)
