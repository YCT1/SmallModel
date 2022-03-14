from tqdm import tqdm
import torch
from torch.nn import Linear,Sigmoid, L1Loss
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import OrderedDict

class NNSR(torch.nn.Module):
    def __init__(self, source_Dimention, target_Dimention) -> None:
        super(NNSR,self).__init__()
        difference  = target_Dimention-source_Dimention


        self.layer1 = Linear(source_Dimention, source_Dimention + 1*difference//5)
        self.layer2 = Linear(source_Dimention + 1*difference//5, source_Dimention + 2*difference//5)
        self.layer3 = Linear(source_Dimention + 2*difference//5, source_Dimention + 3*difference//5)
        self.layer4 = Linear(source_Dimention + 3*difference//5, source_Dimention + 4*difference//5)
        self.layer5 = Linear(source_Dimention + 4*difference//5, target_Dimention)
        self.activition = Sigmoid()
    
    def forward(self, data):
        x = self.layer1(data)
        x = self.activition(x)

        x = self.layer2(x)
        x = self.activition(x)

        x = self.layer3(x)
        x = self.activition(x)

        x = self.layer4(x)
        x = self.activition(x)

        x = self.layer5(x)
        
        return x
        

        
class MyDataset(Dataset):
    def __init__(self, source_data, target_data) -> None:
        super(MyDataset,self).__init__()
        assert source_data.shape[0] == target_data.shape[0]

        self.x = source_data
        self.y = target_data
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


class SuperReso():
    def __init__(self, GPUID) -> None:
        if torch.cuda.is_available():
            self.device = f"cuda:{GPUID}"
            print(f"Running on cuda:{GPUID}")
        else:
            self.device = "cpu"

        self.model = NNSR(35,160).to(self.device)
        self.L1loss = L1Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        pass

    def train(self, source_data, target_data, N_epoch=100):


        trainData = MyDataset(source_data,target_data)
        self.model.train()
        train_dataloader = DataLoader(trainData, batch_size=50)
        
        size = len(train_dataloader.dataset)

        for epoch in range(N_epoch):
            if epoch % 20 == 1:
                print(f"[Epoch: {epoch}] [loss: {loss:>7f}] ")

            with torch.autograd.set_detect_anomaly(True):
                for batch, (source, target) in enumerate(train_dataloader):
                    
                    source, target = source.to(self.device), target.to(self.device)

                    pred = self.model(source)
                    loss = self.L1loss(pred,target)

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    ###


                    if batch == 9 :
                        loss, current = loss.item(), (batch+1) * len(source)
                        #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    pass
            pass
        pass

        print(f"loss: {loss:>7f}")
    def test(self, test_source, target_source):
        testData = MyDataset(test_source,target_source)
        test_dataloader = DataLoader(testData, batch_size=50)
        self.model.eval()

        test_loss, correct = 0,0
        test_results = []
        with torch.no_grad():
            for X,y in test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                result = self.L1loss(pred, y).item()
                test_loss += result

                for X_item, Y_item in zip(pred,y):

                    test_results.append(self.L1loss(X_item, Y_item).item())

                pass

        test_loss /= len(test_dataloader)
        return test_loss, test_results

    def get_parameters(self):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

# np.random.seed(1000)

# source_data, target_data = torch.from_numpy(np.random.normal(0,0.5, (250,35)).astype(np.float32)), torch.from_numpy(np.random.normal(0,0.5, (250,160)).astype(np.float32))
# training_source_data,training_target_data = source_data[0:200], target_data[0:200]

# testing_source_data, testing_target_data  = source_data[200:250], target_data[200:250]

# sp = SuperReso()
# sp.train(training_source_data,training_target_data, N_epoch=1000)
# sp.test(testing_source_data,testing_target_data)


