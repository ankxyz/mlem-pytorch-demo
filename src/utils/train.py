
from torch import nn


class Network(nn.Module):
    def __init__(self, out_features=2) :
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        #14
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),        #7
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),  #3
            nn.AdaptiveAvgPool2d((1,1))                              #flatten
        )
        
        self.dnnModel = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, out_features)
        )
        
    def forward(self, x) :
        output = self.network(x)
        output = output.squeeze()
        output = self.dnnModel(output)
        return output


def get_loss_fn(device):
    return nn.CrossEntropyLoss().to(device)


def evaluate_model(dataloader, model, loss_fn, device):
    print("Run evaluation")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    preds_true = torch.tensor([]).to(device)
    preds_all = torch.tensor([]).to(device)
    probas = torch.tensor([]).to(device)
    img_paths = []
    
    model.eval()
    mlem_model = MlemModel.from_obj(model)

    for X, y, paths in tqdm(dataloader):
        img_paths.extend(paths)
        X, y = X.to(device), y.to(device)
        # pred = model(X)
        mlem_data = MlemData.from_data(X)
        pred = apply(mlem_model, mlem_data, method='predict')
        
        preds_true = torch.cat((preds_true, y), dim=0)
        preds_all = torch.cat((preds_all, pred), dim=0)
        prob = F.softmax(pred, dim=1)
        top_p, _ = prob.topk(1, dim=1)
        probas = torch.cat((probas, top_p.flatten()), dim=0)

        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    acc = correct / size * 100
    print(f"Test Error: \n Accuracy: {(acc):>0.3f}%, Avg loss: {test_loss:>8f} \n")
    
    return preds_all, preds_true, probas, img_paths