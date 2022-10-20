import torch 
from torch import nn
from warnings import warn
from torch.utils.data import TensorDataset, DataLoader

def get_filter(name: str, **kwargs):
    if name == "identity":
        return filter_identity
    elif name == "valid":
        if "invalid_value" in kwargs:
            invalid_value = kwargs.pop("invalid_value")
        else:
            invalid_value = -99.
        return lambda thetas, xs, obs: filter_valids(thetas, xs, obs, invalid_value)
    elif "ABC" in name:
        name_list = name.split("_")
        eps = float(name_list[-1])
        return lambda thetas, xs, obs: ABC_filter(thetas,xs,obs, eps=eps)
    else:
        raise NotImplementedError("We only implement the filters identity, best and best_resample")

def ABC_filter(thetas, xs, obs, eps=0.1):
    ABC_mask = torch.sum((xs-obs)**2,-1).sqrt() < eps
    if ABC_mask.sum() == 0:
        ABC_mask[:100] = 1
        warn("No valid simulations present, please increase the simulation budget...")
    return ABC_mask

def filter_valids(thetas, xs, obs, invalid_value):
    valid_mask = (xs!=invalid_value).all(1)
    if valid_mask.sum() == 0:
        valid_mask[0] = 1
        warn("No valid simulations present, please increase the simulation budget...")
    return valid_mask

def filter_identity(thetas, xs, obs):
    """ This just return all samples"""
    return torch.ones(thetas.shape[0]).long()

def build_classifier(input_dim, hidden_dim=50):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim,hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim,hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim,hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim,hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid())

def train_classifier(classifier, data, epochs=10):
    classifier.train()
    loss_fn = nn.BCELoss(reduce=False)
    optim = torch.optim.Adam(classifier.parameters())
    # class weight
    w = data.dataset.tensors[1].sum()/data.dataset.tensors[1].shape[0]
    for i in range(epochs):
        for x,y in data:
            optim.zero_grad()
            y_pred = classifier(x)
            weight = torch.tensor([w,1-w])
            weight_ = weight[y.view(-1).long()].view_as(y)
            loss = torch.mean(loss_fn(y_pred.squeeze(), y)*weight_)
            loss.backward()
            optim.step()
        if (i % int(epochs/5)) == 0:
            print(loss.detach())
    classifier.eval()
    return classifier

def init_classification_data(samples, y, batch_size=1000):
    y = y.float()
    data = TensorDataset(samples, y)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

def append_new_classification_data(dataloader, samples, y, batch_size=1000):
    y = y.float()
    old_samples, old_y = dataloader.dataset.tensors
    samples = torch.vstack([samples, old_samples])
    y = torch.hstack([y, old_y])
    data = TensorDataset(samples, y)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

class LikelihoodWrapper():
    def parameters(self, *args, **kwargs):
        return self.flow.parameters()
    def eval(self, *args, **kwargs):
        self.flow.eval()
        self.classifier.eval()
    def train(self,*args, **kwargs):
        self.flow.train()
        self.classifier.eval()
    def __init__(self, flow, classifier):
        self.flow = flow
        self.classifier = classifier

    def sample(self, *args, **kwargs):
        Y = self.flow.sample(*args, **kwargs)
        return self.transform.inv(Y)

    def log_prob(self, x, theta, *args,**kwargs):
        log_probs = self.flow.log_prob(x,context=theta,*args, **kwargs)
        log_probs_valid = self.classifier(theta).squeeze()
        return log_probs + log_probs_valid.log()
        
def set_surrogate_likelihood(posterior, classifier):
    likelihood = LikelihoodWrapper(posterior.net, classifier)
    posterior.net = likelihood
    posterior._set_up_for_vi(posterior._vi_parameters)