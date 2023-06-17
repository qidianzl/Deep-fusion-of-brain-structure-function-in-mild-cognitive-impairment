import torch
from torch.autograd import Variable
from torch.nn import functional as F


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(128, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 2)

    def forward(self, x):
        layer1_out = F.relu(self.linear1(x))
        layer2_out = F.relu(self.linear2(layer1_out))
        out = self.linear3(layer2_out)
        return out, layer1_out, layer2_out

batchsize = 4
lambda1, lambda2 = 0.5, 0.01

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# usually following code is looped over all batches
# but let's just do a dummy batch for brevity

inputs = Variable(torch.rand(batchsize, 128))
targets = Variable(torch.ones(batchsize).long())

optimizer.zero_grad()
outputs, layer1_out, layer2_out = model(inputs)
cross_entropy_loss = F.cross_entropy(outputs, targets)

all_linear1_params = torch.cat([x.view(-1) for x in model.linear1.parameters()])
all_linear2_params = torch.cat([x.view(-1) for x in model.linear2.parameters()])
l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)
l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)

loss = cross_entropy_loss + l1_regularization + l2_regularization
loss.backward()
optimizer.step()