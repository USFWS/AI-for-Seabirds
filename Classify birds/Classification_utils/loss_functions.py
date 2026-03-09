
import numpy as np

class_counts = np.bincount(train_dataset.annotations.species_cat)
num_classes = len(class_counts)
total_samples = len(train_dataset.annotations.species_cat)

class_weights = []

for count in class_counts:
    weight = 1 / (count / total_samples)
    class_weights.append(weight)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets]*(1-pt) ** self.gamma* ce_loss).mean()
        return loss

#class_weights

#class_weights = torch.FloatTensor(class_weights)
#class_weights = class_weights.to(device)
