import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, model_config):
        super(DQN, self).__init__()
        layers = []
        input_size = model_config["state_size"]
        
        for _, hidden_size in enumerate(model_config["hidden_layers"]):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, model_config["action_size"]))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)