import torch
from torch import nn

class BYOL_Explore(nn.Module):
    def __init__(self, N=512, M=32):
        super().__init__()
        self.encoder = None
        self.closed_loop_rnn_cell = None
        self.open_loop_rnn_cell = None
        self.predictor = None
        self.target_encoder = None
    
        self.encoder = torch.nn.Module({ # TODO: set in_dim, padding and stride values
            'unit1': torch.nn.ModuleList([
                torch.nn.Conv2d(1, 16, 3, 1, 0),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 1, 0),
                torch.nn.GroupNorm(1, 16),
            ]),
            'unit2': torch.nn.ModuleList([
                torch.nn.Conv2d(1, 32, 3, 1, 0),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 1, 0),
                torch.nn.GroupNorm(1, 32),
            ]),
            'unit3': torch.nn.ModuleList([
                torch.nn.Conv2d(1, 32, 3, 1, 0),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 1, 0),
                torch.nn.GroupNorm(1, 32),
            ]),
            'head': torch.nn.ModuleList([
                torch.nn.Flatten() ,
                torch.nn.Linear(in_dim, N)
            ])
        })

        self.closed_loop_rnn_cell = torch.nn.GRU
    

    def encode(self, obs): 
        # Pass through unit 1
        tmp = obs
        x = obs
        for l in self.encoder['unit1']:
            x = l(x)
        x += tmp # residual connection
        tmp = x

        # Pass through unit 2
        for l in self.encoder['unit2']:
            x = l(x)
        x += tmp # residual connection
        tmp = x

        # Pass through unit 3
        for l in self.encoder['unit3']:
            x = l(x)
        x += tpm # residual connection

        # Pass through final head
        for l in self.encoder['head']:
            x = l(x)
        
        return x

