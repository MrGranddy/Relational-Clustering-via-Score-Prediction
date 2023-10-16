import torch


class Regressor(torch.nn.Module):
    
        def __init__(self, num_dim, hidden_dim=128, num_layers=3):
            super().__init__()
    
            self.layers = torch.nn.ModuleList()

            self.layers.append(
                 torch.nn.Sequential(
                        torch.nn.Linear(num_dim, hidden_dim),
                        torch.nn.ReLU(),
                    )
            )

            for _ in range(num_layers - 2):
                self.layers.append(torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                ))

            self.regression_layer = torch.nn.Linear(hidden_dim, 1)

            self.relu = torch.nn.ReLU()

        def forward(self, x):

            last_features = None

            for layer in self.layers:
                x = layer(x)
            
            last_features = x

            x = self.regression_layer(x)

            return x, last_features
        
