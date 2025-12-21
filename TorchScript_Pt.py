import torch
import torch.nn as nn

# Define and train model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = Net()
scripted_model = torch.jit.script(model)

# Save and load
scripted_model.save("model.pt")
loaded_model = torch.jit.load("model.pt")

print(loaded_model(torch.randn(1, 10)))
