import torch 
import torch.nn as nn 

# 586k parameters
class BasicModel(nn.Module):
    def __init__(self, num_classes = 27):
        super(BasicModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Linear(64*8, 128)       
        self.rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=2, bidirectional=True, dropout= 0.3)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        b, c, h , w = x.size()

        x = x.permute(3, 0, 1, 2)
        x = x.contiguous().view(w, b, c * h)

        x = self.linear(x)
        x, _ =  self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x.log_softmax(2)

if __name__ == '__main__':
    model = BasicModel()
    total = sum(p.numel() for p in model.parameters())
    print(f'Number of Parameters: {total}')