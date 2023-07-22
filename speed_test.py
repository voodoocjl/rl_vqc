import pennylane as qml
import torch
import torch.nn as nn
import time
from FusionModel import QuantumLayer

rnn = nn.RNN(74, 3)
x = torch.rand([25, 100, 74])
x1 = torch.rand([100, 12])

s = time.time()
y = rnn(x)
e = time.time()
print(e-s)

s = time.time()
y = QuantumLayer(x)
e = time.time()
print(e-s)