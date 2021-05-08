# -*- coding: utf-8 -*-
"""
Criado em Wed Mar 31 16:00:00 2021

@author: Jasmine Moreira

1) Preparar dados
2) Criar o modelo (input, output size, forward pass)
3) Criar a função de erro (loss) e o otimizador 
4) Criar o loop de treinamento
   - forward pass: calcular a predição e o erro
   - backward pass: calcular os gradientes
   - update weights: ajuste dos pesos do modelo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time 
import matplotlib.pyplot as plt
import numpy as np

#Preparar dados
v = torch.range(1,10).view(10,1)
l = v**2

#Criar modelo
model = nn.Sequential(nn.Linear(1, 10),
                      nn.ReLU(),
                      nn.Linear(10, 1)
                      )

# Criar função de erro
criterion = nn.L1Loss()

# Otimizador
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Loop treiamento
epochs =20000
for e in range(epochs):  
    output = model(v)
    loss = criterion(output,l)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()   
    
print(f"Training loss: {loss.item()}")
print(model(torch.tensor([5],dtype=torch.float)))

predicted = model(v).detach().numpy()
plt.plot(v.detach(),l.detach(), 'ro')
plt.plot(v.detach(),predicted, 'b')
plt.show()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(r"C:\Users\jasmi\OneDrive\Área de Trabalho\Aprendizado PyTorch\runs")
writer.add_graph(model, v)
