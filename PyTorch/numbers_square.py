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


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(r"C:\Users\jasmi\OneDrive\Área de Trabalho\Aprendizado PyTorch\runs")

use_cuda = False #torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") 

v = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],dtype=torch.float, requires_grad=True).to(device)
l = torch.tensor([[1],[4],[9],[16],[25],[36],[49],[64],[81],[100]],dtype=torch.float).to(device)

model = nn.Sequential(nn.Linear(1, 10),
                      nn.ReLU(),
                      nn.Linear(10, 1)
                      )

if use_cuda:
    model.cuda()

# Define the loss
criterion = nn.L1Loss()
# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.001)
epochs =10000

start = start = time.time()

for e in range(epochs):  
    output = model(v)
    loss = criterion(output,l)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()   
 
writer.add_graph(model, v)
       
print(f"Training loss: {loss.item()}")
print(model(torch.tensor([5],dtype=torch.float).to(device)).item())
print(f'tempo: {time.time()-start}')