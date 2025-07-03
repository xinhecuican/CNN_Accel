# -*- coding: utf-8 -*-
"""

@author: Phuoc Huu
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import time

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1) # [28, 28]
        self.relu1 = nn.ReLU() # [6, 28, 28]
        self.conv2 = nn.Conv2d(6, 16, 3, 1) # [6, 14, 14]
        self.relu2 = nn.ReLU() # [16, 12, 12]
        self.pooling = nn.AvgPool2d(2,2) # [16, 6, 6]
        self.fc1 = nn.Linear(576, 10)
        # self.relu3 = nn.ReLU()
        # self.fc3 = nn.Linear(120, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.relu3(x)
        # x = self.fc3(x)
        x = self.dequant(x)
        output = F.log_softmax(x, dim=1)
        return output

def quantize_and_save_weights(model, test_loader):
    weights_dir = "data/weights_int8/"
    import os
    os.makedirs(weights_dir, exist_ok=True)

    # Chuyển model sang chế độ eval
    model.eval()

    def save_quantized_tensor(tensor, filename):
        # Chuyển về numpy
        tensor_np = tensor.detach().cpu().numpy()
        
        # Tính scale factor để quantize về int8
        max_val = np.abs(tensor_np).max()
        scale = 127.0 / max(max_val, 1e-8)
        
        # Quantize về int8
        tensor_int8 = np.clip(np.round(tensor_np * scale), -128, 127).astype(np.int8)
        
        # Lưu data
        np.savetxt(f"{weights_dir}{filename}", tensor_int8.reshape(-1), fmt='%d')

    # Lấy state dict của model
    state_dict = model.state_dict()
    
    # Lưu weights và biases của từng lớp
    # Conv1
    save_quantized_tensor(state_dict['conv1.weight'], 'w_conv1.txt')
    save_quantized_tensor(state_dict['conv1.bias'], 'b_conv1.txt')
    
    # Conv2
    save_quantized_tensor(state_dict['conv2.weight'], 'w_conv2.txt')
    save_quantized_tensor(state_dict['conv2.bias'], 'b_conv2.txt')
    
    # FC1
    save_quantized_tensor(state_dict['fc1.weight'], 'w_fc1.txt')
    save_quantized_tensor(state_dict['fc1.bias'], 'b_fc1.txt')
    
    # FC2
    # save_quantized_tensor(state_dict['fc2.weight'], 'w_fc2.txt')
    # save_quantized_tensor(state_dict['fc2.bias'], 'b_fc2.txt')
    
    # FC3
    # save_quantized_tensor(state_dict['fc3.weight'], 'w_fc3.txt')
    # save_quantized_tensor(state_dict['fc3.bias'], 'b_fc3.txt')

    print(f"Weights and biases saved to {weights_dir}")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = LeNet5().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    ### Training Phase
    start_time = time.perf_counter()
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    # Code training của bạn ở đây
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Training time: {execution_time:.2f} seconds")
    
    ### Save the model
    if args.save_model:
        torch.save(model.state_dict(), "data/lenet5_model.pt")
        # Quantize và lưu weights
        quantize_and_save_weights(model, test_loader)

        
    
    ### Load the model from lenet5_model.pt file
    print(model)
    model.load_state_dict(torch.load("data/lenet5_model.pt"))
    keys = model.state_dict().keys()
     
    ### Inference Phase
    print('Testing...')
    test(model, device, test_loader)
    
if __name__ == '__main__':
   main()



