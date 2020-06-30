import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import GridworldData
from model import VIN


def train():
    for idx, data in enumerate(train_loader):
        X, S1, S2, labels = data
        if X.shape[0] != args.batch_size:
            continue

        X, S1, S2, labels = X.to(device), S1.to(device), S2.to(device), labels.to(device)

        optimizer.zero_grad()
        output = net(X, S1, S2, args.k)

        loss = loss_fn(output, labels)
        loss.backward()

        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()

        if idx % args.log_interval == 0:
            print(f'{idx}/{len(train_loader)}\tTrain Loss: {loss:5f}\tTrain Acc: {correct/X.shape[0]:4f}')


def test():
    correct = 0
    total = 0
    total_loss = 0.0
    net.eval()

    for data in test_loader:
        X, S1, S2, labels = data
        if X.shape[0] != args.batch_size:
            continue

        X, S1, S2, labels = X.to(device), S1.to(device), S2.to(device), labels.to(device)

        output = net(X, S1, S2, args.k)
        total_loss = loss_fn(output, labels)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

        total += X.shape[0]

    test_loss = total_loss / total
    test_acc = correct / total * 100
    print(f'Test Loss: {test_loss:5f}\tTest Acc: {test_acc:4f}')


def main():
    for epoch in range(args.epochs):
        train()

        test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Trainig config
    parser.add_argument('--data_src', type=str,
                        default='./data/gridworld_8x8.npz',
                        help='path to dataset')
    parser.add_argument('--img_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of input image')
    parser.add_argument('--lr', type=float, default=2.e-3,
                        help='choices: [1.e-2, 5.e-3, 2.e-3, 1.e-3]')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs')
    # Model architecture
    parser.add_argument('--k', type=int, default=10,
                        help='number of value iterations')
    parser.add_argument('--ch_in', type=int, default=2,
                        help='number of channels in input layer')
    parser.add_argument('--ch_hidden', type=int, default=150,
                        help='number of channels in hidden layer')
    parser.add_argument('--ch_q', type=int, default=10,
                        help='number of channels in q layer in VI-module')
    # Optional
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_dataset = GridworldData(
        args.data_src, img_size=args.img_size, train=True, transform=None)
    test_dataset = GridworldData(
        args.data_src, img_size=args.img_size, train=False, transform=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    net = VIN(args.ch_in, args.ch_hidden, args.ch_q).to(device)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, eps=1.e-6)
    loss_fn = nn.CrossEntropyLoss()

    main()
