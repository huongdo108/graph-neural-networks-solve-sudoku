import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import tools
import tests

from model import GNN
from data import Sudoku

# plt.rc('figure', max_open_warning = 0)


import argparse

parser = argparse.ArgumentParser(description="PixelCNN")
parser.add_argument(
    "--n_epochs", default=30, type=int, metavar="N", help="number of epochs to run training loop with default = 11"
)

parser.add_argument(
    "-b", "--batch_size", default=16, type=int, metavar="N", help="batch size training with default = 32"
)

parser.add_argument("--cuda", dest="cuda", action="store_false", help="use cuda")

parser.add_argument("--skip_training", dest="skip_training", action="store_true", help="skip training")

parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, metavar="LR", help="learning rate")


def sudoku_edges():

    flatten_puzzle = np.reshape(np.array(range(81)), (9, 9))
    dic = {}
    for vert_index, outer_value in enumerate(flatten_puzzle):
        if vert_index in (0, 3, 6):
            up = 0
            down = 2
        if vert_index in (1, 4, 7):
            up = -1
            down = 1
        if vert_index in (2, 5, 8):
            up = -2
            down = 0

        for horz_index, inner_value in enumerate(outer_value):
            if horz_index in (0, 3, 6):
                right = 2
                left = 0
            if horz_index in (1, 4, 7):
                right = 1
                left = -1
            if horz_index in (2, 5, 8):
                right = 0
                left = -2

            for i in range(up, down + 1, 1):
                for j in range(left, right + 1, 1):
                    if inner_value not in dic:
                        dic[inner_value] = []
                    if flatten_puzzle[vert_index + i][horz_index + j] != inner_value:
                        dic[inner_value].append(flatten_puzzle[vert_index + i][horz_index + j])

            for i in outer_value:
                if inner_value != i:
                    dic[inner_value].append(i)

            for l in flatten_puzzle:
                if inner_value != l[horz_index]:
                    dic[inner_value].append(l[horz_index])
            dic[inner_value] = set(dic[inner_value])

    src_ids = [[x] * 20 for x in dic.keys()]
    src_ids = torch.Tensor([x for y in src_ids for x in y]).type(torch.LongTensor)
    dst_ids = [y for y in dic.values()]

    dst_ids = torch.Tensor([x for y in dst_ids for x in y]).type(torch.LongTensor)

    return src_ids, dst_ids


def collate(list_of_samples):
    """Merges a list of samples to form a mini-batch.

    Args:
      list_of_samples is a list of tuples (inputs, targets),
          inputs of shape (n_nodes, 9): Inputs to each node in the graph. Inputs are one-hot coded digits
              in the sudoku puzzle. A missing digit is encoded with all zeros. n_nodes=81 for the sudoku graph.
          targets of shape (n_nodes): A LongTensor of targets (correct digits in the sudoku puzzle).

    Returns:
      inputs of shape (batch_size*n_nodes, 9): Inputs to each node in the graph. Inputs are one-hot coded digits
          in the sudoku puzzle. A missing digit is encoded with all zeros. n_nodes=81 for the sudoku graph.
      targets of shape (batch_size*n_nodes): A LongTensor of targets (correct digits in the sudoku puzzle).
      src_ids of shape (batch_size*1620): LongTensor of source node ids for each edge in the large graph.
          The source ids should be between 0 and batch_size * 81.
      dst_ids of shape (batch_size*1620): LongTensor of destination node ids for each edge in the large graph.
          The destination ids should be between 0 and batch_size * 81.
    """
    inputs = [tup[0] for tup in list_of_samples]
    inputs = torch.cat(inputs, 0)
    targets = [tup[1] for tup in list_of_samples]
    targets = torch.cat(targets, 0)
    batch = len(list_of_samples)
    src, dst = sudoku_edges()
    src_ids = torch.cat([src + 81 * i for i in range(batch)], 0)
    dst_ids = torch.cat([dst + 81 * i for i in range(batch)], 0)
    return inputs, targets, src_ids, dst_ids


def fraction_of_solved_puzzles(gnn, testloader, device):
    with torch.no_grad():
        n_test = 0
        n_test_solved = 0
        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            batch_size = inputs.size(0) // 81
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

            outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, 9]
            solution = outputs.view(gnn.n_iters, batch_size, 9, 9, 9)

            final_solution = solution[-1].argmax(dim=3).to(device)
            solved = (final_solution.view(-1, 81) == targets.view(batch_size, 81)).all(dim=1)
            n_test += solved.size(0)
            n_test_solved += solved.sum().item()
            return n_test_solved / n_test


def main():
    """
    
    """
    args = parser.parse_args()
    if args.cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    data_dir = tools.select_data_dir()

    trainset = Sudoku(data_dir, train=True)
    testset = Sudoku(data_dir, train=False)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=collate)
    testloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=collate)

    # Create network
    gnn = GNN(device)
    if not args.skip_training:
        optimizer = torch.optim.Adam(gnn.parameters(), lr=args.learning_rate)
        loss_method = nn.CrossEntropyLoss(reduction="mean")

        for epoch in range(args.n_epochs):
            for i, data in enumerate(trainloader, 0):
                inputs, targets, src_ids, dst_ids = data
                inputs, targets = inputs.to(device), targets.to(device)
                src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
                optimizer.zero_grad()
                gnn.zero_grad()
                output = gnn.forward(inputs, src_ids, dst_ids)
                output = output.to(device)
                output = output.view(-1, output.shape[2])
                targets = targets.repeat(7, 1)
                targets = targets.view(-1)
                loss = loss_method(output, targets)
                loss.backward()
                optimizer.step()

            fraction = fraction_of_solved_puzzles(gnn, testloader, device)

            print("Train Epoch {}: Loss: {:.6f} Fraction: {}".format(epoch + 1, loss.item(), fraction))

        tools.save_model(gnn, "7_gnn.pth")
    else:
        gnn = GNN(device)
        tools.load_model(gnn, "7_gnn.pth", device)

    # Evaluate the trained model
    # Get graph iterations for some test puzzles
    with torch.no_grad():
        inputs, targets, src_ids, dst_ids = iter(testloader).next()
        inputs, targets = inputs.to(device), targets.to(device)
        src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

        batch_size = inputs.size(0) // 81
        outputs = gnn(inputs, src_ids, dst_ids).to(device)  # [n_iters, n_nodes, 9]

        solution = outputs.view(gnn.n_iters, batch_size, 9, 9, 9).to(device)
        final_solution = solution[-1].argmax(dim=3).to(device)
        print("Solved puzzles in the current mini-batch:")
        print((final_solution.view(-1, 81) == targets.view(batch_size, 81)).all(dim=1))

    # Visualize graph iteration for one of the puzzles
    ix = 0
    for i in range(gnn.n_iters):
        tools.draw_sudoku(solution[i, 0], logits=True)

    fraction_solved = fraction_of_solved_puzzles(gnn, testloader,device)
    print(f"Accuracy {fraction_solved}")


if __name__ == "__main__":
    main()
