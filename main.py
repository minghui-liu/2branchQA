import numpy
import torch
import argparse
from dataset import QBLinkDataset
from network import TwoBranchNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qanta')
    parser.add_argument('--device', type=str, default='gpu')
    # parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-file", type=str, default='data/qblink_train')
    parser.add_argument("--dev-file", type=str, default='data/qblink_dev')
    parser.add_argument('--test-file', type=str, default = 'data/qblink_test')
    parser.add_argument('--embedding-file', type=str, default='embeddings.npy')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--print-freq', type=int, default=1000)
    # parser.add_argument('--save-model', type=str, default='checkpoints/qanta_delft_glove.pt')
    # parser.add_argument('--load-model', type=str, default='checkpoints/qanta_delft_glove.pt')
    # parser.add_argument('--num-layers', type=int, default=2)
    # parser.add_argument('--input-size', type=int, default=300)
    # parser.add_argument('--hidden-size', type=int, default=300)
    # parser.add_argument('--dropout', type=int, default=0.5)
    # parser.add_argument('--resume', action='store_true', default=False)
    # parser.add_argument('--log-file', type=str, default = '../../experiments/qanta_glove.log')
    # parser.add_argument('--test', action='store_true', default=False)
    # parser.add_argument("--self-attn", action='store_true', default=False)
    return parser.parse_args()

def train(train_loader, model, optimizer, similarity, epoch, device, args):
    model.train()
    for batch_idx, (question, evidence, candidate, label) in enumerate(train_loader):
        question, evidence, candidate, label = question.to(device), evidence.to(device), candidate.to(device), label.to(device)
        # print(f"Batch {batch_idx}", question.shape, evidence.shape, candidate.shape, label.shape)
        # print(question, evidence, candidate, label)
        optimizer.zero_grad()
        question_cond, evidence_cond = model(question, evidence, candidate)
        # print(zqy.shape, zey.shape)
        loss = similarity(question_cond, evidence_cond) * (-label)
        loss = loss.sum()
        # print(loss.shape)
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(question), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        


# def validate(dev_loader, model, similarity, device):
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (question, evidence, candidate, pos) in enumerate(dev_loader):
#             print(question.shape, evidence.shape, candidate.shape, pos.shape)
#             # print(question, evidence, candidate, pos)
#             break


# def test(test_loader, model, similarity, device, args):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for batch_idx, (question, evidence, candidate, pos) in enumerate(test_loader):
#             zqy, zey = model(question, evidence, candidate)
#             test_loss += similarity(zqy, zey) * (-pos)
#             pred = zqy.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(pos.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)


if __name__ == "__main__":

    args = parse_args()
    device = torch.device("cpu")
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda")

    # load data
    train_data = QBLinkDataset(args.train_file, args.embedding_file)
    dev_data = QBLinkDataset(args.dev_file, args.embedding_file)
    # test_data = QBLinkDataset(args.test_file, args.embedding_file)

    # create data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # load network
    embedding_dim = 300
    model = TwoBranchNet(300)
    model.to(device)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # create similarity function
    similarity = torch.nn.CosineSimilarity(dim=1)
    
    for epoch in range(args.epochs):
        # train
        train_loss = train(train_loader, model, optimizer, similarity, epoch, device, args)

        # validate
        # val_loss = validate(dev_loader, model, similarity, device)


    
