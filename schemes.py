import torch
from datasets import MOSIDataLoaders
import FusionModel
import torch.optim as optim
import copy
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def display(metrics):
    print("\nTest mae: {}".format(metrics['mae']))
    # print("Test correlation: {}".format(metrics['corr']))
    # print("Test multi-class accuracy: {}".format(metrics['multi_acc']))
    # print("Test binary accuracy: {}".format(metrics['bi_acc']))
    # print("Test f1 score: {}\n".format(metrics['f1']))


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    for data_a, data_v, data_t, target in data_loader:
        data_a, data_v, data_t = data_a.to(device), data_v.to(
            device), data_t.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data_a, data_v, data_t)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_a, data_v, data_t, target in data_loader:
            data_a = data_a.to(device)
            data_v = data_v.to(device)
            data_t = data_t.to(device)
            target = target.to(device)
            output = model(data_a, data_v, data_t)
            instance_loss = criterion(output, target).item()
            total_loss += instance_loss
    return total_loss / len(data_loader.dataset)


def evaluate(model, data_loader, device):
    model.eval()
    metrics = {}
    with torch.no_grad():
        data_a, data_v, data_t, target = next(iter(data_loader))
        data_a, data_v, data_t = data_a.to(device), data_v.to(
            device), data_t.to(device)
        output = model(data_a, data_v, data_t)
    output = output.cpu().numpy()
    target = target.numpy()
    metrics['mae'] = np.mean(np.absolute(output - target)).item()
    # metrics['corr'] = np.corrcoef(output, target)[0][1].item()
    metrics['multi_acc'] = round(
        sum(np.round(output) == np.round(target)) / float(len(target)),
        5).item()
    true_label = (target >= 0)
    pred_label = (output >= 0)
    metrics['bi_acc'] = accuracy_score(true_label, pred_label).item()
    metrics['f1'] = f1_score(true_label, pred_label, average='weighted').item()
    return metrics


def Scheme(config):
    # random.seed(args.seed)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    print(torch.rand(1))
    args = config.model
    design = config.state
    device = config.device
    if torch.cuda.is_available() and config.device == 'cuda':
        print("using cuda device\n")  
    else:
        print("using cpu device")
    train_loader = config.train_loader
    val_loader = config.val_loader
    test_loader = config.test_loader
    model = FusionModel.QNet(args, design).to(device)
    if config.actual_layer > 0:
        model.load_state_dict(torch.load('results/temp.pth'))
    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = optim.Adam([
        {'params': model.ClassicalLayer_a.parameters()},
        {'params': model.ClassicalLayer_v.parameters()},
        {'params': model.ClassicalLayer_t.parameters()},
        # {'params': model.ProjLayer_a.parameters()},
        # {'params': model.ProjLayer_v.parameters()},
        # {'params': model.ProjLayer_t.parameters()},
        {'params': model.QuantumLayer.parameters(), 'lr': args["qlr"]},
        {'params': model.Regressor.parameters()}
        ], lr=args["clr"])
    val_loss_list = []
    best_val_loss = 10000
    for epoch in range(args["epochs"]):
        train(model, train_loader, optimizer, criterion, device)
        # train_loss = test(model, train_loader, criterion, args)
        # train_loss_list.append(train_loss)
        val_loss = test(model, val_loader, criterion, device)
        val_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(epoch, val_loss, 'saving model')
            best_model = copy.deepcopy(model)
        else:
            print(epoch, val_loss)
    metrics = evaluate(best_model, test_loader, device)
    display(metrics)
    report = {
        'val_loss_list': val_loss_list,
        'best_val_loss': best_val_loss,
        'metrics': metrics
    }
    torch.save(best_model.state_dict(), f'results/temp.pth')
    return best_model, report
