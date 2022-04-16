import os
import shutil
import time
import torch

from constants import B_TRUE, DEVICE
from loaders.features import CausalEmbeddingsDataset
from models.autoencoder import AutoEncoder
from models.causal_autoencoder import CausalAutoEncoder
from models.models import ConvolutionalAE
from utils import compute_total_loss
from os import mkdir

def save_checkpoint(state, is_best, filepath):
    # dirname = os.path.dirname(__file__)
    # filepath = os.path.join(dirname, filepath)

    if not os.path.exists(filepath):
        mkdir(filepath)

    torch.save(state, os.path.join(filepath, 'causal_ae.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'causal_ae.pth.tar'), os.path.join(filepath, 'causal_ae_best.pth.tar'))

def test(model, test_loader):
    loss_fn = compute_total_loss
    test_loss = 0.
    for sample in test_loader:
        with torch.no_grad():
            images, obs_dict, _ = sample
            test_loss += loss_fn(images, obs_dict, model, B_TRUE, use_ground_truth=True)

    test_loss /= len(test_loader.dataset)
    print('\nTest set loss: {:.6f}\n'.format(test_loss))
    return test_loss


def train(model, optimizer, train_loader, epoch, batch_size):
    loss_fn = compute_total_loss
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        images, obs_dict, _ = sample
        # (alpha) acyclicity: 1e9
        # (beta) ground truth loss: 1e2
        # (gamma) image reconstruction: 1e6
        # (rho) faithfulness: ~1
        loss = loss_fn(images, obs_dict, model, B_TRUE, use_ground_truth=True,
            alpha=0.00000001,beta=0.01,gamma=0.00001,rho=1)
        loss.backward(retain_graph=True)
        optimizer.step()
        print('Epoch ' + str(epoch))
        print('Train set, batch ' + str(batch_idx) + ' loss: {:.6f}\n'.format(loss))
    print()


if __name__ == '__main__':
    num_epochs = 50
    batch_size = 16
    learning_rate = 0.001

    dataset = CausalEmbeddingsDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset, test_dataset = train_dataset, test_dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CausalAutoEncoder(
        dataset.schema,
        embedding_dim=8
    ).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters())

    best_loss = 100.
    start_time = time.time()
    for epoch in range(num_epochs):
        train(model, optimizer, train_loader, epoch, batch_size)
        loss = test(model, test_loader)
        is_best = loss > best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()
        }, is_best, filepath='logs')


