import os
import shutil
import time
import torch

from loaders.features import CausalEmbeddingsDataset
from models.autoencoder import AutoEncoder
from models.models import ConvolutionalAE
from utils import compute_total_loss
from os import mkdir

def save_checkpoint(state, is_best, filepath):
    #mkdir(filepath)
    torch.save(state, os.path.join(filepath, 'flow_ckpt.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'flow_ckpt.pth.tar'), os.path.join(filepath, 'flow_best.pth.tar'))

def test(model, test_loader):
    loss_fn = torch.nn.MSELoss()
    test_loss = 0.
    for sample in test_loader:
        with torch.no_grad():
            images, labels, _ = sample
            pred_images, image_emb = model(images)
            test_loss += loss_fn(images, pred_images)

    test_loss /= len(test_loader.dataset)
    print('\nTest set loss: {:.6f}\n'.format(test_loss))
    return test_loss


def train(model, optimizer, train_loader, epoch, batch_size):
    loss_fn = torch.nn.MSELoss()
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        images, obs_dict, _ = sample
        pred_images, image_emb = model(images)
        loss = loss_fn(images, pred_images)
        #loss = compute_total_loss(images, obs_dict, model)
        loss.backward()
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AutoEncoder(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128),
        strides=(2, 2, 2),
    )

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


