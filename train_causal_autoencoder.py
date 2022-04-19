import argparse
import os
import shutil
import time
from os import mkdir

import torch

from constants import B_TRUE, DEVICE
from loaders.features import CausalEmbeddingsDataset
from models.causal_autoencoder import CausalAutoEncoder
from utils import compute_total_loss
import spell.metrics as spell_metrics

def save_checkpoint(state, is_best, filepath):
    # dirname = os.path.dirname(__file__)
    # filepath = os.path.join(dirname, filepath)

    if not os.path.exists(filepath):
        mkdir(filepath)

    torch.save(
        state, os.path.join(filepath, "causal_ae_" + str(state['epoch']) + ".pth.tar")
    )
    if is_best:
        shutil.copyfile(
            os.path.join(filepath, "causal_ae_" + str(state['epoch']) + ".pth.tar"),
            os.path.join(filepath, "causal_ae_best.pth.tar"),
        )


def test(args, model, test_loader, B_true):
    loss_fn = compute_total_loss
    test_loss = 0.0
    obs_count = 0
    for sample in test_loader:
        with torch.no_grad():
            images, obs_dict, _ = sample
            batch_loss = loss_fn(images, obs_dict, model, B_true, use_ground_truth=True)
            test_loss += batch_loss.detach().item()

            obs_count += images.shape[0]

    test_loss /= obs_count
    print("\nTest set loss: {:.6f}\n".format(test_loss))
    spell_metrics.send_metric("val_loss_avg", test_loss)
    return test_loss


def train(args, model, optimizer, train_loader, epoch, B_true):
    loss_fn = compute_total_loss
    train_avg_loss = 0.0
    obs_count = 0
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        images, obs_dict, _ = sample
        # (alpha) acyclicity: 1e9
        # (beta) ground truth loss: 1e2
        # (gamma) image reconstruction: 1e6
        # (rho) faithfulness: ~1
        loss = loss_fn(
            images,
            obs_dict,
            model,
            B_TRUE,
            use_ground_truth=True,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            rho=args.rho,
        )
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Epoch " + str(epoch))
        print("Train set, batch " + str(batch_idx) + " loss: {:.6f}\n".format(loss))
        train_avg_loss += loss.detach().item()
        obs_count += images.shape[0]
    print()
    train_avg_loss /= obs_count
    print("\nTrain set loss: {:.6f}\n".format(train_avg_loss))
    spell_metrics.send_metric("train_loss_avg", train_avg_loss)

def get_trained_causal_autoencoder(args, B_true):
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    patience = args.patience
    random_seed = args.random_seed
    embedding_dimension = args.embedding_dimension

    dataset = CausalEmbeddingsDataset(embedding_dimension=embedding_dimension)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(random_seed)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        generator = torch.Generator().manual_seed(random_seed)
    )

    torch.manual_seed(random_seed)
    model = CausalAutoEncoder(dataset.schema, embedding_dim=embedding_dimension).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    best_loss = 1000.0
    best_epoch = 0
    curr_patience = patience
    start_time = time.time()
    for epoch in range(num_epochs):
        if curr_patience > 0:
            train(args, model, optimizer, train_loader, epoch, B_true)
            val_loss = test(args, model, test_loader, B_true)
            scheduler.step(val_loss)
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            if is_best:
                curr_patience = patience
                best_epoch = epoch
            else:
                curr_patience -= 1

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                filepath="logs",
            )

    print(f"Training completed, best loss {best_loss} at epoch {best_epoch}")
    return dataset.schema, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train causal AE")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.00000001,
        help="scaling factor on the acyclicity loss (default 0.00000001)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="scaling factor on the ground truth loss (default 0.01)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.00001,
        help="scaling factor on the image reconstruction loss (default 0.00001)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1,
        help="scaling factor on the faithfulness loss (default 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="max number of epochs to train (default 150)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="how many epochs to continue training after no improvement",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default 0.001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="batch size (default 16)"
    )
    parser.add_argument(
        "--embedding-dimension", type=int, default=8, help="size of the image embedding (default 8)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="random seed for weight initialization, train/test split, and shuffling"
    )

    args = parser.parse_args()
    get_trained_causal_autoencoder(args, B_TRUE)