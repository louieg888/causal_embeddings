import math

import torch
import numpy as np

from constants import DEVICE
import spell.metrics as spell_metrics

from loaders.features import CausalEmbeddingsDataset


def f(W, schema):
    dims = torch.tensor(list(schema.values()))
    dims_cumsum = torch.cumsum(dims, dim=0)

    def get_grid(lst):
        grid_xs, grid_ys = torch.meshgrid(lst, lst)
        original_shape = grid_ys.size()
        locations = torch.tensor(
            [(x, y) for x, y in zip((grid_xs.flatten()), grid_ys.flatten())]
        )
        locations = locations.reshape((*original_shape, 2))
        return locations

    locations_grid = get_grid(dims_cumsum)
    dimensions_grid = get_grid(dims)
    index_grid = get_grid(torch.tensor(range(len(dims))))

    def get_sum(index):
        location = locations_grid[tuple(index)]
        dimension = dimensions_grid[tuple(index)]

        loc_x, loc_y = location
        dim_x, dim_y = dimension

        s = torch.sum(W[loc_x - dim_x : loc_x, loc_y - dim_y : loc_y])
        return s

    flattened_index_grid = index_grid.reshape((np.prod(index_grid.size()) // 2, 2))
    f_W_entries = torch.stack([get_sum(entry) for entry in flattened_index_grid])
    f_W = f_W_entries.reshape(index_grid.size()[:2]).to(DEVICE)

    f_W *= (1 - torch.eye(f_W.shape[0])).to(DEVICE)

    return f_W.to(DEVICE)


def ground_truth_W_loss(B_true, dag_layer):
    """
    Penalize relationships that exist that are not supposed to exist.
    encourages sparsity in the weight matrix
    """
    W_est = dag_layer.reconstruct_W(dag_layer.w_est).to(DEVICE)

    # reconstruct_W
    W_est_abs = f(torch.abs(W_est), dag_layer.schema).to(DEVICE)
    W_est_abs *= (1 - B_true).to(DEVICE)
    #
    return torch.sum(W_est_abs)


def acyclicity_loss(dag_layer):
    """
    DAGness, aka no acyclicity is allowed
    """

    def h(W):
        W = W.to(DEVICE)
        """Evaluate value and gradient of acyclicity constraint."""
        # fW = f(torch.abs(W))
        d = W.shape[0]
        # cum_trace = 0
        # for i in range(1, 26):
        #     cum_trace += torch.matrix_power(W, i) / math.factorial(i)
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - d
        # h = cum_trace - d
        return h

    W_est = dag_layer.reconstruct_W(dag_layer.w_est).to(DEVICE)
    return h(f(torch.abs(W_est), dag_layer.schema))


def local_function_faithfullness_loss(X, dag_layer, B_true=None):
    """
    || X - XW || ^ 2
    """
    X = X.to(DEVICE)

    if B_true is not None:
        X_hat = dag_layer(X)
        diff = X - X_hat

        schema_values = list(dag_layer.schema.values())
        schema_values_cumsum = np.cumsum(schema_values)

        for i in range(B_true.shape[0]):
            if B_true.sum(axis=0)[i] == 0:
                end_ind = schema_values_cumsum[i]
                start_ind = end_ind - schema_values[i]
                diff[:, start_ind:end_ind] = 0

        return 0.5 / X_hat.shape[0] * torch.sum(diff**2)

    else:
        X_hat = dag_layer(X)
        diff = X - X_hat
        return 0.5 / X_hat.shape[0] * torch.sum(diff**2)


def compute_total_loss(
    images,
    obs_dict,
    conv_ae,
    B_true=None,
    alpha=1,
    beta=1,
    gamma=1,
    rho=1,
    use_ground_truth=False,
):
    X, X_hat, pred_ims = conv_ae(obs_dict, images)
    
    dag_layer = conv_ae.dag_layer

    if use_ground_truth:
        _ground_truth_loss = beta * ground_truth_W_loss(B_true, dag_layer)
    else:
        _ground_truth_loss = 0

    _local_function_faithfulness_loss = rho * local_function_faithfullness_loss(
        X, dag_layer, B_true=B_true
    )

    _acyclicity_loss = alpha * acyclicity_loss(dag_layer)
    _image_recon_loss = gamma * torch.mean((pred_ims - images) ** 2)

    print(f"local function faithfulness loss: {_local_function_faithfulness_loss}")
    print(f"acyclicity loss: {_acyclicity_loss}")
    print(f"image reconstruction loss: {_image_recon_loss}")
    print(f"ground truth loss: {_ground_truth_loss}")
    print(f(dag_layer.reconstruct_W(dag_layer.w_est), dag_layer.schema))
    print()

    spell_metrics.send_metric(
        "faith_loss",
        local_function_faithfullness_loss(X, dag_layer, B_true=B_true).item(),
    )
    spell_metrics.send_metric(
        "faith_loss_scaled",
        (local_function_faithfullness_loss(X, dag_layer, B_true=B_true) * rho).item(),
    )

    spell_metrics.send_metric("im_loss", torch.mean((pred_ims - images) ** 2).item())
    spell_metrics.send_metric(
        "im_loss_scaled", (torch.mean((pred_ims - images) ** 2) * gamma).item()
    )

    spell_metrics.send_metric("cycle_loss", acyclicity_loss(dag_layer).item())
    spell_metrics.send_metric(
        "cycle_loss_scaled", (acyclicity_loss(dag_layer) * alpha).item()
    )

    spell_metrics.send_metric("GT_loss", ground_truth_W_loss(B_true, dag_layer).item())
    spell_metrics.send_metric(
        "GT_loss_scaled", (ground_truth_W_loss(B_true, dag_layer) * beta).item()
    )

    return (
        _local_function_faithfulness_loss
        + _acyclicity_loss
        + _ground_truth_loss
        + _image_recon_loss
    )

def get_parent_child_relationships(B, schema):
    nodes = list(schema.keys())
    d = B.shape[0]
    relationships = []

    for from_ind in range(d):
        for to_ind in range(d):
            if B[from_ind][to_ind]:
                relationships.append((nodes[from_ind], nodes[to_ind]))

    return relationships

def get_lstsq_loss(args, B, model=None, schema=None):
    from train_causal_autoencoder import get_trained_causal_autoencoder
    """
    Retrieves the least square losses for all of the parent-emb relations.
    """
    if model is None or schema is None:
        schema, model = get_trained_causal_autoencoder(args, B)

    dataset = CausalEmbeddingsDataset(embedding_dimension=args.embedding_dimension)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True,
        generator=torch.Generator().manual_seed(args.random_seed)
    )

    # just one sample because batch size = len(train_dataset)
    images, obs_dict, _ = train_loader[0]

    im_embs = model.autoencoder(images)
    lstsq_losses = {}

    for (parent, child) in get_parent_child_relationships(B, schema):
        lsq_A = obs_dict[parent]
        lsq_B = im_embs

        lstsq_errors = torch.linalg.lstsq(lsq_A, lsq_B).residuals
        lstsq_losses[parent] = 0.5 / len(train_dataset) * torch.norm(lstsq_errors)

    return lstsq_losses

def run_experiment(B_lst, B_true, args):
    og_losses = (B_true, get_lstsq_loss(args, B_true))
    other_losses = [(B, get_lstsq_loss(args, B)) for B in B_lst]

    return og_losses, other_losses