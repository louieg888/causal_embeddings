import torch

def ground_truth_W_loss(W_true, dag_layer):
    """
    1/2n ||W_est - W_true|| ^ 2
    """
    W_est = dag_layer.reconstruct_W(dag_layer.w_est)

    # reconstruct_W
    W_est_abs = torch.abs(W_est)
    W_est_abs_binary = torch.sigmoid(W_est_abs)
    #
    return torch.sum(torch.square(W_true - W_est_abs_binary))


def acyclicity_loss(dag_layer):
    """
    DAGness, aka no acyclicity is allowed
    """

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

            s = torch.sum(W[loc_x - dim_x: loc_x, loc_y - dim_y: loc_y])
            return s

        flattened_index_grid = index_grid.reshape((np.prod(index_grid.size()) // 2, 2))
        f_W_entries = torch.stack([get_sum(entry) for entry in flattened_index_grid])
        f_W = f_W_entries.reshape(index_grid.size()[:2])

        f_W = (1 - torch.eye(f_W.shape[0])) * f_W

        return f_W

    def h(W, schema):
        """Evaluate value and gradient of acyclicity constraint."""
        # fW = f(torch.abs(W))
        fW = f(W, schema=schema)
        fd = fW.shape[0]
        fE = torch.matrix_exp(fW * fW)
        h = torch.trace(fE) - fd
        return h

    W_est = dag_layer.reconstruct_W(dag_layer.w_est)
    return h(f(torch.abs(W_est)), schema=dag_layer)


def local_function_faithfullness_loss(X, dag_layer):
    """
    || X - XW || ^ 2
    """
    X_hat = dag_layer(X)
    diff = X - X_hat
    return 0.5 / X_hat.shape[0] * torch.sum(diff ** 2)

def compute_total_loss(images, obs_dict, conv_ae, W_true, alpha=1, beta=1, gamma=1, use_ground_truth=False):
    X, X_hat, pred_ims = conv_ae(obs_dict, images)
    dag_layer = conv_ae.dag_layer

    if use_ground_truth:
        ground_truth_loss = beta * ground_truth_W_loss(W_true, dag_layer)
    else:
        ground_truth_loss = 0

    return local_function_faithfullness_loss(X, dag_layer) \
           + alpha * acyclicity_loss(dag_layer) \
           + ground_truth_loss \
           + gamma * torch.sum((pred_ims - images) ** 2)
