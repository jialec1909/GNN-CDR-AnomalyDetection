import torch
from torch_geometric.data import Data

from pygod.utils.utility import check_parameter


def gen_contextual_outlier_manually(data, n, scale, seed=None):
    """Generating contextual outliers. We randomly select ``n`` nodes as the
    attribute perturbation candidates. For each selected node :math:`i`,
    we randomly pick another ``k`` nodes from the data and select the
    node :math:`j` whose attributes :math:`x_j` deviate the most from
    node :math:`i`'s attribute :math:`x_i` among ``k`` nodes by
    maximizing the Euclidean distance :math:`\| x_i − x_j \|`.
    Afterwards, we then substitute the attributes :math:`x_i` of node
    :math:`i` to :math:`x_j`.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.
    n : int
        Number of nodes converting to outliers.
    scale : float
        Scale based on the original attributes for each outlier node.
    seed : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : torch_geometric.data.Data
        The contextual outlier graph with modified node attributes.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0
        represents normal nodes.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")

    if isinstance(n, int):
        check_parameter(n, low=0, high=data.num_nodes, param_name='n')
    else:
        raise ValueError("n should be int, got %s" % n)

    if isinstance(scale, (int, float)):
        check_parameter(scale, low=0, high=data.num_nodes - n, param_name='scale')
    else:
        raise ValueError("scale should be float, got %s" % scale)

    if seed:
        torch.manual_seed(seed)

    outlier_idx = torch.randperm(data.num_nodes)[:n] # random select index of outliers

    for i, idx in enumerate(outlier_idx):
        data.x[idx] = data.x[idx]*scale

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    return data, y_outlier


def gen_transformer_outlier_manually(data, scale=10, outlier_fraction=0.1):
    """
    Parameters
    ----------
    data : torch.Tensor 
        transformer prediction sequences. The input data. (b, t, f)
    outlier_fraction (float):
        Number of nodes/time points converting to outliers. (fraction)
    scale : float
        Scale based on the original attributes for each outlier node.


    Returns
    -------
    data : torch.Tensor
        tensor sequence with contaminated data.  (b, t, f)    
    label_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0
        represents normal nodes. (b, t, f)
    """

    b, t, f = data.shape
    n_outliers = int(t * f * outlier_fraction)
    

    # label the outlier positions, shape (b, t, f)
    label_outlier = torch.zeros(b, t, f, dtype=torch.long)

    for b_idx in range(b):
        # select randomly n_outliers positions to be outliers out of t*f positions.
        outlier_positions = torch.randperm(t * f)[:n_outliers]
        
        for pos in outlier_positions:
            row = pos // f
            col = pos % f
            data[b_idx, row, col] *= scale
            label_outlier[b_idx, row, col] = 1

    return data, label_outlier

