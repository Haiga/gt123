import numpy as np
import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE


def lndcg(y_pred, y_true, ats=None, gain_function=lambda x: x, padding_indicator=PADDED_Y_VALUE):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = 0.  # if idcg == 0 , set ndcg to 0

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def lndcg1(y_pred, y_true, ats=None, gain_function=lambda x: x, padding_indicator=PADDED_Y_VALUE):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = 1.  # if idcg == 0 , set ndcg to 0

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def ndcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = 0.  # if idcg == 0 , set ndcg to 0

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def ndcg1(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = 1.  # if idcg == 0 , set ndcg to 0

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_

def georisk(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = 1.0  # if idcg == 0 , set ndcg to 1.0

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    mat = torch.ones([ndcg_.size()[0], 2]) * 1.0
    mat[:, 0] = ndcg_[:, 0]
    alpha = 3

    numSystems = mat.shape[1]
    numQueries = mat.shape[0]
    Tj = torch.zeros(numQueries)
    Si = torch.zeros(numSystems)
    geoRisk = torch.zeros(numSystems)
    zRisk = torch.zeros(numSystems)
    mSi = torch.zeros(numSystems)

    for i in range(numSystems):
        Si[i] = torch.sum(mat[:, i])
        mSi[i] = torch.mean(mat[:, i])

    for j in range(numQueries):
        Tj[j] = torch.sum(mat[j, :])

    N = torch.sum(Tj)

    for i in range(numSystems):
        tempZRisk = 0
        for j in range(numQueries):
            eij = Si[i] * (Tj[j] / N)
            xij_eij = mat[j, i] - eij
            if eij != 0:
                ziq = xij_eij / torch.sqrt(eij)
            else:
                ziq = 0
            if xij_eij < 0:
                ziq = (1 + alpha) * ziq
            tempZRisk = tempZRisk + ziq
        zRisk[i] = tempZRisk

    c = numQueries

    value = zRisk[0] / c
    m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    ncd = m.cdf(value)
    r = torch.zeros([1, 1])
    r[0][0] = torch.sqrt((Si[0] / c) * ncd)

    return r


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def mrr(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
    """
    Mean Reciprocal Rank at k.

    Compute MRR at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MRR values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))

    ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(len(y_true), len(ats))

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask

    return result
