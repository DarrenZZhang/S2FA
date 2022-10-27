import torch

def choose_keys(consensis_predictions, consensus_knowledge, negative_value, positive_value):
    change_p, change_n = False, False
    positive_index, negative_index = 0, 0
    for i in range(len(consensis_predictions)):
        if consensus_knowledge[i][0] == 1:
            if consensis_predictions[i] > negative_value:
                negative_value = consensis_predictions[i]
                negative_index = i
                change_n = True
        else:
            if consensis_predictions[i] > positive_value:
                positive_value = consensis_predictions[i]
                positive_index = i
                change_p = True

    return positive_index, negative_index, change_p, change_n

def contrastive_loss(feature_a, positive, negative):
    a = feature_a**2
    sum_a = torch.sum(a, dim=1).unsqueeze(1)
    p = positive**2
    sum_p = torch.sum(p, dim=1).unsqueeze(0)
    pt = positive.t()
    n = negative**2
    sum_n = torch.sum(n, dim=1).unsqueeze(0)
    nt = negative.t()

    dis_p = torch.sqrt(sum_a+sum_p-2*feature_a.mm(pt))
    dis_n = torch.sqrt(sum_a+sum_n-2*feature_a.mm(nt))

    return torch.min(dis_p, dis_n)