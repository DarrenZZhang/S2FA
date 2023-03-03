import torch
import torch.nn as nn
import numpy as np
from utils.federated_utils import *
from utils.avgmeter import AverageMeter
from utils.mmd import MMD_loss
from sklearn.metrics import roc_auc_score
import numpy as np
from utils.contrastive_utils import contrastive_loss
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def train(train_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
          num_classes, domain_weight, source_domains, sparsity_mmd, confidence_gate_begin,
          confidence_gate_end, communication_rounds, feature_rep, positive_b, negative_b, beta, t_sne):
    task_criterion = nn.CrossEntropyLoss().cuda()
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()

    model_aggregation_frequency = communication_rounds
    # feature_rep_source = [torch.tensor([0]*128, dtype=float).cuda()] * 4
    feature_rep_source = [[],[],[],[]]
    source_num = [0] * 4
    for f in range(model_aggregation_frequency):
        loss_mmd = MMD_loss(kernel_type='linear')
        # Train model locally on source domains
        for index, train_dloader, model, classifier, optimizer, classifier_optimizer, represent_fea in zip(range(source_domain_num),
                                                                                                           train_dloader_list[1:],
                                                                                                           model_list[1:],
                                                                                                           classifier_list[1:],
                                                                                                           optimizer_list[1:],
                                                                                                           classifier_optimizer_list[1:],
                                                                                                           feature_rep):
            total_mmd = 0
            for i, (image_s, label_s) in enumerate(train_dloader):
                source_num[index] = source_num[index] + len(label_s)
                image_s = image_s.cuda()
                label_s = label_s.long().cuda()
                # reset grad
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                # each source domain do optimize
                feature_s = model(image_s)
                # get each sample to obtain the distribution
                feature_s = torch.flatten(feature_s, start_dim=1)
                # extract source feature to show tsne
                if t_sne:
                    for fea in feature_s:
                        feature_rep_source[index].append(fea.cpu().detach().numpy())
                # for fea in feature_s:
                #     tmp = (fea > 0).type(torch.FloatTensor)
                #     feature_rep_source[index] = feature_rep_source[index] + tmp.cuda()
                output_s = classifier(feature_s)
                if sparsity_mmd is True and epoch != 0:
                    mmd_loss = 0
                    task_loss_s = task_criterion(output_s, label_s)
                    feature_s = torch.softmax(feature_s, dim=1)
                    mmd_loss += loss_mmd(feature_s,
                                         torch.unsqueeze(represent_fea.cuda(), dim=0))
                    total_mmd += mmd_loss
                    loss = task_loss_s + mmd_loss

                else:
                    task_loss_s = task_criterion(output_s, label_s)
                    loss = task_loss_s
                loss.backward()
                # task_loss_s.backward()
                optimizer.step()
                classifier_optimizer.step()

            total_mmd = total_mmd/source_num[index]
            # writer.add_scalar(tag="Train/target_domain_{}_mmd".format(source_domains[index]), scalar_value=total_mmd, global_step=epoch + 1)
    # feature_rep_source = [torch.softmax(feature_rep_source[i], dim=0) for i in range(source_domain_num)]
    # feature_rep_source = [feature_rep_source[i] / source_num[i] for i in range(source_domain_num)]
    # print(feature_rep_source)

    # Domain adaptation on target domain
    # confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    confidence_gate = confidence_gate_begin
    # We use I(n_i>=1)/(N_T) to adjust the weight for knowledge distillation domain
    target_weight = [0, 0]
    consensus_focus_dict = {}
    for i in range(1, len(train_dloader_list)):
        consensus_focus_dict[i] = 0
    feature_rep = [torch.tensor([0]*128, dtype=float)] * 4
    feature_rep_target = [[],[],[],[]]
    target_num = 0
    negative_key, positive_key = 0, 0
    negative_value, positive_value = 0, 0
    target_domain_losses = AverageMeter()
    for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
        target_num += len(label_t)
        optimizer_list[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()
        image_t = image_t.cuda()
        # Knowledge Vote
        with torch.no_grad():
            fea = [torch.flatten(model_list[i](image_t), start_dim=1) for i in range(1, source_domain_num + 1)]
            # extract target feature to show tsne
            if t_sne:
                for l in range(source_domain_num):
                    feature_rep_target[l].append(fea[l][0].cpu().detach().numpy())
            for j in range(source_domain_num):
                for k in range(len(label_t)):
                    tmp = (fea[j][k] > 0).type(torch.FloatTensor)
                    feature_rep[j] = feature_rep[j] + tmp
            knowledge_list = [torch.softmax(classifier_list[i](fea[i-1]), dim=1).unsqueeze(1) for
                              i in range(1, source_domain_num + 1)]
            knowledge_list = torch.cat(knowledge_list, 1)
        consensis_predictions, consensus_knowledge, consensus_weight = knowledge_vote(knowledge_list, confidence_gate,
                                                                  num_classes=num_classes)
        # choose the negative and positive keys
        for i in range(len(consensis_predictions)):
            if consensus_knowledge[i][0] == 1:
                if consensis_predictions[i] > negative_value:
                    negative_value = consensis_predictions[i]
                    negative_key = image_t[i].unsqueeze(0)
            else:
                if consensis_predictions[i] > positive_value:
                    positive_value = consensis_predictions[i]
                    positive_key = image_t[i].unsqueeze(0)
        #
        target_weight[0] += torch.sum(consensus_weight).item()
        target_weight[1] += consensus_weight.size(0)
        # Perform data augmentation with mixup
        # lam = np.random.beta(2, 2)
        # batch_size = image_t.size(0)
        # index = torch.randperm(batch_size).cuda()
        # mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]
        # feature_t = model_list[0](mixed_image)
        feature_t = model_list[0](image_t)
        feature_t = torch.flatten(feature_t, start_dim=1)
        # contrastive loss
        cont_loss = 0
        if type(positive_b) is not int and type(negative_b) is not int:
            # if consensus_knowledge[0][1] == 0:
            #     cont_loss = contrastive_loss(feature_t, torch.flatten(model_list[0](negative_b), start_dim=1))
            # else:
            #     cont_loss = contrastive_loss(feature_t, torch.flatten(model_list[0](positive_b), start_dim=1))
            positive = torch.flatten(model_list[0](positive_b), start_dim=1)
            negative = torch.flatten(model_list[0](negative_b), start_dim=1)
            cont_loss = contrastive_loss(feature_t, positive, negative)
            cont_loss = torch.mean(cont_loss)
            # loss.backward()
            # optimizer_list[0].step()
            # classifier_optimizer_list[0].step()
        output_t = classifier_list[0](feature_t)
        # output_t = torch.log_softmax(output_t, dim=1)
        task_loss_t = torch.mean(consensus_weight * (task_criterion(output_t, consensus_knowledge[0][1].unsqueeze(0).long().cuda()) + beta * cont_loss))
        # task_loss_t = cont_loss * 0.1
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        task_loss_t.backward()
        optimizer_list[0].step()
        classifier_optimizer_list[0].step()

        # if sparsity_mmd is True:
        #     mmd_loss = [loss_mmd(feature_t, torch.unsqueeze(feature_rep_source[i].cuda(), dim=0)) for i in range(
        #         source_domain_num)]
        #     task_loss_t = task_loss_t + + 0.1 * 0.25 * (mmd_loss[0] + mmd_loss[1] + mmd_loss[2] + mmd_loss[3])


        # Calculate sparsity
        # feature_t = torch.flatten(feature_t, start_dim=1)
        # for j in range(len(consensus_weight)):
        #     if consensus_weight[j] == 0:
        #         continue
        #     if consensus_knowledge[j][0] == 1:
        #         sparsity_one += feature_t[j]
        #         target_num_one += 1
        #     else:
        #         sparsity_zero += feature_t[j]
        #         target_num_zero += 1
        # Calculate consensus focus
        consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
                                                         source_domain_num, num_classes)
    writer.add_scalar(tag="Train/target_domain_loss", scalar_value=target_domain_losses.avg,
                      global_step=epoch + 1)
    # perform tsne
    if t_sne and (epoch == 0 or epoch >= 150) and (epoch % 10 == 0):
        for i in range(source_domain_num):
            tsne = TSNE(n_components=2)
            emb_test = np.concatenate((feature_rep_source[i], feature_rep_target[i]))
            t = tsne.fit_transform(emb_test)
            t_min, t_max = t.min(0), t.max(0)
            t_norm = (t - t_min) / (t_max - t_min)
            for l in range(len(emb_test)):
                if l < len(feature_rep_source[i]):
                    s1 = plt.scatter(t_norm[l, 0], t_norm[l, 1], color='r')
                else:
                    s2 = plt.scatter(t_norm[l, 0], t_norm[l, 1], color='g')
            plt.legend((s1, s2), ('{}'.format(source_domains[i]), 'target'), loc = 2)
            plt.show()

    # Consensus Focus Re-weighting
    target_parameter_alpha = target_weight[0] / target_weight[1]
    target_weight = round(target_parameter_alpha / (source_domain_num + 1), 4)
    epoch_domain_weight = []
    source_total_weight = 1 - target_weight
    for i in range(1, source_domain_num + 1):
        epoch_domain_weight.append(consensus_focus_dict[i])
    consensus_max = sum(epoch_domain_weight)
    if 0 in epoch_domain_weight:
        for v in range(source_domain_num):
            if epoch_domain_weight[v] == 0:
                epoch_domain_weight[v] = epoch_domain_weight[v] + 1e-3
    epoch_domain_weight = [round(source_total_weight * v / sum(epoch_domain_weight), 4) for v in
                           epoch_domain_weight]
    epoch_domain_weight.insert(0, target_weight)
    # Update domain weight with moving average
    if epoch == 0:
        domain_weight = epoch_domain_weight
    else:
        domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)

    # 源代码，可恢复
    # Model aggregation
    federated_average(model_list, domain_weight)
    # classifier_average
    federated_average(classifier_list, domain_weight)

    # avg 消融实验
    # fedAvg(model_list)
    # fedAvg(classifier_list)

    # average featrue
    # feature_rep_num = [feature_rep[i] / target_num for i in range(source_domain_num)]
    # print(feature_rep_num)
    feature_rep = [torch.softmax(feature_rep[i], dim=0) for i in range(source_domain_num)]
    # target training
    # Recording domain weight in logs
    writer.add_scalar(tag="Train/target_domain_weight", scalar_value=target_weight, global_step=epoch + 1)
    writer.add_scalar(tag="Train/target_domain_max_source", scalar_value=consensus_max, global_step=epoch + 1)
    for i in range(0, len(train_dloader_list) - 1):
        writer.add_scalar(tag="Train/source_domain_{}_weight".format(source_domains[i]),
                          scalar_value=domain_weight[i + 1], global_step=epoch + 1)
    if epoch > 1000:
        print("Source Domains:{}, Domain Weight :{}".format(source_domains, domain_weight[1:]))
    # sys.stdout.flush()
    return domain_weight, feature_rep, positive_key, negative_key


def test(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, num_classes=126,
         top_5_accuracy=True):
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.eval()
    for classifier in classifier_list:
        classifier.eval()
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    output = []
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            output_t = classifier_list[0](torch.flatten(model_list[0](image_t), start_dim=1))
        output.append(torch.softmax(output_t, dim=1))
        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
        task_loss_t = task_criterion(output_t, label_t)
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)
    # writer.add_scalar(tag="Test/target_domain_{}_loss".format(target_domain), scalar_value=target_domain_losses.avg,
    #                   global_step=epoch + 1)
    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    _, y_pred = torch.topk(tmp_score, k=1, dim=1)
    TP, TN, FP, FN = 0, 0, 0, 0
    sen, spe, ppv, npv, auc = 0, 0, 0, 0, 0
    for pred, real in zip(y_pred, y_true):
        if pred == real:
            if pred == 1:
                TP += 1
            else:
                TN += 1
        else:
            if pred == 1:
                FP += 1
            else:
                FN += 1
    output = torch.cat(output, 0).cpu()
    pre = [output[i][1] for i in range(output.shape[0])]
    acc = (TP + TN) / (TP + TN + FP + FN)
    try:
        auc = roc_auc_score(y_true.cpu().numpy(), np.array(pre))
    except:
        ValueError
    try:
        sen, spe, ppv, npv = TP / (TP + FN), TN / (TN + FP), TP / (TP + FP), TN / (
                TN + FN)
    except:
        ZeroDivisionError
    bac = (sen + spe) / 2
    # top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/target_domain_{}_accuracy".format(target_domain).format(target_domain),
                      scalar_value=acc,
                      global_step=epoch + 1)
    if epoch > 1:
        print("Target Domain {} Accuracy {:.4f} Sen {:.4f} Spe {:.4f} Auc:{:.4f} bac {:.4f} ppv {:.4f} npv {:.4f}".format(
                target_domain,
                acc, sen, spe, auc,
                bac, ppv, npv))
    return acc

    # calculate loss, accuracy for source domains
    # for s_i, domain_s in enumerate(source_domains):
    #     tmp_score = []
    #     tmp_label = []
    #     test_dloader_s = test_dloader_list[s_i + 1]
    #     for _, (image_s, label_s) in enumerate(test_dloader_s):
    #         if _ == 0:
    #             continue
    #         image_s = image_s.cuda()
    #         label_s = label_s.long().cuda()
    #         with torch.no_grad():
    #             output_s = classifier_list[s_i + 1](torch.flatten(model_list[s_i + 1](image_s), start_dim=1))
    #         label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
    #         task_loss_s = task_criterion(output_s, label_s)
    #         source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
    #         tmp_score.append(torch.softmax(output_s, dim=1))
    #         # turn label into one-hot code
    #         tmp_label.append(label_onehot_s)
    #     writer.add_scalar(tag="Test/source_domain_{}_loss".format(domain_s), scalar_value=source_domain_losses[s_i].avg,
    #                       global_step=epoch + 1)
    #     tmp_score = torch.cat(tmp_score, dim=0).detach()
    #     tmp_label = torch.cat(tmp_label, dim=0).detach()
    #     _, y_true = torch.topk(tmp_label, k=1, dim=1)
    #     _, y_pred = torch.topk(tmp_score, k=1, dim=1)
    #     top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    #     print("domain_s_{}_acc:{}".format(domain_s, top_1_accuracy_s))
    #     writer.add_scalar(tag="Test/source_domain_{}_accuracy_top1".format(domain_s), scalar_value=top_1_accuracy_s,
    #                       global_step=epoch + 1)