# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang
# @File    : train.py
from builtins import print
import ignite.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
class BCELoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()
    def forward(self,frame_prob, tar,frame_level_time=None):
        return nn.functional.binary_cross_entropy(input=frame_prob, target=tar)

# class MSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self,frame_prob, tar):
#         return F.mse_loss(frame_prob, tar)

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,logit,target):
        _,label = torch.max(target,dim=-1)
        label = label.long()
        return nn.CrossEntropyLoss()(logit, label)
class BCELossWithLabelSmoothing(nn.Module):
    """docstring for BCELoss"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
    def forward(self, clip_prob, frame_prob, tar):
        n_classes = clip_prob.shape[-1]
        with torch.no_grad():
            tar = tar * (1 - self.label_smoothing) + (
                1 - tar) * self.label_smoothing / (n_classes - 1)
        return nn.functional.binary_cross_entropy(clip_prob, tar)

class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,w_out,s_out):
        # print('w_out ',w_out.shape)
        # print('s_out ',s_out.shape)
        return F.mse_loss(s_out, w_out, reduction='mean')

class Loss_join(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_join, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.CE = CELoss()

    def update(self, output):
        decision,decision_up, label, embed_label, logit = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        average_loss = self._loss_fn(decision[:,0],label[:,0])
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_join_clr(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_join_clr, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.CLR = SniCoLoss()

    def update(self, output):
        decision,decision_up, frame_level_target, time,embed_label,logit,contrast_pairs = output
        average_loss = self._loss_fn(decision,frame_level_target) + 0.01*self.CLR(contrast_pairs)
        #average_loss = self._loss_fn(decision,frame_level_target)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
    def update(self, output):
        decision,decision_up, clip_level_target = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        average_loss = self._loss_fn(decision[:,0],clip_level_target[:,0])
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_stage1(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_stage1, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.kd_loss = KDLoss()
    def update(self, output):
        w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time, clip_level_target = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        average_loss = self._loss_fn(w_clip_decision[:,0],clip_level_target[:,0])
        dis_loss = self.kd_loss(w_out,s_out)
        loss = average_loss + dis_loss
        # loss  = average_loss
        if len(loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += loss.item() * N
        self._num_examples += N

class Loss_stage1_mask(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_stage1_mask, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.kd_loss = KDLoss()
    def update(self, output):
        w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time, clip_level_target, mask_loss = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        average_loss = self._loss_fn(w_clip_decision[:,0],clip_level_target[:,0])
        dis_loss = self.kd_loss(w_out,s_out)
        loss = average_loss + dis_loss
        # loss  = average_loss
        if len(loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += loss.item() * N
        self._num_examples += N


class Loss_stage1_urban_ad(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_stage1_urban_ad, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.kd_loss = KDLoss()
    def update(self, output):
        w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time, clip_level_target,E_ad,is_source = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        is_source = is_source.squeeze()
        w_clip_decision = w_clip_decision[is_source==1.0] # choose the source domain
        clip_level_target = clip_level_target[is_source==1.0]
        w_out = w_out[is_source==1.0]
        s_out = s_out[is_source==1.0]
        average_loss = self._loss_fn(w_clip_decision[:,0],clip_level_target[:,0])
        dis_loss = self.kd_loss(w_out,s_out)
        loss = average_loss + 5*dis_loss
        # loss  = average_loss
        if len(loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += loss.item() * N
        self._num_examples += N


class Loss_stage2(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_stage2, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.kd_loss = KDLoss()
    def update(self, output):
        w_decision_time, w_clip_decision, w_decision_up, clip_level_target = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        average_loss = self._loss_fn(w_clip_decision[:,0],clip_level_target[:,0])
        # dis_loss = self.kd_loss(w_out,s_out)
        # loss = average_loss + dis_loss
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_stage2_mask(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_stage2_mask, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.kd_loss = KDLoss()
    def update(self, output):
        w_decision_time, w_clip_decision, w_decision_up, clip_level_target, mask_loss = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        average_loss = self._loss_fn(w_clip_decision[:,0],clip_level_target[:,0])
        # dis_loss = self.kd_loss(w_out,s_out)
        # loss = average_loss + dis_loss
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_stage3(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_stage3, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.kd_loss = KDLoss()
    def update(self, output):
        decision_time, decision_up, pseudo_label, clip_level_target = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        mat_clip = clip_level_target[:,0].unsqueeze(1).repeat(1,decision_time.shape[1])
        pseudo_label = pseudo_label*mat_clip
        average_loss = self._loss_fn(decision_time, pseudo_label)
        # dis_loss = self.kd_loss(w_out,s_out)
        # loss = average_loss + dis_loss
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N


class Loss_stage3_mask(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_stage3_mask, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.kd_loss = KDLoss()
    def update(self, output):
        decision_time, decision_up, pseudo_label, clip_level_target, mask_loss = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        mat_clip = clip_level_target[:,0].unsqueeze(1).repeat(1,decision_time.shape[1])
        pseudo_label = pseudo_label*mat_clip
        average_loss = self._loss_fn(decision_time, pseudo_label)
        # dis_loss = self.kd_loss(w_out,s_out)
        # loss = average_loss + dis_loss
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_simnet(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_simnet, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
    def update(self, output):
        sim_res, ans_label  = output
        average_loss = self._loss_fn(sim_res, ans_label)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_s_ad(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_s_ad, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
    def update(self, output):
        decision, decision_up, frame_level_target, time, E, is_source = output
        is_source = is_source.squeeze()
        decision = decision[is_source==1.0]
        frame_level_target = frame_level_target[is_source==1.0]
        average_loss = self._loss_fn(decision,frame_level_target)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_s(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_s, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
    def update(self, output):
        decision, decision_up, frame_level_target, time = output
        # is_source = is_source.squeeze()
        # decision = decision[is_source==1.0]
        # frame_level_target = frame_level_target[is_source==1.0]
        average_loss = self._loss_fn(decision,frame_level_target)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N


class Loss_s_mask(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_s_mask, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
    def update(self, output):
        decision, decision_up, frame_level_target, time, mask_loss = output
        # is_source = is_source.squeeze()
        # decision = decision[is_source==1.0]
        # frame_level_target = frame_level_target[is_source==1.0]
        average_loss = self._loss_fn(decision,frame_level_target)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.25):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)
        return loss

    def consin(self,q,k,neg):
        neg = torch.mean(neg,1)
        loss_w = 0.6*torch.cosine_similarity(q,k,dim=1).mean() - 0.4*torch.cosine_similarity(q,neg,dim=1).mean()
        # print('loss_w ',loss_w)
        return loss_w
    
    def forward(self, contrast_pairs):
        HA_refinement = self.consin(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.consin(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )
        # HA_refinement_consin = self.consin(
        #     torch.mean(contrast_pairs['HA'], 1), 
        #     torch.mean(contrast_pairs['EA'], 1), 
        #     contrast_pairs['EB']
        # )

        # HB_refinement_consin = self.consin(
        #     torch.mean(contrast_pairs['HB'], 1), 
        #     torch.mean(contrast_pairs['EB'], 1), 
        #     contrast_pairs['EA']
        # )
        # loss = HA_refinement + HB_refinement + HA_refinement_consin + HB_refinement_consin
        loss = HA_refinement + HB_refinement
        return loss
        
class SniCoLoss_consin(nn.Module):
    def __init__(self):
        super(SniCoLoss_consin, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.5):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)
        return loss

    def consin(self,q,k,neg):
        neg = torch.mean(neg,1)
        loss_w = 0.6*torch.cosine_similarity(q,k,dim=1).mean() - 0.4*torch.cosine_similarity(q,neg,dim=1).mean()
        # print('loss_w ',loss_w)
        return loss_w
    
    def forward(self, contrast_pairs):
        # HA_refinement = self.NCE(
        #     torch.mean(contrast_pairs['HA'], 1), 
        #     torch.mean(contrast_pairs['EA'], 1), 
        #     contrast_pairs['EB']
        # )

        # HB_refinement = self.NCE(
        #     torch.mean(contrast_pairs['HB'], 1), 
        #     torch.mean(contrast_pairs['EB'], 1), 
        #     contrast_pairs['EA']
        # )
        HA_refinement = self.consin(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.consin(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )
        loss = HA_refinement + HB_refinement
        return loss
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_total = loss_cls + 0.01 * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.5, gamma=2):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*torch.log(predict)).mean() + (-(1-target)*tmp2*torch.log(1-predict)).mean()
        #loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss

class FocalLoss_s(nn.Module):
    def __init__(self,alpha=0.65, gamma=2):
        super(FocalLoss_s,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #self.bce = BCELoss()
    def forward(self, predict, target, frame_level_time=None):
        is_one = (target > 0.5).float()

        # print('is_one ',is_one[0])
        # print('target ',target[0])
        # assert 1==2
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        loss = (-is_one*tmp*torch.log(predict)).mean() + (-(1-is_one)*tmp2*torch.log(1-predict)).mean()
        return loss


class FocalLoss_plus(nn.Module):
    def __init__(self,alpha=0.8, gamma=2,w=0.25):
        super(FocalLoss_plus,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        # print('frame_level_time ',frame_level_time[0:5])
        slack_time = torch.exp(-self.w*frame_level_time)
        # print('slack_time ',slack_time[0:5])
        # assert 1==2
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*slack_time*torch.log(predict)).mean() + (-(1-target)*tmp2*torch.log(1-predict)).mean()
        #loss = (-target*slack_time*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        #loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss

class FocalLoss_plus_total_time(nn.Module):
    def __init__(self,alpha=0.7, gamma=2,w=0.25):
        super(FocalLoss_plus_total_time,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        # print('frame_level_time ',frame_level_time[0:5])
        slack_time = 1.0 + frame_level_time*2.0
        # print('slack_time ',slack_time[:,0])
        # assert 1==2
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        loss = (-target*tmp*slack_time*torch.log(predict)).mean() + (-(1-target)*tmp2*torch.log(1-predict)).mean()
        #loss = (-target*slack_time*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        return loss