# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def maxmargin_loss(pos, neg, size, margin=0.3, tie=False): #change margin in this function
    loss_zeros = torch.zeros(size, 1).cuda()
    empty_margin = torch.zeros(pos.size()).cuda()


    if tie:
        return torch.sum(loss_zeros)
    else:
        filled_margin = empty_margin.new_full(pos.size(), margin)
        
        loss = torch.max(filled_margin + neg - pos, loss_zeros)

        return torch.mean(loss) #mean for batch
        

def compute_maxmargin_loss(lprobs, target, margin):
  
    target = target.unsqueeze(-1)
 
    sent_pos_lprobs = lprobs.gather(dim=-1, index=target)
    

    sent_pos_lprobs = sent_pos_lprobs.squeeze(-1)

    sent_neg_lprobs = torch.max(lprobs, dim=-1).values
  
    sent_pos_sum_lprobs = torch.sum(sent_pos_lprobs, dim=-1, keepdim=True).cuda()
    sent_neg_sum_lprobs = torch.sum(sent_neg_lprobs, dim=-1, keepdim=True).cuda()

    sent_pos_mean_lprobs = torch.mean(sent_pos_lprobs, dim=-1, keepdim=True).cuda()
    sent_neg_mean_lprobs = torch.mean(sent_neg_lprobs, dim=-1, keepdim=True).cuda()
    

    # choose what to send here
    ploss = maxmargin_loss(sent_pos_mean_lprobs, sent_neg_mean_lprobs, sent_pos_mean_lprobs.size()[0])
    ## ploss = maxmargin_loss(sent_sum_lprobs, target_sum_lprobs, sent_sum_lprobs.size()[0])

    return ploss



def label_smoothed_nll_loss(lprobs, target, epsilon, margin, ignore_index=None, reduce=True): 
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy_max_margin_loss')
class LabelSmoothedCrossEntropyMaxMarginCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, margin): 
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.margin = margin 

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on
        parser.add_argument('--maxmargin_loss_margin', default=0.1, dest='margin', type=float, help="margin for maxmargin finetuning loss")           
       

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True): 
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        
        preserved_lprobs = lprobs
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        preserved_target = target
      
        target = target.view(-1, 1)
        
        margin = 0.3
   
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, margin, ignore_index=self.padding_idx, reduce=reduce,
        )
        maxmargin_loss = compute_maxmargin_loss(preserved_lprobs, preserved_target, margin)
        combined_loss = loss + maxmargin_loss

        return combined_loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
