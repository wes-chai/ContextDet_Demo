import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms

from util import box_ops


class CondNMSPostProcess(nn.Module):
    def __init__(self, num_queries):
        super(CondNMSPostProcess, self).__init__()
        self.num_queries = num_queries

    @torch.no_grad()
    def forward(self, outputs, target_sizes, pred_names, mask_infos):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        bs = len(out_logits)
        results = []

        for b in range(bs):
            b_scores, b_boxes, b_names = [], [], []
            b_start_id, b_end_id = [], []
            name = []
            for name_i in pred_names[b]:
                name.append([name_i] * self.num_queries)
            start_id, end_id = [], []
            for (start, end) in mask_infos[b].keys():
                start_id.append([start] * self.num_queries)
                end_id.append([end] * self.num_queries)
            prob = out_logits[b][0][:, -1:].sigmoid()
            if len(prob) == 0:
                continue
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox[b][0])
            img_h, img_w = target_sizes[b]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
            boxes = boxes * scale_fct[None, :]
            num_patch = len(prob) // self.num_queries
            prob = prob.view(num_patch, self.num_queries, -1)
            boxes = boxes.view(num_patch, self.num_queries, -1)
            for t in range(num_patch):
                ind = prob[t].squeeze(1).topk(100).indices
                prob_prenms = prob[t][ind]
                box_prenms = boxes[t][ind]
                lbl_prenms = torch.zeros_like(prob_prenms)
                nms_ind = batched_nms(box_prenms, prob_prenms[:, 0], lbl_prenms[:, 0], 0.7)[:20]
                b_scores.append(prob_prenms[nms_ind])
                b_boxes.append(box_prenms[nms_ind])

                b_names += [name[t][int(i)] for i in nms_ind]
                b_start_id += [start_id[t][int(i)] for i in nms_ind]
                b_end_id += [end_id[t][int(i)] for i in nms_ind]
            b_scores = torch.cat(b_scores).cpu().squeeze(1)
            b_boxes = torch.cat(b_boxes).cpu()
            out = {'scores': b_scores, 'boxes': b_boxes, 'names': b_names,
                   'start_id': b_start_id, 'end_id': b_end_id}
            results.append(out)
        return results
