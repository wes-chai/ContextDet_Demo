import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import (NestedTensor, inverse_sigmoid,
                       nested_tensor_from_tensor_list)

from .blip2_decoder import BLIP2Decoder
from .deformable_detr.backbone import build_backbone
from .deformable_detr.deformable_detr import DeformableDETR
from .transformer import build_ov_transformer


class ContextDET(DeformableDETR):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, llm_decoder=None):
        super().__init__(backbone, transformer, num_classes, num_queries, num_feature_levels,
                         aux_loss, with_box_refine, two_stage)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm_decoder = llm_decoder
        hidden_dim = transformer.d_model
        out_size = self.llm_decoder.model.opt_proj.out_features
        self.llm_proj = nn.Linear(out_size, hidden_dim, device=self.device)
        self.start_end_proj = nn.Linear(hidden_dim, 2)
        for layer in [self.llm_proj, self.start_end_proj]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        # word_embed_proj_dim = llm_decoder.model.opt_model.config.word_embed_proj_dim
        vocab_size = llm_decoder.model.opt_model.config.vocab_size
        self.fc_logits = nn.Linear(hidden_dim, vocab_size)

    def forward(self, samples, blip2_samples, mask_infos=None, task_button=None, threshold=0.3):
        logits, hidden_states, input_ids, output_text = self.llm_decoder.model.forward(
            blip2_samples, task_button=task_button)
        hidden_states = hidden_states.detach()
        hidden_states = self.llm_proj(hidden_states)

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        out = {}
        start_end_proj = self.start_end_proj(hidden_states)
        out['pred_mlm_logits'] = self.fc_logits(hidden_states)
        out['pred_start'] = start_end_proj[:, :, 0:1]
        out['pred_end'] = start_end_proj[:, :, 1:2]
        out['output_text'] = output_text
        if self.training:
            k = min([len(mask_info) for mask_info in mask_infos])
            k = min(k, 2)
            select_ids = [random.sample(mask_info.keys(), k) for mask_info in mask_infos]
            # select_ids = [random.choices(list(mask_info.keys()), k=4) for mask_info in mask_infos]
            llm_feat = []
            for b in range(len(select_ids)):
                llm_feat_b = []
                hidden_states_b = hidden_states[b, :, :]
                for start, end in select_ids[b]:
                    llm_feat_b.append(hidden_states_b[start: end + 1].mean(dim=0, keepdim=True))
                llm_feat.append(torch.cat(llm_feat_b)[None])
            llm_feat = torch.cat(llm_feat)
            query_embeds = None
            if not self.two_stage:
                query_embeds = self.query_embed.weight
            hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = (
                self.transformer(srcs, masks, pos, query_embeds, llm_feat, k)
            )
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[lvl](hs[lvl])
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)

            out.update({'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                        'init_reference': init_reference})
            out['select_ids'] = select_ids

            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
                for temp in out["aux_outputs"]:
                    temp["select_ids"] = select_ids

            if self.two_stage:
                enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
                out['enc_outputs'] = {
                    'pred_logits': enc_outputs_class,
                    'pred_boxes': enc_outputs_coord,
                    'anchors': anchors,
                }
        else:
            bs = len(samples.tensors)
            mask_infos_pred = [{} for _ in range(bs)]
            llm_feat = []
            tokenizer = self.llm_decoder.model.opt_tokenizer
            if mask_infos is None:
                if task_button == 'Cloze Test':
                    mask_infos = []
                    output_texts = []
                    for b in range(bs):
                        mask_infos_b = {}
                        output_texts_b = []
                        for ind, token in enumerate(input_ids[b]):
                            if token == tokenizer.mask_token_id:
                                mask_infos_b[(ind, ind)] = ''
                                pred_token = out['pred_mlm_logits'][b, ind:ind + 1, :]
                                pred_token = pred_token.argmax(1).item()
                                output_texts_b.append( pred_token )
                                output_texts_b.append( 1437 )
                                input_ids[b, ind: ind + 1] = pred_token
                            else:
                                output_texts_b.append( token.item() )
                        mask_infos.append(mask_infos_b)
                        output_texts.append(tokenizer.decode(output_texts_b[1:]))
                    out['output_text'] = output_texts
                else:
                    mask_infos = []
                    for b in range(bs):
                        starts = (out['pred_start'][b, :, 0].sigmoid() > threshold).nonzero().squeeze(1)
                        ends = (out['pred_end'][b, :, 0].sigmoid() > threshold).nonzero().squeeze(1)
                        if len(starts) == 0:
                            starts = out['pred_start'][b, :].argmax(0)
                        if len(ends) == 0:
                            ends = out['pred_end'][b, :].argmax(0)
                        mask_infos_b = {}
                        for start, end in zip(starts, ends):
                            mask_infos_b[(int(start), int(end))] = ''
                        mask_infos.append(mask_infos_b)
            for b in range(bs):
                llm_feat_b = []
                hidden_states_b = hidden_states[b, :, :]
                for start, end in mask_infos[b].keys():
                    llm_feat_b.append(hidden_states_b[start: end + 1].mean(dim=0, keepdim=True))
                    pred_name = tokenizer.decode(input_ids[b, start: end + 1]).strip()
                    mask_infos_pred[b][(int(start), int(end))] = pred_name
                llm_feat.append(torch.cat(llm_feat_b)[None])
            out['mask_infos_pred'] = mask_infos_pred

            query_embeds = None
            if not self.two_stage:
                query_embeds = self.query_embed.weight

            outputs_classes_list = []
            outputs_coords_list = []
            for b in range(bs):
                srcs_b = [i[b: b + 1] for i in srcs]
                masks_b = [i[b: b + 1] for i in masks]
                pos_b = [i[b: b + 1] for i in pos]
                k = len(mask_infos[b])
                if k == 0:
                    outputs_classes_list.append(torch.zeros(0, 2).to(self.device))
                    outputs_coords_list.append(torch.zeros(0, 4).to(self.device))
                    continue
                num_repeat = math.ceil(k / 4)
                outputs_classes = []
                outputs_coords = []
                for ind in range(num_repeat):
                    llm_feat_b = llm_feat[b][:, ind * 4: (ind + 1) * 4]
                    hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = (
                        self.transformer(srcs_b, masks_b, pos_b, query_embeds, llm_feat_b, llm_feat_b.shape[1])
                    )
                    lvl = hs.shape[0] - 1
                    reference = inter_references[lvl - 1]
                    reference = inverse_sigmoid(reference)
                    outputs_class = self.class_embed[lvl](hs[lvl])
                    tmp = self.bbox_embed[lvl](hs[lvl])
                    if reference.shape[-1] == 4:
                        tmp += reference
                    else:
                        assert reference.shape[-1] == 2
                        tmp[..., :2] += reference
                    outputs_coord = tmp.sigmoid()
                    outputs_classes.append(outputs_class.flatten(0, 1))
                    outputs_coords.append(outputs_coord.flatten(0, 1))
                outputs_classes = torch.cat(outputs_classes)[None]
                outputs_coords = torch.cat(outputs_coords)[None]
                outputs_classes_list.append(outputs_classes)
                outputs_coords_list.append(outputs_coords)

            out.update({'pred_logits': outputs_classes_list,
                        'pred_boxes': outputs_coords_list})
        return out