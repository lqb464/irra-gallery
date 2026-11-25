from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        # Keep raw CLIP features for the base row; Gallery Context attends over
        # the complete image set and returns one contextualized feature per image.
        base_gfeats, base_gids = gfeats, gids
        if getattr(model, 'gallery_context_enabled', False):
            with torch.no_grad():
                gfeats = model.encode_gallery(base_gfeats)

        return qfeats, gfeats, qids, gids, base_gfeats, base_gids

    def eval(self, model, i2t_metric=False, log=True, return_metrics=False):
        model = model.eval()
        qfeats, gfeats, qids, gids, base_gfeats, base_gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats.float(), p=2, dim=1) # text features
        gfeats = F.normalize(gfeats.float(), p=2, dim=1) # image features
        base_gfeats = F.normalize(base_gfeats.float(), p=2, dim=1)

        def retrieval_metrics(gallery_features, gallery_ids):
            similarity = qfeats @ gallery_features.t()
            cmc, mean_ap, mean_inp, _ = rank(
                similarity=similarity, q_pids=qids, g_pids=gallery_ids,
                max_rank=10, get_mAP=True)
            return cmc.cpu().numpy(), float(mean_ap.cpu()), float(mean_inp.cpu()), similarity

        base_cmc, base_map, base_minp, base_similarity = retrieval_metrics(
            base_gfeats, base_gids)
        t2i_cmc, t2i_mAP, t2i_mINP, similarity = retrieval_metrics(gfeats, gids)
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i-base', *base_cmc[[0, 4, 9]], base_map, base_minp])
        if getattr(model, 'gallery_context_enabled', False):
            table.add_row(['t2i-hard-contrastive', *t2i_cmc[[0, 4, 9]], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        if log:
            self.logger.info('\n' + str(table))

        metrics = {
            'text_to_image_r1': float(t2i_cmc[0]),
            'text_to_image_r5': float(t2i_cmc[4]),
            'text_to_image_r10': float(t2i_cmc[9]),
            'text_to_image_mAP': float(t2i_mAP),
            'text_to_image_mINP': float(t2i_mINP),
            'base_text_to_image_r1': float(base_cmc[0]),
            'base_text_to_image_r5': float(base_cmc[4]),
            'base_text_to_image_r10': float(base_cmc[9]),
            'base_text_to_image_mAP': base_map,
            'base_text_to_image_mINP': base_minp,
        }
        if return_metrics:
            return metrics
        return metrics['text_to_image_r1']
