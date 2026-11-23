"""A second supervised-contrastive objective over IRRA's hard candidates."""

from typing import Dict, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HardCandidateContrastive(nn.Module):
    """Project and contrast each query's top-ranked gallery candidates.

    Candidate selection uses the base IRRA similarity. During training, known
    same-identity images from the minibatch are appended so each contrastive
    set has positives. At inference, the projected similarity only reranks the
    base top-K; all other gallery scores remain unchanged.
    """

    def __init__(
        self,
        embed_dim: int,
        candidate_topk: int = 8,
        projection_dim: int = 128,
        temperature: float = 0.07,
        score_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if candidate_topk < 1:
            raise ValueError("candidate_topk must be positive")
        if projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if score_weight < 0:
            raise ValueError("score_weight must be non-negative")

        self.candidate_topk = candidate_topk
        self.temperature = temperature
        self.score_weight = score_weight
        self.text_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim),
        )
        nn.init.normal_(self.text_projection[-1].weight, std=0.01)
        nn.init.normal_(self.image_projection[-1].weight, std=0.01)

    @staticmethod
    def _normalize(features: Tensor) -> Tensor:
        return F.normalize(features.float(), dim=-1)

    def _project_text(self, features: Tensor) -> Tensor:
        dtype = self.text_projection[0].weight.dtype
        return self._normalize(self.text_projection(features.to(dtype=dtype)))

    def _project_images(self, features: Tensor) -> Tensor:
        dtype = self.image_projection[0].weight.dtype
        return self._normalize(self.image_projection(features.to(dtype=dtype)))

    @staticmethod
    def _unique_gallery(
        image_features: Tensor, pids: Tensor, image_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if image_ids is None:
            image_ids = torch.arange(image_features.shape[0], device=image_features.device)
        _, inverse = torch.unique(image_ids, sorted=True, return_inverse=True)
        first_indices = []
        for unique_index in range(int(inverse.max().item()) + 1):
            first_indices.append(
                torch.nonzero(inverse == unique_index, as_tuple=False)[0, 0]
            )
        first_indices = torch.stack(first_indices)
        return (
            image_features[first_indices],
            pids[first_indices],
            image_ids[first_indices],
        )

    @staticmethod
    def _supervised_contrastive_loss(
        features: Tensor, labels: Tensor, temperature: float
    ) -> Tensor:
        """SupCon over one query plus its hard image candidates."""
        features = F.normalize(features.float(), dim=-1)
        labels = labels.reshape(-1)
        count = features.shape[0]
        logits = (features @ features.t()) / temperature
        not_self = ~torch.eye(count, dtype=torch.bool, device=features.device)
        logits = logits.masked_fill(~not_self, float("-inf"))
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        positive_mask = labels[:, None].eq(labels[None, :]) & not_self
        logits_for_exp = logits.masked_fill(~not_self, float("-inf"))
        log_prob = logits - torch.logsumexp(logits_for_exp, dim=1, keepdim=True)
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        mean_positive_log_prob = (
            (log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1))
            / positive_count.clamp_min(1)
        )
        return -mean_positive_log_prob[valid].mean()

    def training_losses(
        self,
        text_features: Tensor,
        image_features: Tensor,
        pids: Tensor,
        image_ids: Tensor,
    ) -> Dict[str, Tensor]:
        gallery_features, gallery_pids, gallery_image_ids = self._unique_gallery(
            image_features, pids, image_ids
        )
        base_scores = self._normalize(text_features) @ self._normalize(gallery_features).t()
        text_projected = self._project_text(text_features)
        image_projected = self._project_images(gallery_features)

        losses: List[Tensor] = []
        candidate_counts: List[Tensor] = []
        gallery_count = gallery_features.shape[0]
        seed_count = min(self.candidate_topk, gallery_count)

        for row in range(text_features.shape[0]):
            seeds = base_scores[row].detach().topk(seed_count).indices
            has_positive = (gallery_pids[seeds] == pids[row]).any().item()
            if has_positive:
                candidate_ids = seeds
            else:
                # Add one positive only when none is already in the hard top-K.
                if image_ids is not None:
                    paired_image = torch.nonzero(
                        gallery_image_ids == image_ids[row], as_tuple=False
                    ).flatten()
                else:
                    paired_image = torch.empty(
                        0, dtype=torch.long, device=gallery_pids.device
                    )
                if paired_image.numel() == 0:
                    paired_image = torch.nonzero(
                        gallery_pids == pids[row], as_tuple=False
                    ).flatten()[:1]
                candidate_ids = torch.cat([seeds, paired_image[:1]], dim=0)
            candidate_ids = torch.unique(candidate_ids)

            contrastive_features = torch.cat(
                [text_projected[row : row + 1], image_projected[candidate_ids]], dim=0
            )
            contrastive_labels = torch.cat(
                [pids[row : row + 1], gallery_pids[candidate_ids]], dim=0
            )

            losses.append(
                self._supervised_contrastive_loss(
                    contrastive_features,
                    contrastive_labels,
                    temperature=self.temperature,
                )
            )
            candidate_counts.append(
                torch.tensor(float(candidate_ids.numel()), device=pids.device)
            )

        return {
            "contrastive": torch.stack(losses).mean(),
            "candidate_count": torch.stack(candidate_counts).mean(),
        }

    @torch.no_grad()
    def rerank_scores(
        self,
        text_features: Tensor,
        gallery_features: Tensor,
        base_similarity: Tensor,
    ) -> Tensor:
        final_similarity = base_similarity.clone()
        gallery_count = gallery_features.shape[0]
        if gallery_count == 0:
            return final_similarity

        text_projected = self._project_text(text_features)
        image_projected = self._project_images(gallery_features)
        seed_count = min(self.candidate_topk, gallery_count)

        for row in range(text_features.shape[0]):
            candidate_ids = base_similarity[row].detach().topk(seed_count).indices
            projected_scores = image_projected[candidate_ids] @ text_projected[row]
            final_similarity[row, candidate_ids] = (
                base_similarity[row, candidate_ids]
                + self.score_weight * projected_scores
            )
        return final_similarity
