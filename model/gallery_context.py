"""Trainable gallery-context branch for IRRA.

The module implements a lightweight adaptation of the proposed workflow:
top-K text-to-image candidates -> image-neighbour expansion -> query-conditioned
memory -> confidence-gated graph convolution -> one-sided score adjustment.
"""

from typing import Dict, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GalleryContext(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        candidate_topk: int = 8,
        neighbor_k: int = 4,
        graph_topk: int = 4,
        memory_layers: int = 1,
        score_mode: str = "lower",
        score_adjustment_weight: float = 1.0,
        margin: float = 0.2,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        if score_mode not in {"lower", "raise"}:
            raise ValueError("score_mode must be 'lower' or 'raise'")
        if candidate_topk < 1 or neighbor_k < 0 or graph_topk < 0 or memory_layers < 1:
            raise ValueError(
                "candidate_topk and memory_layers must be positive; neighbour counts cannot be negative"
            )
        if score_adjustment_weight < 0:
            raise ValueError("score_adjustment_weight must be non-negative")
        if embed_dim % 8:
            raise ValueError("embed_dim must be divisible by 8")

        self.candidate_topk = candidate_topk
        self.neighbor_k = neighbor_k
        self.graph_topk = graph_topk
        self.score_mode = score_mode
        self.score_adjustment_weight = score_adjustment_weight
        self.margin = margin
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.memory_encoder = nn.TransformerEncoder(layer, num_layers=memory_layers)
        self.pair_projection = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.match_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.graph_projection = nn.Linear(embed_dim, embed_dim)
        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim * 3 + 1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        nn.init.normal_(self.pair_projection[0].weight, std=0.01)
        nn.init.normal_(self.match_predictor[0].weight, std=0.01)
        nn.init.normal_(self.match_predictor[-1].weight, std=0.01)
        nn.init.normal_(self.graph_projection.weight, std=0.01)
        nn.init.normal_(self.score_predictor[0].weight, std=0.01)
        nn.init.normal_(self.score_predictor[-1].weight, std=0.01)
        nn.init.constant_(self.score_predictor[-1].bias, -4.0)

    @staticmethod
    def _normalize(features: Tensor) -> Tensor:
        return F.normalize(features.float(), dim=-1)

    def _contextual_scores(self, query: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
        """Return same-ID logits and a non-negative score change for one query."""
        dtype = self.pair_projection[0].weight.dtype
        query = query.reshape(1, -1).to(dtype=dtype)
        candidates = candidates.to(dtype=dtype)
        count = candidates.shape[0]

        sequence = torch.cat([query.unsqueeze(1), candidates.unsqueeze(0)], dim=1)
        contextual = self.memory_encoder(sequence).squeeze(0)
        query_context = contextual[0].expand(count, -1)
        candidate_context = contextual[1:]

        query_rows = query.expand(count, -1)
        pair = self.pair_projection(
            torch.cat(
                [
                    query_rows,
                    candidates,
                    torch.abs(query_rows - candidates),
                    query_rows * candidates,
                ],
                dim=-1,
            )
        )
        nodes = pair + candidate_context
        match_logits = self.match_predictor(
            torch.cat([query_context, nodes], dim=-1)
        ).squeeze(-1).float()
        confidence = torch.sigmoid(match_logits)

        visual_similarity = self._normalize(candidates) @ self._normalize(candidates).t()
        if count == 1:
            adjacency = torch.ones_like(visual_similarity)
        else:
            neighbor_count = min(self.graph_topk + 1, count)
            nearest = visual_similarity.detach().topk(neighbor_count, dim=-1).indices
            edge_mask = torch.zeros_like(visual_similarity, dtype=torch.bool)
            edge_mask.scatter_(1, nearest, True)
            edge_mask = edge_mask | edge_mask.t()
            edge_weights = visual_similarity.clamp_min(0.0) * edge_mask.float()
            adjacency = edge_weights + torch.eye(
                count, dtype=edge_weights.dtype, device=edge_weights.device
            )

        node_confidence = 0.1 + 0.9 * confidence
        edge_gate = torch.sqrt(node_confidence[:, None] * node_confidence[None, :])
        adjacency = adjacency * edge_gate
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        graph_context = self.graph_projection(
            adjacency.to(dtype=dtype) @ nodes
        )
        score_input = torch.cat(
            [
                query_context,
                nodes,
                graph_context,
                confidence.to(dtype=dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        delta = F.softplus(self.score_predictor(score_input).squeeze(-1).float())
        return match_logits, delta

    def _expand_training_candidates(
        self,
        query_scores: Tensor,
        gallery_features: Tensor,
        gallery_similarity: Tensor,
    ) -> Tensor:
        gallery_count = gallery_features.shape[0]
        seed_count = min(self.candidate_topk, gallery_count)
        seeds = query_scores.detach().topk(seed_count).indices

        neighbor_count = min(self.neighbor_k + 1, gallery_count)
        neighbors = gallery_similarity[seeds].detach().topk(
            neighbor_count, dim=-1
        ).indices.reshape(-1)
        return torch.unique(torch.cat([seeds, neighbors], dim=0))

    @staticmethod
    def _focal_loss(logits: Tensor, targets: Tensor, gamma: float, alpha: float) -> Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()

    def training_losses(
        self,
        text_features: Tensor,
        image_features: Tensor,
        pids: Tensor,
        image_ids: Tensor,
    ) -> Dict[str, Tensor]:
        """Train on candidates selected without identity labels; use PIDs only for loss targets."""
        if image_ids is None:
            image_ids = torch.arange(image_features.shape[0], device=image_features.device)
        _, inverse = torch.unique(image_ids, sorted=True, return_inverse=True)
        first_indices = []
        for unique_index in range(int(inverse.max().item()) + 1):
            first_indices.append(torch.nonzero(inverse == unique_index, as_tuple=False)[0, 0])
        first_indices_tensor = torch.stack(first_indices)
        gallery_features = image_features[first_indices_tensor]
        gallery_pids = pids[first_indices_tensor]

        text_norm = self._normalize(text_features)
        gallery_norm = self._normalize(gallery_features)
        base_scores = text_norm @ gallery_norm.t()
        gallery_similarity = gallery_norm.detach() @ gallery_norm.detach().t()

        ranking_losses: List[Tensor] = []
        confidence_losses: List[Tensor] = []
        candidate_counts: List[Tensor] = []
        delta_values: List[Tensor] = []
        for row in range(text_features.shape[0]):
            candidate_ids = self._expand_training_candidates(
                base_scores[row],
                gallery_features,
                gallery_similarity,
            )
            candidate_features = gallery_features[candidate_ids]
            candidate_scores = base_scores[row, candidate_ids]
            targets = (gallery_pids[candidate_ids] == pids[row]).float()

            match_logits, delta = self._contextual_scores(
                text_features[row], candidate_features
            )
            # Logging-only statistics; detach so these values never affect gradients.
            delta_values.append(delta.detach().float())
            sign = -1.0 if self.score_mode == "lower" else 1.0
            adjusted_scores = candidate_scores + (
                sign * self.score_adjustment_weight * delta
            )

            positive_scores = adjusted_scores[targets > 0.5]
            negative_scores = adjusted_scores[targets <= 0.5]
            if positive_scores.numel() and negative_scores.numel():
                pairwise = F.softplus(
                    self.margin + negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)
                )
                ranking_losses.append(pairwise.mean())
            else:
                ranking_losses.append(adjusted_scores.sum() * 0.0)

            confidence_losses.append(
                self._focal_loss(
                    match_logits,
                    targets,
                    gamma=self.focal_gamma,
                    alpha=self.focal_alpha,
                )
            )
            candidate_counts.append(torch.tensor(float(candidate_ids.numel()), device=pids.device))

        all_deltas = torch.cat(delta_values)
        return {
            "rank": torch.stack(ranking_losses).mean(),
            "confidence": torch.stack(confidence_losses).mean(),
            "candidate_count": torch.stack(candidate_counts).mean(),
            "delta_mean": all_deltas.mean(),
            "delta_std": all_deltas.std(unbiased=False),
            "delta_min": all_deltas.min(),
            "delta_max": all_deltas.max(),
        }

    @torch.no_grad()
    def rerank_scores(
        self,
        text_features: Tensor,
        gallery_features: Tensor,
        base_similarity: Tensor,
    ) -> Tensor:
        """Adjust the merged top-K plus image-neighbour candidate set per query."""
        final_similarity = base_similarity.clone()
        gallery_count = gallery_features.shape[0]
        if gallery_count == 0:
            return final_similarity
        seed_count = min(self.candidate_topk, gallery_count)
        neighbor_count = min(self.neighbor_k + 1, gallery_count)
        gallery_norm = self._normalize(gallery_features)

        for row in range(text_features.shape[0]):
            seeds = base_similarity[row].detach().topk(seed_count).indices
            neighbor_similarity = gallery_norm[seeds] @ gallery_norm.t()
            neighbors = neighbor_similarity.detach().topk(
                neighbor_count, dim=-1
            ).indices.reshape(-1)
            candidate_ids = torch.unique(torch.cat([seeds, neighbors], dim=0))
            _, delta = self._contextual_scores(
                text_features[row], gallery_features[candidate_ids]
            )
            sign = -1.0 if self.score_mode == "lower" else 1.0
            final_similarity[row, candidate_ids] = (
                base_similarity[row, candidate_ids]
                + sign * self.score_adjustment_weight * delta
            )
        return final_similarity
