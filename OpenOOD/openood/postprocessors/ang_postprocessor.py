import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class ANCOGuidePostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        # config에서 K 읽기 (없으면 기본값 50)
        self.K = getattr(self.config.postprocessor, 'K', 50)

    def safe_l2_norm(self, x, eps=1e-12):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + eps)

    def spherical_mean(self, xs, max_iter=100, tol=1e-7):
        mu = xs.mean(dim=0)
        mu = mu / (torch.norm(mu) + 1e-12)

        for _ in range(max_iter):
            cos_t = (xs @ mu).clamp(-1 + 1e-9, 1 - 1e-9)
            theta = torch.acos(cos_t)
            sin_t = torch.sqrt(1 - cos_t**2).clamp(min=1e-9)

            u = (xs - cos_t.unsqueeze(1) * mu.unsqueeze(0)) / sin_t.unsqueeze(1)
            v = theta.unsqueeze(1) * u
            v_mean = v.mean(dim=0)

            if torch.norm(v_mean) < tol:
                break

            alpha = torch.norm(v_mean)
            mu = torch.cos(alpha) * mu + torch.sin(alpha) * (v_mean / (alpha + 1e-12))
            mu = mu / (torch.norm(mu) + 1e-12)

        return mu

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        print('\nEstimating COCOGuide statistics from training set...')
        all_feats = []
        all_labels = []

        net.eval()
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'], desc='Setup: '):
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)

                _, features = net(data, return_feature=True)
                features = self.safe_l2_norm(features)

                all_feats.append(features.cpu())
                all_labels.append(labels.cpu())

        all_feats = torch.cat(all_feats, dim=0).to(self.device)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)

        D = all_feats.shape[1]

        # --------------------------------------------------
        # 1) class spherical means
        # --------------------------------------------------
        class_means = []
        valid_mask = []

        for c in range(self.num_classes):
            fs = all_feats[all_labels.eq(c)]
            if fs.shape[0] > 0:
                mu_c = self.spherical_mean(fs)
                class_means.append(mu_c)
                valid_mask.append(True)
            else:
                class_means.append(torch.zeros(D, device=self.device))
                valid_mask.append(False)

        self.class_mean = torch.stack(class_means, dim=0)   # [C, D]
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)

        # --------------------------------------------------
        # 2) global covariance on tangent vectors of class means
        # --------------------------------------------------
        valid_class_means = self.class_mean[valid_mask]

        global_mean = self.spherical_mean(valid_class_means)

        cos_t = (valid_class_means @ global_mean).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_t)
        sin_t = torch.sqrt(1.0 - cos_t**2).clamp(min=1e-7)

        u_tangent = (
            valid_class_means - cos_t.unsqueeze(1) * global_mean.unsqueeze(0)
        ) / sin_t.unsqueeze(1)
        v_centers = theta.unsqueeze(1) * u_tangent  # [C_valid, D]

        centered_v = v_centers - v_centers.mean(dim=0, keepdim=True)

        if v_centers.shape[0] > 1:
            cov = (centered_v.t() @ centered_v) / (v_centers.shape[0] - 1)
        else:
            cov = torch.eye(D, device=self.device)

        cov = cov + 1e-6 * torch.eye(D, device=self.device)

        self.cov = cov
        self.avg_sigma = torch.sqrt(torch.trace(cov) / D)
        self.valid_mask = valid_mask

        self.setup_flag = True

    @torch.no_grad()
    def compute_classwise_coco_scores(self, x: torch.Tensor):
        """
        x: [B, D] normalized features
        return: [B, C] classwise COCO scores
        """
        eps = 1e-12
        B = x.shape[0]
        C = self.class_mean.shape[0]

        sims_per_class = []

        for c in range(C):
            mu_c = self.class_mean[c]

            # empty class
            if not self.valid_mask[c] or torch.norm(mu_c) < eps:
                sims_per_class.append(
                    torch.full((B,), -1e6, dtype=x.dtype, device=x.device)
                )
                continue

            cos_t = (x @ mu_c).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos_t)
            sin_t = torch.sqrt(1.0 - cos_t.pow(2)).clamp(min=1e-7)

            u = (x - cos_t.unsqueeze(1) * mu_c.unsqueeze(0)) / sin_t.unsqueeze(1)

            # var_u = u^T Σ u
            var_u = torch.sum((u @ self.cov) * u, dim=1).clamp(min=eps)
            sigma_u = torch.sqrt(var_u)

            theta_adj = theta * (self.avg_sigma / sigma_u)
            theta_adj = torch.clamp(theta_adj, max=torch.pi)

            sim_c = torch.cos(theta_adj)   # [B]
            sims_per_class.append(sim_c)

        return torch.stack(sims_per_class, dim=1)   # [B, C]

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        data = data.to(self.device)
        logits, features = net(data, return_feature=True)

        pred = logits.argmax(dim=1)
        x = self.safe_l2_norm(features)

        # classwise COCO scores: [B, C]
        all_scores = self.compute_classwise_coco_scores(x)

        # base score = max over classes
        base_score, _ = torch.max(all_scores, dim=1)

        # guidance = mean of top-K class-center scores
        k_eff = min(self.K, all_scores.shape[1])
        guide_score = torch.topk(
            all_scores, k=k_eff, dim=1, largest=True, sorted=False
        ).values.mean(dim=1)

        # final COCOGuide score
        conf = base_score * guide_score

        return pred, conf