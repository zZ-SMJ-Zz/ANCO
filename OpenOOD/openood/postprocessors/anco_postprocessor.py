import os
from typing import Any
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

class ANCOPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        # 장치를 동적으로 설정 (가능하면 cuda, 아니면 cpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def safe_l2_norm(self, x, eps=1e-12):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + eps)

    def spherical_mean(self, xs, max_iter=100, tol=1e-7):
        # 입력 xs가 이미 해당 장치에 있다고 가정
        mu = xs.mean(dim=0)
        mu = mu / torch.norm(mu)

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
            mu = torch.cos(alpha) * mu + torch.sin(alpha) * (v_mean / alpha)
            mu = mu / torch.norm(mu)
        return mu

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            print('\n Estimating Angular statistics from training set...')
            all_feats = []
            all_labels = []
            
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'], desc='Setup: '):
                    # 데이터를 명시적 장치로 이동
                    data, labels = batch['data'].to(self.device), batch['label']
                    _, features = net(data, return_feature=True)
                    # 메모리 절약을 위해 정규화 후 CPU로 잠시 보냄
                    all_feats.append(self.safe_l2_norm(features).cpu())
                    all_labels.append(labels)

            all_feats = torch.cat(all_feats).to(self.device)
            all_labels = torch.cat(all_labels).to(self.device)
            D = all_feats.shape[1]

            # 1. 클래스별 구면 중심 계산
            self.class_mean = []
            for c in range(self.num_classes):
                fs = all_feats[all_labels.eq(c)]
                if len(fs) > 0:
                    self.class_mean.append(self.spherical_mean(fs))
                else:
                    self.class_mean.append(torch.zeros(D).to(self.device))
            self.class_mean = torch.stack(self.class_mean) # 이미 장치 위에 있음

            # 2. Global Mean & 투영
            global_mean = self.spherical_mean(self.class_mean)
            cos_t = (self.class_mean @ global_mean).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos_t)
            sin_t = torch.sqrt(1.0 - cos_t**2).clamp(min=1e-7)

            u_tangent = (self.class_mean - cos_t.unsqueeze(1) * global_mean.unsqueeze(0)) / sin_t.unsqueeze(1)
            v_centers = theta.unsqueeze(1) * u_tangent

            # 3. 공분산 계산
            centered_v = v_centers - v_centers.mean(dim=0)
            cov = (centered_v.t() @ centered_v) / (v_centers.shape[0] - 1)
            cov += torch.eye(D).to(self.device) * 1e-6
            
            self.precision = torch.inverse(cov)
            self.avg_sigma = torch.sqrt(torch.trace(cov) / D)
            
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # 1. 입력 데이터를 모델 장치로 이동
        data = data.to(self.device)
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)
        
        # 2. 피처 정규화 및 필요한 파라미터 장치 동기화
        x = self.safe_l2_norm(features)
        class_mean = self.class_mean.to(self.device)
        precision = self.precision.to(self.device)
        avg_sigma = self.avg_sigma.to(self.device)
        
        # precision으로부터 공분산 행렬 복원 (Σ = precision^-1)
        cov = torch.inverse(precision)
        
        C = self.num_classes
        all_theta_adj = []

        # 3. 클래스별 루프 연산
        for c in range(C):
            mu_c = class_mean[c]
            
            cos_t = (x @ mu_c).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos_t)
            sin_t = torch.sqrt(1.0 - cos_t.pow(2)).clamp(min=1e-7)
            
            # u 방향 벡터
            u = (x - cos_t.unsqueeze(1) * mu_c.unsqueeze(0)) / sin_t.unsqueeze(1)
            
            # 방향별 분산: var_u = u^T @ Σ @ u
            var_u = torch.sum((u @ cov) * u, dim=1)
            sigma_u = torch.sqrt(var_u.clamp(min=1e-12))
            
            # 각도 조정
            theta_adj = theta * (avg_sigma / sigma_u)
            all_theta_adj.append(-theta_adj)

        all_theta_adj = torch.stack(all_theta_adj, dim=1)
        conf, _ = torch.max(all_theta_adj, dim=1)
        
        return pred, conf