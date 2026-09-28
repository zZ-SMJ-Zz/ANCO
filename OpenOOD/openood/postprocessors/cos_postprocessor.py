import torch
import torch.nn as nn
from typing import Any
from tqdm import tqdm
from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

class CosPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            print('\n Estimating class-wise mean vectors for Cosine Similarity...')
            all_feats = []
            all_labels = []
            
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'], desc='Setup: '):
                    data, labels = batch['data'].to(self.device), batch['label']
                    _, features = net(data, return_feature=True)
                    # 코사인 유사도를 위해 미리 L2 정규화 후 수집
                    all_feats.append(torch.nn.functional.normalize(features, p=2, dim=-1).cpu())
                    all_labels.append(labels)

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            
            # 1. 클래스별 평균 벡터 계산 (Centroid)
            self.class_means = []
            for c in range(self.num_classes):
                mask = all_labels.eq(c)
                if mask.any():
                    # 클래스 c에 해당하는 피처들의 평균을 구한 뒤 다시 정규화
                    c_mean = all_feats[mask].mean(dim=0)
                    self.class_means.append(torch.nn.functional.normalize(c_mean, p=2, dim=0))
                else:
                    self.class_means.append(torch.zeros(all_feats.shape[1]))
            
            # [C, D] 형태의 텐서로 저장
            self.class_means = torch.stack(self.class_means).to(self.device)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        data = data.to(self.device)
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)
        
        # 2. 입력 피처 L2 정규화: ||x|| = 1
        x = torch.nn.functional.normalize(features, p=2, dim=-1) # [B, D]
        
        # 3. 코사인 유사도 계산: (B, D) @ (D, C) = (B, C)
        # 이미 x와 class_means가 정규화되어 있으므로 내적(matmul)이 곧 코사인 유사도입니다.
        cos_sim = torch.matmul(x, self.class_means.t()) # [B, C]
        
        # 가장 높은 유사도를 Confidence로 사용
        conf, _ = torch.max(cos_sim, dim=1)
        
        return pred, conf