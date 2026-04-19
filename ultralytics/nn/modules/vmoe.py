import torch
import torch.nn as nn
import torch.distributed as dist

class VMoETopKRouter(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_dim: int,
        noise_std: float = 1.0,
        temperature: float = 1.0,
        deterministic_infer: bool = True,
        importance_loss_weight: float = 1.0,
        load_loss_weight: float = 1.0,
        gshard_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.temperature = temperature
        self.deterministic_infer = deterministic_infer
        self.importance_loss_weight = importance_loss_weight
        self.load_loss_weight = load_loss_weight
        self.gshard_loss_weight = gshard_loss_weight
        self.dense = nn.Linear(in_dim, num_experts, bias=False)

    @staticmethod
    def _cv_squared(x: torch.Tensor) -> torch.Tensor:
        # coefficient of variation squared: (std/mean)^2
        return (x.std(unbiased=False) / (x.mean() + 1e-12)).pow(2)

    def _importance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        # gates: (B,E)
        importance = gates.sum(dim=0)  # (E,)
        return self._cv_squared(importance)

    def _gshard_loss(self, gates: torch.Tensor) -> torch.Tensor:
        E = gates.shape[-1]
        mean_gates = gates.mean(dim=0)  # (E,)
        top1 = torch.zeros_like(gates)
        top1.scatter_(1, gates.argmax(dim=1, keepdim=True), 1.0)
        mean_top1 = top1.mean(dim=0)
        return (mean_top1 * mean_gates).mean() * (E ** 2)

    def _load_loss_top1(self, logits: torch.Tensor, logits_noisy: torch.Tensor, noise_std: float) -> torch.Tensor:
        thr = logits_noisy.max(dim=1).values  # (B,)
        z = (thr[:, None] - logits) / (noise_std + 1e-12)
        normal = torch.distributions.Normal(0.0, 1.0)
        p = 1.0 - normal.cdf(z)             # (B,E)
        p_mean = p.mean(dim=0)              # (E,)
        return self._cv_squared(p_mean)

    def forward(self, feats: torch.Tensor, training: bool):
        """
          expert_id: int(Top-1)
          metrics: dict(auxiliary_loss 等)
        """
        logits = self.dense(feats) / self.temperature   # (B,E)
        gates = torch.softmax(logits, dim=-1)           # (B,E)

        importance = self._importance_loss(gates)

        # 推理默认 deterministic
        deterministic = (not training) and self.deterministic_infer

        if deterministic or self.noise_std == 0.0:
            gshard = self._gshard_loss(gates)
            aux = self.importance_loss_weight * importance + self.gshard_loss_weight * gshard
            # 用 batch 平均 gates 的 argmax 作为整个 batch 的 expert（保持同一patch size）
            expert_id = int(gates.mean(dim=0).argmax().item())
            metrics = {
                "patch_router_importance_loss": importance.detach(),
                "patch_router_gshard_loss": gshard.detach(),
                "patch_router_auxiliary_loss": aux,
            }
            return expert_id, metrics

        # 训练：加噪声路由（对齐 vmoe noisy gating 的 spirit）
        noise_std = (1.0 / self.num_experts) * self.noise_std
        noise = torch.randn_like(logits) * noise_std
        logits_noisy = logits + noise
        gates_noisy = torch.softmax(logits_noisy, dim=-1)

        load = self._load_loss_top1(logits=logits, logits_noisy=logits_noisy, noise_std=noise_std)
        gshard = self._gshard_loss(gates_noisy)
        aux = (
            self.importance_loss_weight * importance
            + self.load_loss_weight * load
            + self.gshard_loss_weight * gshard
        )

        expert_id = int(gates_noisy.mean(dim=0).argmax().item())
        metrics = {
            "patch_router_importance_loss": importance.detach(),
            "patch_router_load_loss": load.detach(),
            "patch_router_gshard_loss": gshard.detach(),
            "patch_router_auxiliary_loss": aux,
        }
        return expert_id, metrics
