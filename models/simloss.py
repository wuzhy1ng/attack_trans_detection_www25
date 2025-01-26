import torch
import torch.nn.functional as F


class SimLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, text_features, graph_features):
        # normalized features
        graph_embeds = graph_features / graph_features.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # compute graph-text loss
        targets = torch.arange(
            graph_embeds.shape[0],
            dtype=torch.long,
            device=self.temperature.device,
        )
        logits = (text_embeds @ graph_embeds.T) / self.temperature.exp()
        loss = F.cross_entropy(logits, targets)
        loss += F.cross_entropy(logits.t(), targets)
        return loss / 2.0
