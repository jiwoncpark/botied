import copy
from typing import Any, Dict, Union
import torch
from torch import nn
import torch_geometric


class RGCN(nn.Module):
    """Relational GCN wrapper from Schlichtkrull, Kipf, et al. 2017
        https://arxiv.org/abs/1703.06103
    """
    def __init__(
            self, params: Dict[str, Any], device: torch.device,
            target_mean: torch.tensor = None, target_std: torch.tensor = None,
            mode: str = None):
        """Initialize model

        Args:
            params: Dict of parameters used to build model architecture
        """
        super().__init__()
        self.params = params  # From config['model']['models'][m]['modules']
        self.d = self.params.get('hidden_dim', 128)

        # GNN modules
        self.gnn = nn.ModuleList()
        for l in range(self.params['n_hidden_layers']):
            in_channels = self.params['n_atom_types'] if l == 0 else (
                self.d)
            self.gnn.append(torch_geometric.nn.RGCNConv(
                in_channels, self.d, num_relations=4))
            self.gnn.append(nn.ReLu())
            self.gnn.append(nn.LayerNorm(normalized_shape=self.d))

        # Pooling module
        self.pool = self.params.get('pool', 'global_mean_pool')

        # Prediction head
        self.dense = nn.ModuleList() if self.params.get(
            'dense') is not None else None
        if self.dense is not None:
            for l, channels in enumerate(self.params['dense']):
                in_features = self.d if l == 0 else self.params['dense'][l-1]
                self.dense.append(
                    nn.Linear(in_features=in_features, out_features=channels))
                if l+1 != len(channels):
                    self.dense.append(nn.ReLU())
                    self.dense.append(nn.LayerNorm(normalized_shape=channels))
                    self.dense.append(nn.Dropout(0.2))

        # Posterior modules (prediction and denoising)
        self.n_posterior_samples = self.params.get('n_posterior_samples', None)
        self.logit_sigma = copy.deepcopy(
            self.dense) if self.n_posterior_samples is not None and (
                self.dense is not None) else None
        self.logit_sigma.append(nn.Softplus())

        # Logit activation modules (final prediction layers)
        self.act = nn.Sequential(nn.ReLu()) if self.dense is not None else None

        # Misc params
        self.device = device
        self.target_mean = target_mean
        self.target_std = target_std
        self.mode = mode

    def embed(self, data: Union[
        torch_geometric.data.Data,
        torch_geometric.data.batch.Batch]) -> torch.Tensor:
        """Constructs learned embedding vector
        Args:
            data: Graph or batch object with attributes for modeling
        Returns:
            x: Convolved nodes
        """
        return self.gnn(data.x, data.edge_index, edge_type=data.edge_type)

    def pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pools node representations to construct molecule embedding

        Args:
            x: Node representations
            data: Batch index tensor

        Returns:
            x: Pooled molecule embedding

        """
        # Pools representations
        if self.pool == 'global_mean_pool':
            x = torch_geometric.nn.global_mean_pool(x, batch=batch)
        elif self.pool == 'global_add_pool':
            x = torch_geometric.nn.global_add_pool(x, batch=batch)

        return x

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits from learned embeddings

        Args:
            x: Learned embedding vector

        Returns:
            x: Logits
        """
        for layer in self.dense:
            x = layer(x)
        return x

    def compute_logit_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """Computes parametric logit std from penultimate layer

        Args:
            x: Penultimate representation

        Returns:
            s: Logit sigma
        """
        for layer in self.logit_sigma:
            x = layer(x)
        # Add 1e-5 for numerical stability in build_posterior
        return x + 1e-5

    def build_posterior(
        self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Draws samples from parameterized logits

        Args:
            x: Predicted mean logit
            s: Predicted logit std

        Returns:
            x: Sampled logits
        """
        return torch.distributions.normal.Normal(x, s)

    def activate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Computes predictions from predicted logits

        Args:
            x: Logits

        Returns:
            x: Predictions
        """
        if self.act is not None:
            return self.act(x)
        return x

    def forward(self, data: Union[
        torch_geometric.data.Data,
        torch_geometric.data.batch.Batch], return_embeddings: bool = False,
        return_logits: bool = False) -> torch.Tensor:
        """Constructs learned embedding vector
        Args:
            data: Graph or batch object with attributes for modeling
            return_embeddings: Whether to return the embeddings
            return_logits: Whether to return the logits

        Returns:
            prediction data
        """
        embeddings = self.embed(data.to(self.device))
        parameterize = True if self.logit_sigma is not None else False
        logits = self.compute_logits(embeddings)
        posterior = None
        if parameterize:
            mu = logits
            sigma = self.compute_logit_sigma(embeddings)
            # Added sample dimension [n_samples, batch, targets]
            posterior = self.build_posterior(mu, sigma)
            logits = posterior.rsample(
                sample_size=torch.Size([self.n_posterior_samples]))
            preds = self.activate_logits(
                logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            preds = self.activate_logits(logits)
        if self.target_mean is not None and self.target_std is not None:
            preds = (preds*self.target_std.to(self.device)) + (
                self.target_mean.to(self.device))

        return {
            'preds': preds, 'embeddings': embeddings, 'logits': logits,
            'posterior': posterior}
