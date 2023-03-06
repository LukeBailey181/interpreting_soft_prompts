import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter


class ContinuousPrefix(nn.Module):
    """Continuous prefix embedding that that will be learnt during the training
    process. Once learnt, it is prepended to all inputs in the embedding space.
    """

    def __init__(self, embedding_layer, prefix_len):
        super(ContinuousPrefix, self).__init__()
        init_idx = torch.randint(0, embedding_layer.weight.shape[0], (prefix_len,))

        self.continuous_embeds = nn.ParameterList()
        for i in init_idx:
            continuous_embed = nn.Parameter(torch.zeros(embedding_layer.embedding_dim))
            with torch.no_grad():
                continuous_embed.copy_(embedding_layer.weight[i])
            self.continuous_embeds.append(continuous_embed)

    def freeze_prefix_at_index(self, idx: int):
        self.continuous_embeds[idx].requires_grad = False

    def is_frozen(self, idx: int):
        return not self.continuous_embeds[idx].requires_grad

    def __getitem__(self, idx: int):
        return self.continuous_embeds[idx]

    def __len__(self):
        return len(self.continuous_embeds)


class MLP(nn.Module):
    """Standard MLP implementation"""

    def __init__(
        self,
        c_in,
        c_hidden,
        no_hidden_layers,
        c_out,
    ):
        super(MLP, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_hidden_layers = no_hidden_layers
        self.c_out = c_out

        activation_fns = [nn.Tanh() for _ in range(no_hidden_layers)]
        linears = [nn.Linear(c_hidden, c_hidden) for _ in range(no_hidden_layers)]

        self.mlp = nn.Sequential(
            nn.Linear(c_in, c_hidden),
            *[l for tup in zip(activation_fns, linears) for l in tup],
            nn.Tanh(),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, batch):
        return self.mlp(batch)


class ContinuousPrefixMLP(nn.Module):
    """MLP used to reparametrize prefix. Only used if implementing
    Prefix-Tuning as described in https://arxiv.org/pdf/2101.00190.pdf.
    """

    def __init__(
        self,
        prefix_len,
        hidden_dim,
        embedding_dim,
        no_hidden_layers=0,
    ):
        super(ContinuousPrefixMLP, self).__init__()
        self.prefix_len = prefix_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.no_hidden_layers = no_hidden_layers

        self.continuous_embeds = nn.Parameter(torch.zeros((prefix_len, hidden_dim)))
        with torch.no_grad():
            torch.nn.init.normal_(self.continuous_embeds)

        self._is_frozen = [False for _ in range(prefix_len)]

        self.mlp = MLP(
            hidden_dim,
            embedding_dim,
            no_hidden_layers,
            embedding_dim,
        )

    def freeze_prefix_at_index(self, idx: int):
        self._is_frozen[idx] = False

    def is_frozen(self, idx: int):
        return self._is_frozen[idx]

    def __getitem__(self, idx):
        row = self.continuous_embeds[idx]
        return self.mlp(row)

    def __len__(self):
        return self.prefix_len


class PromptInputEmbedding(nn.Module):
    """Wrapper around continuous prefix allowing use of MLP reparametrization
    or not.
    """

    def __init__(
        self,
        embedding_layer,
        prefix_len,
        parameterize_embeds_with_mlp=False,
        hidden_prefix_dim=None,
        k=1,
    ) -> None:
        super(PromptInputEmbedding, self).__init__()
        self.embedding_layer = embedding_layer
        self.prefix_len = prompt_len
        self.hidden_prefix_dim = hidden_prefix_dim
        self.k = k
        self.iters = 0

        # Initialize continuous embeddings
        if not parameterize_embeds_with_mlp:
            self.continuous_embeds = ContinuousPrefix(
                self.embedding_layer, self.prefix_len
            )
        else:
            assert hidden_prefix_dim is not None
            self.continuous_embeds = ContinuousPrefixMLP(
                self.prefix_len,
                self.hidden_prefix_dim,
                self.embedding_layer.embedding_dim,
            )

        self.prefix = [0] * prefix_len

    def _prepend_prefix(self, input: torch.LongTensor):
        # Discard dummy prefix
        input = input[:, : -self.prefix_len]

        # Tensor-ify frozen prefix indices and reshape them
        prefix_tensor = input.new_tensor(self.prefix, requires_grad=False)
        batch_dim = input.shape[0]
        prefix_tensor = prefix_tensor.unsqueeze(0).expand(
            batch_dim, *prefix_tensor.shape
        )

        return torch.cat([prefix_tensor, input], dim=1)

    def _select_prefix_token(self, options: torch.LongTensor):
        # Replace later if you want more complex behavior for k > 1
        return options[0]

    def freeze_prefix_at_index(self, index: int):
        self.continuous_embeds.freeze_prefix_at_index(index)
        token_idxs = self.deembed(self.continuous_embeds[index], k=self.k)
        self.prefix[index] = self._select_prefix_token(token_idxs)

    def forward(self, input: torch.LongTensor) -> torch.FloatTensor:
        input = self._prepend_prefix(input)
        embeds = self.embedding_layer(input)

        # Fill in continuous prefix embeddings at non-frozen indices
        for i in range(self.prefix_len):
            if not self.continuous_embeds.is_frozen(i):
                ce = self.continuous_embeds[i]
                embeds[:, i] = ce

        return embeds

    def deembed(self, emb, k=1) -> torch.LongTensor:
        # compute L2 distance for one token across all embeddings in one go
        # could deembed the whole prompt at once by unsqueezing another 0th dimension as batch
        dists = torch.cdist(emb.unsqueeze(0), self.embedding_layer.weight)[0]

        _, topk_indices = torch.topk(dists, k, largest=False, dim=0)
        return topk_indices
