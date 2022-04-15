import torch
from allennlp.common.checks import ConfigurationError
from allennlp.modules.conditional_random_field import ConditionalRandomField


class CRF(ConditionalRandomField):
    def __init__(
        self,
        num_tags: int,
        reduction: str = 'sum',
        **kwargs
    ) -> None:
        super().__init__(num_tags, **kwargs)
        if reduction not in {'sum', 'mean'}:
            raise ConfigurationError(f"Wrong reduction: {reduction}")
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Computes the log likelihood.
        """

        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool, device=inputs.device)
        else:
            # The code below fails in weird ways if this isn't a bool tensor, so we make sure.
            mask = mask.to(torch.bool)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        if self.reduction == 'sum':
            return torch.sum(log_numerator - log_denominator)
        elif self.reduction == 'mean':
            return torch.mean(log_numerator - log_denominator)
        else:
            raise RuntimeError(f"Wrong reduction: {self.reduction}")
