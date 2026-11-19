"""Adapt one ProGen block while retaining an immutable reference distribution.

The original embedding, lower blocks, final normalization and output projection
are shared read-only. Only a copy of the final transformer block is trainable.
Sampling uses detached caches; fitting recomputes complete trajectories so the
policy likelihood has a gradient through that copied block.
"""

import copy

import torch

from ..backbone import FrozenProGen


class LastBlockProGen:
    """FrozenProGen-compatible policy with a separately trainable final block.

    ``reference`` and ``model`` refer to the untouched original backbone. Only
    ``last_block`` belongs in the optimizer/checkpoint for this adaptation.
    Evaluation mode disables dropout without disabling parameter gradients.
    This wrapper implements equal-length, unpadded sequences on one device, as
    used by the IDR pilot; it does not implement model parallelism or head masks.
    """

    def __init__(self, reference: FrozenProGen):
        if any(parameter.requires_grad for parameter in reference.model.parameters()):
            raise ValueError("The original ProGen reference must remain frozen")
        if any(module.training for module in reference.model.modules()):
            raise ValueError("The original ProGen reference must be in evaluation mode")
        if reference.model.transformer.model_parallel:
            raise ValueError("Last-block adaptation requires one model device")
        if not reference.model.transformer.h:
            raise ValueError("ProGen needs at least one transformer block")
        self.reference = reference
        self.model = reference.model
        self.device = reference.device
        self.hidden_size = reference.hidden_size
        self.last_block = copy.deepcopy(self.model.transformer.h[-1])
        self.last_block.requires_grad_(True)
        self.last_block.eval()
        self.work = {
            "forward_calls": 0,
            "input_sequences": 0,
            "input_tokens": 0,
            "frozen_lower_block_token_evaluations": 0,
            "adapted_block_token_evaluations": 0,
            "reference_block_token_evaluations": 0,
        }

    def parameters(self):
        """Expose only the copied block, never the frozen reference parameters."""
        return self.last_block.parameters()

    def residue_ids(self, alphabet):
        return self.reference.residue_ids(alphabet)

    def encode(self, sequences):
        return self.reference.encode(sequences)

    @torch.no_grad()
    def _lower_states(self, token_ids, past_key_values=None, use_cache=False):
        """Run shared frozen layers, retaining no training activation graph."""
        if token_ids.ndim != 2 or not all(token_ids.shape):
            raise ValueError("Expected nonempty batch-by-token input IDs")
        transformer = self.model.transformer
        layer_count = len(transformer.h)
        if past_key_values is None:
            past_key_values = (None,) * layer_count
        elif len(past_key_values) != layer_count:
            raise ValueError("The cache must contain one entry per transformer block")
        token_ids = token_ids.to(self.device)
        self.work["forward_calls"] += 1
        self.work["input_sequences"] += token_ids.shape[0]
        self.work["input_tokens"] += token_ids.numel()
        self.work["frozen_lower_block_token_evaluations"] += token_ids.numel() * (
            layer_count - 1
        )
        # ProGen uses rotary positions inside each attention block, including
        # the cached prefix offset. There is no added position embedding here.
        hidden = transformer.drop(transformer.wte(token_ids))
        presents = []
        for index, block in enumerate(transformer.h[:-1]):
            result = block(
                hidden,
                layer_past=past_key_values[index],
                use_cache=use_cache,
                output_attentions=False,
            )
            hidden = result[0]
            if use_cache:
                presents.append(result[1])
        return hidden, past_key_values[-1], presents

    def _adapt(self, lower_states, last_past=None, use_cache=False):
        # The same deterministic policy is used for sampling and likelihoods;
        # eval mode still permits gradients through all copied parameters.
        self.last_block.eval()
        self.work["adapted_block_token_evaluations"] += (
            lower_states.shape[0] * lower_states.shape[1]
        )
        return self.last_block(
            lower_states,
            layer_past=last_past,
            use_cache=use_cache,
            output_attentions=False,
        )

    def _project(self, hidden):
        # Frozen weights still transmit gradients to the adapted hidden states.
        features = self.model.transformer.ln_f(hidden)
        logits = self.model.lm_head(features)
        return features.float(), logits.float()

    def forward(self, sequences):
        """Return differentiable features [B,L+1,D] and logits [B,L+1,V].

        The extra position is the original forward-direction marker. As with
        FrozenProGen, callers align next-residue labels to the preceding state.
        Existing final-feature caches cannot replace this policy computation.
        """
        lower, _, _ = self._lower_states(self.encode(sequences))
        return self._project(self._adapt(lower)[0])

    def forward_with_reference(self, sequences):
        """Return policy features/logits, then detached reference features/logits.

        Both paths share the same frozen lower states. Only the copied final
        block contributes trainable parameters; the original final block and
        projection define the fixed p0 used in the terminal reward. Work counters
        include the additional reference-block token evaluations explicitly.
        """
        lower, _, _ = self._lower_states(self.encode(sequences))
        features, logits = self._project(self._adapt(lower)[0])
        with torch.no_grad():
            self.work["reference_block_token_evaluations"] += (
                lower.shape[0] * lower.shape[1]
            )
            original_hidden = self.model.transformer.h[-1](
                lower, use_cache=False, output_attentions=False
            )[0]
            reference_features, reference_logits = self._project(original_hidden)
        return features, logits, reference_features, reference_logits

    @torch.no_grad()
    def step(self, token_ids, past_key_values=None):
        """Return detached next-token features/logits and per-layer KV caches.

        A fresh generation starts with no cache. Discard caches when copied
        weights change; training always recomputes full trajectories instead.
        """
        lower, last_past, presents = self._lower_states(
            token_ids, past_key_values, use_cache=True
        )
        result = self._adapt(lower, last_past, use_cache=True)
        presents.append(result[1])
        features, logits = self._project(result[0][:, -1])
        return features, logits, tuple(presents)
