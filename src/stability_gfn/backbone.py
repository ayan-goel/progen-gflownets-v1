"""Frozen ProGen2 features, teacher-forced logits and incremental decoding."""

from pathlib import Path

import torch
from tokenizers import Tokenizer

from .environment import PrefixTree
from .vendor.progen.configuration_progen import ProGenConfig
from .vendor.progen.modeling_progen import ProGenForCausalLM

TOKENIZER_PATH = Path(__file__).parent / "vendor/progen/tokenizer.json"


class FrozenProGen:
    """Keep the original backbone and output projection frozen in evaluation mode."""

    def __init__(self, model, device="cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=dtype).eval()
        self.model.requires_grad_(False)
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        # The literal direction marker '1' is not the tokenizer's special BOS ID.
        self.boundary_id = self.tokenizer.encode("1").ids
        if len(self.boundary_id) != 1:
            raise ValueError("Expected a single forward-direction boundary token")
        self.hidden_size = model.config.n_embd
        self.work = {"forward_calls": 0, "input_sequences": 0, "input_tokens": 0}
        self.residue_ids("ACDEFGHIKLMNPQRSTVWY")

    @classmethod
    def from_checkpoint(cls, path, device, precision="float32"):
        """Load only local files; GPU execution is explicitly requested by the caller."""
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if precision not in dtypes:
            raise ValueError(f"Unknown precision {precision}")
        if device == "cpu" and precision != "float32":
            raise ValueError("CPU checks use float32")
        model = ProGenForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            torch_dtype=dtypes[precision],
        )
        return cls(model, device, dtypes[precision])

    @classmethod
    def tiny(cls, seed=0):
        """Random, 2-layer CPU fixture using the real ProGen implementation, not weights."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = ProGenForCausalLM(
                ProGenConfig(
                    vocab_size=32,
                    n_positions=128,
                    n_ctx=128,
                    n_embd=64,
                    n_layer=2,
                    n_head=8,
                    rotary_dim=8,
                    n_inner=64,
                )
            )
        return cls(model)

    def residue_ids(self, alphabet):
        """Verify the one-residue/one-token assumption before using action indices."""
        token_ids = []
        for residue in alphabet:
            encoded = self.tokenizer.encode(residue).ids
            if len(encoded) != 1 or self.tokenizer.decode(encoded) != residue:
                raise ValueError(f"Residue {residue} does not round-trip as one token")
            token_ids.append(encoded[0])
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("Residue token IDs are not unique")
        return token_ids

    def encode(self, sequences):
        """Encode equal-length prefixes without padding or an artificial stop action."""
        if not sequences or len({len(sequence) for sequence in sequences}) != 1:
            raise ValueError("Backbone batches must have equal-length sequences")
        return torch.tensor(
            [
                self.boundary_id + self.tokenizer.encode(sequence).ids
                for sequence in sequences
            ],
            dtype=torch.long,
            device=self.device,
        )

    @torch.no_grad()
    def forward(self, sequences):
        """Return final-layer states and original logits at every input position."""
        self.model.eval()
        self.work["forward_calls"] += 1
        self.work["input_sequences"] += len(sequences)
        self.work["input_tokens"] += sum(len(sequence) + 1 for sequence in sequences)
        states = self.model.transformer(
            self.encode(sequences), use_cache=False
        ).last_hidden_state
        logits = self.model.lm_head(states).float()
        return states.float(), logits

    @torch.no_grad()
    def step(self, token_ids, past_key_values=None):
        """Consume direction marker or next residue, preserving all forced residues."""
        self.work["forward_calls"] += 1
        self.work["input_sequences"] += token_ids.shape[0]
        self.work["input_tokens"] += token_ids.numel()
        result = self.model.transformer(
            token_ids.to(self.device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden = result.last_hidden_state[:, -1]
        return (
            hidden.float(),
            self.model.lm_head(hidden).float(),
            result.past_key_values,
        )

    @torch.no_grad()
    def sequence_features(self, sequences):
        """Mean final-layer residue features; exclude direction marker and no EOS."""
        states, _ = self.forward(sequences)
        return states[:, 1:].mean(dim=1).cpu()


def cache_tree(backbone: FrozenProGen, tree: PrefixTree, batch_size=16):
    """Compute each distinct prefix once, grouped by length to avoid padding errors."""
    features = torch.empty(len(tree.prefixes), backbone.hidden_size)
    logits = torch.empty(len(tree.prefixes), len(tree.context.alphabet))
    residue_ids = backbone.residue_ids(tree.context.alphabet)
    for length in range(len(tree.context.reference)):
        indices = [
            index for index, prefix in enumerate(tree.prefixes) if len(prefix) == length
        ]
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            states, full_logits = backbone.forward(
                [tree.prefixes[index] for index in selected]
            )
            features[selected] = states[:, -1].cpu()
            logits[selected] = full_logits[:, -1, residue_ids].cpu()
    if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
        raise ValueError("Non-finite frozen backbone output")
    return features, logits
