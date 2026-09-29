"""Independent model adapters. Each returns named raw scores for one variant."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .data import Reference, Variant


QTL_ROOT = Path("/data/yitian_workspace/DNA/benchmarks/qtl_Borzoi")
AG_RESEARCH = Path("/data/yitian_workspace/DNA/alphagenome_research/src")


class NTv3:
    name = "ntv3_100m_post"
    default_score = "position_llr"
    direction = -1  # lower ALT-vs-REF likelihood => higher pathogenicity proxy

    def __init__(self, reference: Reference, checkpoint: Path, device: str = "cuda",
                 length: int = 32768):
        self.reference, self.checkpoint, self.device, self.length = reference, checkpoint, device, length
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "model.safetensors")
        self.model = self.tokenizer = self.species_ids = None

    def load(self):
        import torch
        # Transformers copies trusted model code into this import cache.
        # The default ~/.cache is read-only on some benchmark nodes.
        modules_cache = Path(__file__).resolve().parents[1] / "runs/.hf_modules"
        modules_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_MODULES_CACHE", str(modules_cache))
        from transformers import AutoModel, AutoTokenizer
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("NTv3 requires CUDA for the requested device")
        # The config's auto_map points to a separately cached official base-code revision.
        common = dict(local_files_only=True, trust_remote_code=True,
                      revision="b7292f0bd1b5b004d28561783a1890e1ec6f94fd",
                      code_revision="0ecff3637f0d3ba5b686d1095083218157c2ca34",
                      cache_dir=str(QTL_ROOT / ".benchmark_deps/ntv3_hf"))
        model_id = "InstaDeepAI/NTv3_100M_post"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32,
                                               **common).to(self.device).eval()
        self.species_ids = self.model.encode_species(["human"]).to(self.device)

    def score(self, variant: Variant) -> dict[str, float]:
        import torch
        if self.model is None:
            self.load()
        ref, alt, offset = self.reference.window(variant, self.length)
        base_ids = {b: self.tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        if len(set(base_ids.values())) != 4 or any(x is None for x in base_ids.values()):
            raise ValueError("NTv3 tokenizer does not expose single-base A/C/G/T tokens")

        def forward(seq):
            ids = self.tokenizer(seq, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if ids.shape != (1, self.length):
                raise ValueError(f"Expected single-base tokens of length {self.length}, got {tuple(ids.shape)}")
            with torch.inference_mode():
                output = self.model(input_ids=ids.to(self.device), species_ids=self.species_ids)
            return ids[0], output

        ref_ids, ref_output = forward(ref)
        logits = ref_output.logits[0].float().cpu()
        position = float(logits[offset, base_ids[variant.alt]] - logits[offset, base_ids[variant.ref]])
        valid = torch.isin(ref_ids, torch.tensor(list(base_ids.values())))
        ref_mean = torch.log_softmax(logits, -1).gather(1, ref_ids[:, None]).squeeze(1)[valid].mean()
        ref_tracks = getattr(ref_output, "bigwig_tracks_logits", None)
        del logits, ref_output
        alt_ids, alt_output = forward(alt)
        alt_logits = alt_output.logits[0].float().cpu()
        valid_alt = torch.isin(alt_ids, torch.tensor(list(base_ids.values())))
        alt_mean = torch.log_softmax(alt_logits, -1).gather(1, alt_ids[:, None]).squeeze(1)[valid_alt].mean()
        result = {"position_llr": position, "seq_pllr": float(alt_mean - ref_mean)}
        alt_tracks = getattr(alt_output, "bigwig_tracks_logits", None)
        if ref_tracks is not None and alt_tracks is not None:
            a, b = ref_tracks.float().cpu(), alt_tracks.float().cpu()
            delta = torch.log2((b + 1e-6) / (a + 1e-6))
            middle = delta[:, int(delta.shape[1] * .3125):int(delta.shape[1] * .6875)]
            result["log2fc_max"] = float(middle.abs().max())
        return result


class Borzoi:
    name = "borzoi_replicate_0"
    default_score = "rna_max_abs_delta"
    direction = 1

    def __init__(self, reference: Reference, checkpoint: Path, params: Path, targets: Path):
        self.reference, self.checkpoint, self.params, self.targets = reference, checkpoint, params, targets
        for path in (checkpoint, params, targets):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.model = None

    def load(self):
        import pandas as pd
        from baskerville import seqnn
        parameters = json.loads(self.params.read_text())["model"]
        self.length = int(parameters["seq_length"])
        targets = pd.read_csv(self.targets, sep="\t", index_col=0)
        # Full RNA set only; this is a regulatory-effect proxy, not pathogenicity training.
        indexes = targets.index[targets["description"].str.startswith("RNA:")].to_numpy()
        if len(indexes) == 0:
            raise ValueError("No RNA targets found")
        self.model = seqnn.SeqNN(parameters)
        self.model.restore(str(self.checkpoint), 0)
        self.model.build_slice(indexes, False)

    def score(self, variant: Variant) -> dict[str, float]:
        if self.model is None:
            self.load()
        ref, alt, _ = self.reference.window(variant, self.length)
        def encode(sequence):
            values = np.zeros((len(sequence), 4), dtype=np.float32)
            for base, index in zip("ACGT", range(4)):
                values[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8) == ord(base), index] = 1
            return values[None]
        ref_pred = self.model.model(encode(ref), training=False).numpy()[0]
        alt_pred = self.model.model(encode(alt), training=False).numpy()[0]
        delta = alt_pred.astype(np.float32) - ref_pred.astype(np.float32)
        return {"rna_max_abs_delta": float(np.max(np.abs(delta)))}


class AlphaGenome:
    name = "alphagenome_all_folds"
    default_score = "raw_max_abs"
    direction = 1

    def __init__(self, checkpoint: Path, fasta: Path, reference_dir: Path, device: int = 0,
                 length: int = 2**20):
        self.checkpoint, self.fasta, self.reference_dir = checkpoint, fasta, reference_dir
        self.device, self.length = device, length
        self.model = None

    def load(self):
        import sys
        import jax
        from alphagenome.models import dna_model as api
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(self.checkpoint)
        if str(AG_RESEARCH) not in sys.path:
            sys.path.insert(0, str(AG_RESEARCH))
        from alphagenome_research.model import dna_model
        assets = {
            "gtf_feather_path": self.reference_dir / "gencode41_basic_nort.gtf.feather",
            "splice_site_starts_feather_path": self.reference_dir / "gencode41_basic_nort.splice_sites_starts.feather",
            "splice_site_ends_feather_path": self.reference_dir / "gencode41_basic_nort.splice_sites_ends.feather",
            "pas_feather_path": self.reference_dir / "polyadb_human_v3.feather",
        }
        for path in (self.fasta, Path(str(self.fasta) + ".fai"), *assets.values()):
            if not path.is_file():
                raise FileNotFoundError(path)
        devices = jax.devices("gpu")
        if self.device >= len(devices):
            raise RuntimeError(f"AlphaGenome GPU {self.device} unavailable")
        settings = {
            api.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
                fasta_path=str(self.fasta), **{key: str(value) for key, value in assets.items()}),
            api.Organism.MUS_MUSCULUS: dna_model.OrganismSettings(),
        }
        self.model = dna_model.create(str(self.checkpoint), organism_settings=settings,
                                      device=devices[self.device])

    def score(self, variant: Variant) -> dict[str, float]:
        from alphagenome.data import genome
        from alphagenome.models import variant_scorers
        if self.model is None:
            self.load()
        candidate = genome.Variant(chromosome=variant.chr_name, position=variant.pos,
                                   reference_bases=variant.ref, alternate_bases=variant.alt)
        interval = candidate.reference_interval.resize(self.length)
        scorers = list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.values())
        results = self.model.score_variant(interval, candidate, variant_scorers=scorers)
        table = variant_scorers.tidy_scores(results, match_gene_strand=True)
        if table is None or table.empty:
            return {}
        result = {}
        for column, metric in (("quantile_score", "quantile_max_abs"),
                               ("raw_score", "raw_max_abs")):
            if column in table:
                values = np.asarray(table[column], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    result[metric] = float(np.max(np.abs(finite)))
        return result


def make_model(name: str, reference: Reference, device: str = "cuda"):
    if name == "ntv3":
        snapshot = (QTL_ROOT / ".benchmark_deps/ntv3_hf/models--InstaDeepAI--NTv3_100M_post"
                    / "snapshots/b7292f0bd1b5b004d28561783a1890e1ec6f94fd")
        return NTv3(reference, snapshot, device=device)
    if name == "borzoi":
        return Borzoi(reference, QTL_ROOT / "models/replicate_0/model0_best.h5",
                      QTL_ROOT / "configs/params_pred.json", QTL_ROOT / "configs/targets_human.txt")
    if name == "alphagenome":
        return AlphaGenome(Path("/data/yitian_workspace/DNA/alphagenome_models/all_folds"),
                           reference.path, QTL_ROOT / "reference/alphagenome")
    raise ValueError(f"Unknown model {name!r}")
