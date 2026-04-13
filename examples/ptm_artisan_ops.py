from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import biotite.structure as struc
import hydride
import numpy as np
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES
from atomworks.io import parse
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray
from pydantic import BaseModel, Field

from artisan.operations.base.operation_definition import OperationDefinition
from artisan.orchestration.engine.inputs import resolve_output_reference
from artisan.schemas import ArtifactResult, GroupByStrategy
from artisan.schemas.artifact.file_ref import FileRefArtifact
from artisan.schemas.artifact.metric import MetricArtifact
from artisan.schemas.artifact.registry import ArtifactTypeDef
from artisan.schemas.artifact.types import ArtifactTypes
from artisan.schemas.specs.input_models import ExecuteInput, PostprocessInput, PreprocessInput
from artisan.schemas.specs.input_spec import InputSpec
from artisan.schemas.specs.output_spec import OutputSpec
from artisan.utils.hashing import compute_artifact_id
from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from spoof_cif import build_ptm_binder_workshop_inputs, extract_chain_sequence

PHOSPHATE_ATOMS = ("P", "O1P", "O2P", "O3P")
HBOND_CUTOFF_DIST = 2.5
HBOND_CUTOFF_ANGLE = 120.0
HBOND_PH = 7.0
HBOND_REQUIRE_CROSS_RESIDUE = True
SASA_PROBE_RADIUS = 1.4
SASA_VDW_RADII = "Single"
SASA_POINT_NUMBER = 1000
SASA_POINT_DISTR = "Fibonacci"
BINDER_CHAIN_ID = "A"


def _jsonify(value: Any) -> Any:
    return json.loads(json.dumps(value, default=float))


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_stem_and_extension(path: Path) -> tuple[str, str | None]:
    extension = "".join(path.suffixes)
    if extension:
        return path.name[: -len(extension)], extension
    return path.name, None


def make_file_ref_draft(
    path: str | Path,
    *,
    step_number: int,
    metadata: dict[str, Any] | None = None,
) -> FileRefArtifact:
    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    original_name, extension = _file_stem_and_extension(path)
    return FileRefArtifact.draft(
        path=str(path),
        content_hash=compute_artifact_id(content),
        size_bytes=len(content),
        step_number=step_number,
        metadata=metadata or {},
        original_name=original_name,
        extension=extension,
    )


def load_atom_array(path: str | Path, *, hydrogen_policy: str = "remove") -> AtomArray:
    parsed = parse(
        path,
        hydrogen_policy=hydrogen_policy,
        keep_cif_block=True,
    )
    assemblies = parsed.get("assemblies", {})
    if assemblies:
        if "1" in assemblies:
            return assemblies["1"][0]
        first_key = next(iter(assemblies))
        return assemblies[first_key][0]
    return parsed["asym_unit"][0]


def resolve_file_ref_paths(delta_root: str | Path, output_ref) -> list[Path]:
    import polars as pl

    delta_root = Path(delta_root).resolve()
    artifact_ids = resolve_output_reference(output_ref, delta_root)
    if not artifact_ids:
        return []

    table_path = delta_root / ArtifactTypeDef.get_table_path(ArtifactTypes.FILE_REF)
    rows = (
        pl.scan_delta(str(table_path))
        .filter(pl.col("artifact_id").is_in(artifact_ids))
        .select("artifact_id", "path")
        .collect()
    )
    path_by_id = {
        row["artifact_id"]: Path(row["path"]).expanduser().resolve()
        for row in rows.iter_rows(named=True)
    }
    return [path_by_id[artifact_id] for artifact_id in artifact_ids if artifact_id in path_by_id]


def paired_common_indices(
    reference: AtomArray,
    mobile: AtomArray,
    ref_mask: np.ndarray,
    mobile_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref_indices = np.flatnonzero(ref_mask)
    mobile_indices = np.flatnonzero(mobile_mask)
    ref_lookup = {
        (str(reference.chain_id[i]), int(reference.res_id[i]), str(reference.atom_name[i])): i
        for i in ref_indices
    }
    mobile_lookup = {
        (str(mobile.chain_id[i]), int(mobile.res_id[i]), str(mobile.atom_name[i])): i
        for i in mobile_indices
    }
    common_keys = [key for key in ref_lookup if key in mobile_lookup]
    if not common_keys:
        raise ValueError("No common atoms found between the reference and mobile selections")
    ref_paired = np.array([ref_lookup[key] for key in common_keys], dtype=int)
    mobile_paired = np.array([mobile_lookup[key] for key in common_keys], dtype=int)
    return ref_paired, mobile_paired


def align_mobile_on_binder_backbone(
    reference: AtomArray,
    mobile: AtomArray,
    binder_chain_id: str = BINDER_CHAIN_ID,
) -> tuple[AtomArray, float]:
    ref_mask = (
        (reference.chain_id == binder_chain_id)
        & np.isin(reference.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
    )
    mobile_mask = (
        (mobile.chain_id == binder_chain_id)
        & np.isin(mobile.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
    )
    ref_idx, mobile_idx = paired_common_indices(reference, mobile, ref_mask, mobile_mask)
    _, transform = struc.superimpose(reference[ref_idx], mobile[mobile_idx])
    mobile_aligned = transform.apply(mobile)
    alignment_rmsd = float(struc.rmsd(reference[ref_idx], mobile_aligned[mobile_idx]))
    return mobile_aligned, alignment_rmsd


def rmsd_for_masks(
    reference: AtomArray,
    mobile: AtomArray,
    ref_mask: np.ndarray,
    mobile_mask: np.ndarray,
    *,
    allow_mismatch: bool = False,
) -> tuple[float, int]:
    if allow_mismatch:
        ref_idx, mobile_idx = paired_common_indices(reference, mobile, ref_mask, mobile_mask)
    else:
        ref_idx = np.flatnonzero(ref_mask)
        mobile_idx = np.flatnonzero(mobile_mask)
        if len(ref_idx) != len(mobile_idx):
            raise ValueError(
                f"Selection size mismatch: reference={len(ref_idx)} mobile={len(mobile_idx)}"
            )
    if len(ref_idx) == 0:
        raise ValueError("Selection matched no atoms")
    return float(struc.rmsd(reference[ref_idx], mobile[mobile_idx])), len(ref_idx)


def same_residue(atom_array: AtomArray, atom_i: int, atom_j: int) -> bool:
    return (
        atom_array.chain_id[atom_i] == atom_array.chain_id[atom_j]
        and atom_array.res_id[atom_i] == atom_array.res_id[atom_j]
    )


def format_hbond_connection(
    atom_array: AtomArray,
    donor_idx: int,
    hydrogen_idx: int,
    acceptor_idx: int,
) -> str:
    donor = (
        f"{atom_array.chain_id[donor_idx]}:"
        f"{atom_array.res_name[donor_idx]}{atom_array.res_id[donor_idx]}:"
        f"{atom_array.atom_name[donor_idx]}"
    )
    hydrogen = atom_array.atom_name[hydrogen_idx]
    acceptor = (
        f"{atom_array.chain_id[acceptor_idx]}:"
        f"{atom_array.res_name[acceptor_idx]}{atom_array.res_id[acceptor_idx]}:"
        f"{atom_array.atom_name[acceptor_idx]}"
    )
    return f"{donor} -- {hydrogen} --> {acceptor}"


def prepare_atom_array_for_hbonds(atom_array: AtomArray, ph: float = HBOND_PH) -> AtomArray:
    prepared = atom_array.copy()
    if "H" in prepared.element:
        prepared = prepared[prepared.element != "H"]
    prepared.bonds = struc.connect_via_residue_names(prepared)
    prepared.charge = hydride.estimate_amino_acid_charges(prepared, ph=ph)
    prepared_with_h, _ = hydride.add_hydrogen(prepared)
    prepared_with_h.coord = hydride.relax_hydrogen(prepared_with_h)
    return prepared_with_h


def compute_phosphosite_hbond_metrics(
    atom_array: AtomArray,
    *,
    chain_id: str,
    residue_id: int,
    res_name: str = "PTR",
) -> dict[str, Any]:
    prepared = prepare_atom_array_for_hbonds(atom_array, ph=HBOND_PH)
    phosphosite_mask = (
        (prepared.chain_id == chain_id)
        & (prepared.res_id == residue_id)
        & (prepared.res_name == res_name)
    )
    if not phosphosite_mask.any():
        raise ValueError(f"No atoms found for {chain_id}:{res_name}{residue_id}")

    hbond_result = struc.hbond(
        prepared,
        selection1_type="both",
        cutoff_dist=HBOND_CUTOFF_DIST,
        cutoff_angle=HBOND_CUTOFF_ANGLE,
    )
    triplets = hbond_result[0] if isinstance(hbond_result, tuple) else hbond_result

    all_connections: list[str] = []
    phosphosite_connections: list[str] = []
    for donor_idx, hydrogen_idx, acceptor_idx in triplets:
        if HBOND_REQUIRE_CROSS_RESIDUE and same_residue(prepared, donor_idx, acceptor_idx):
            continue
        connection = format_hbond_connection(
            prepared,
            donor_idx=int(donor_idx),
            hydrogen_idx=int(hydrogen_idx),
            acceptor_idx=int(acceptor_idx),
        )
        all_connections.append(connection)
        if phosphosite_mask[donor_idx] or phosphosite_mask[acceptor_idx]:
            phosphosite_connections.append(connection)

    return {
        "total_hbonds": float(len(all_connections)),
        "phosphosite_hbonds": float(len(phosphosite_connections)),
        "connections": all_connections,
        "phosphosite_connections": phosphosite_connections,
    }


def compute_selection_sasa_metrics(atom_array: AtomArray, mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        raise ValueError("Selection matched no atoms for SASA calculation")

    sasa_kwargs = {
        "probe_radius": SASA_PROBE_RADIUS,
        "vdw_radii": SASA_VDW_RADII,
        "point_number": SASA_POINT_NUMBER,
        "point_distr": SASA_POINT_DISTR,
    }
    full_complex_sasa = struc.sasa(atom_array, **sasa_kwargs)
    isolated_subset = atom_array[mask]
    isolated_sasa = struc.sasa(isolated_subset, **sasa_kwargs)

    total_iso = float(np.nansum(isolated_sasa))
    total_complex = float(np.nansum(full_complex_sasa[mask]))
    buried = float(total_iso - total_complex)
    fraction_buried = float(buried / total_iso) if total_iso > 0 else float("nan")
    return {
        "total_iso": total_iso,
        "total_complex": total_complex,
        "buried": buried,
        "fraction_buried": fraction_buried,
    }


def compute_binder_aligned_rmsd_metrics(
    reference: AtomArray,
    mobile: AtomArray,
    *,
    target_chain_id: str,
    ptm_residue_id: int,
    binder_chain_id: str = BINDER_CHAIN_ID,
    ptm_resname: str = "PTR",
) -> dict[str, float]:
    mobile_aligned, binder_alignment_rmsd = align_mobile_on_binder_backbone(
        reference,
        mobile,
        binder_chain_id=binder_chain_id,
    )

    peptide_mask_ref = reference.chain_id == target_chain_id
    peptide_mask_mobile = mobile_aligned.chain_id == target_chain_id
    peptide_ca_mask_ref = peptide_mask_ref & (reference.atom_name == "CA")
    peptide_ca_mask_mobile = peptide_mask_mobile & (mobile_aligned.atom_name == "CA")

    ptr_mask_ref = (
        peptide_mask_ref
        & (reference.res_id == ptm_residue_id)
        & (reference.res_name == ptm_resname)
    )
    ptr_mask_mobile = (
        peptide_mask_mobile
        & (mobile_aligned.res_id == ptm_residue_id)
        & (mobile_aligned.res_name == ptm_resname)
    )
    po4_mask_ref = ptr_mask_ref & np.isin(reference.atom_name, PHOSPHATE_ATOMS)
    po4_mask_mobile = ptr_mask_mobile & np.isin(mobile_aligned.atom_name, PHOSPHATE_ATOMS)

    whole_peptide_ca_rmsd, _ = rmsd_for_masks(
        reference,
        mobile_aligned,
        peptide_ca_mask_ref,
        peptide_ca_mask_mobile,
    )
    whole_peptide_all_atom_rmsd, _ = rmsd_for_masks(
        reference,
        mobile_aligned,
        peptide_mask_ref,
        peptide_mask_mobile,
        allow_mismatch=True,
    )
    ptr_all_atom_rmsd, _ = rmsd_for_masks(
        reference,
        mobile_aligned,
        ptr_mask_ref,
        ptr_mask_mobile,
        allow_mismatch=True,
    )
    po4_only_rmsd, _ = rmsd_for_masks(
        reference,
        mobile_aligned,
        po4_mask_ref,
        po4_mask_mobile,
        allow_mismatch=True,
    )

    return {
        "binder_backbone_alignment_rmsd": binder_alignment_rmsd,
        "whole_peptide_ca_rmsd": whole_peptide_ca_rmsd,
        "whole_peptide_all_atom_rmsd": whole_peptide_all_atom_rmsd,
        "ptr_all_atom_rmsd": ptr_all_atom_rmsd,
        "po4_only_rmsd": po4_only_rmsd,
    }


class SpoofPTMTarget(OperationDefinition):
    name: ClassVar[str] = "spoof_ptm_target"
    description: ClassVar[str] = "Create a spoofed PTM target CIF plus matching RFD3 JSON input"
    inputs: ClassVar[dict[str, InputSpec]] = {}

    class OutputRole(StrEnum):
        structures = "structures"
        config = "config"
        metrics = "metrics"

    outputs: ClassVar[dict[str, OutputSpec]] = {
        OutputRole.structures: OutputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            description="Spoofed target CIF",
            infer_lineage_from={"inputs": []},
        ),
        OutputRole.config: OutputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            description="RFD3 input JSON",
            infer_lineage_from={"inputs": []},
        ),
        OutputRole.metrics: OutputSpec(
            artifact_type=ArtifactTypes.METRIC,
            description="Workshop metadata for the spoofed target",
            infer_lineage_from={"inputs": []},
        ),
    }

    class Params(BaseModel):
        sequence: str
        binder_length: int = 100
        target_chain_id: str = "B"
        ptm_resname: str = "PTR"
        example_name: str = "pvpnpd_ptr_workshop"
        output_dir: str | None = None

    params: Params

    def execute(self, inputs: ExecuteInput) -> dict[str, Any]:
        output_dir = _ensure_dir(self.params.output_dir or inputs.execute_dir)
        workshop = build_ptm_binder_workshop_inputs(
            name=self.params.example_name,
            sequence=self.params.sequence,
            binder_length=self.params.binder_length,
            out_dir=output_dir,
            target_chain_id=self.params.target_chain_id,
            ptm_resname=self.params.ptm_resname,
        )
        workshop["cif_path"] = str(Path(workshop["cif_path"]).resolve())
        workshop["json_path"] = str(Path(workshop["json_path"]).resolve())
        workshop.pop("atom_array", None)
        workshop["rfd3_input"] = _jsonify(workshop["rfd3_input"])
        return {"workshop": workshop}

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        workshop = dict(inputs.memory_outputs["workshop"])
        cif_path = Path(workshop["cif_path"])
        json_path = Path(workshop["json_path"])
        structure_draft = make_file_ref_draft(
            cif_path,
            step_number=inputs.step_number,
            metadata={"example_name": workshop["name"], "kind": "spoofed_target"},
        )
        config_draft = make_file_ref_draft(
            json_path,
            step_number=inputs.step_number,
            metadata={"example_name": workshop["name"], "kind": "rfd3_config"},
        )
        metric_content = {
            "example_name": workshop["name"],
            "sequence": workshop["sequence"],
            "tokens": workshop["tokens"],
            "sequence_length": workshop["sequence_length"],
            "ptm_positions": workshop["ptm_positions"],
            "ptm_residue_id": workshop["ptm_residue_id"],
            "binder_length": self.params.binder_length,
            "target_chain_id": self.params.target_chain_id,
        }
        metric_draft = MetricArtifact.draft(
            content=metric_content,
            original_name=f"{workshop['name']}_spoof_metrics.json",
            step_number=inputs.step_number,
            metadata={"example_name": workshop["name"]},
        )
        return ArtifactResult(
            success=True,
            artifacts={
                self.OutputRole.structures: [structure_draft],
                self.OutputRole.config: [config_draft],
                self.OutputRole.metrics: [metric_draft],
            },
            metadata={
                "example_name": workshop["name"],
                "cif_path": str(cif_path),
                "json_path": str(json_path),
                "ptm_residue_id": workshop["ptm_residue_id"],
                "tokens": workshop["tokens"],
            },
        )


class RunRFD3Design(OperationDefinition):
    name: ClassVar[str] = "run_rfd3_design"
    description: ClassVar[str] = "Generate binder backbones with RFD3 from the spoofed PTM target"

    class InputRole(StrEnum):
        structures = "structures"
        config = "config"

    class OutputRole(StrEnum):
        structures = "structures"
        metrics = "metrics"

    inputs: ClassVar[dict[str, InputSpec]] = {
        InputRole.structures: InputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            materialize=True,
            description="Spoofed target CIF",
        ),
        InputRole.config: InputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            materialize=True,
            description="RFD3 input JSON",
        ),
    }
    outputs: ClassVar[dict[str, OutputSpec]] = {
        OutputRole.structures: OutputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            description="RFD3-designed complexes",
            infer_lineage_from={"inputs": [InputRole.config]},
        ),
        OutputRole.metrics: OutputSpec(
            artifact_type=ArtifactTypes.METRIC,
            description="Compact metadata about the generated RFD3 structures",
            infer_lineage_from={"outputs": [OutputRole.structures]},
        ),
    }
    group_by: ClassVar[GroupByStrategy | None] = GroupByStrategy.ZIP

    class Params(BaseModel):
        ckpt_path: str = "rfd3"
        diffusion_batch_size: int = 1
        n_batches: int = 1
        output_dir: str | None = None

    params: Params = Params()

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        items: list[dict[str, str]] = []
        for group in inputs.grouped():
            structure_artifact = group[self.InputRole.structures]
            config_artifact = group[self.InputRole.config]
            items.append(
                {
                    "structure_path": str(structure_artifact.materialized_path),
                    "config_path": str(config_artifact.materialized_path),
                    "name": config_artifact.original_name or Path(config_artifact.path).stem,
                }
            )
        return {"items": items}

    def execute(self, inputs: ExecuteInput) -> dict[str, Any]:
        output_dir = _ensure_dir(self.params.output_dir or inputs.execute_dir)
        config = RFD3InferenceConfig(
            ckpt_path=self.params.ckpt_path,
            diffusion_batch_size=self.params.diffusion_batch_size,
        )
        engine = RFD3InferenceEngine(**config)
        records: list[dict[str, Any]] = []
        for item in inputs.inputs["items"]:
            outputs = engine.run(
                inputs=item["config_path"],
                out_dir=None,
                n_batches=self.params.n_batches,
            )
            for example_id, per_example_outputs in outputs.items():
                for model_index, output in enumerate(per_example_outputs):
                    base_path = output_dir / f"{example_id}_rfd3_model_{model_index}"
                    saved_path = Path(
                        to_cif_file(
                            output.atom_array,
                            base_path,
                            file_type="cif",
                            include_entity_poly=False,
                        )
                    ).resolve()
                    records.append(
                        {
                            "example_id": example_id,
                            "model_index": model_index,
                            "cif_path": str(saved_path),
                            "metrics": {
                                "example_id": example_id,
                                "model_index": model_index,
                                "output_name": saved_path.stem,
                            },
                        }
                    )
        return {"records": records}

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        structure_drafts: list[FileRefArtifact] = []
        metric_drafts: list[MetricArtifact] = []
        structure_paths: list[str] = []

        for record in inputs.memory_outputs["records"]:
            cif_path = Path(record["cif_path"])
            structure_paths.append(str(cif_path))
            structure_drafts.append(
                make_file_ref_draft(
                    cif_path,
                    step_number=inputs.step_number,
                    metadata={
                        "example_id": record["example_id"],
                        "model_index": record["model_index"],
                    },
                )
            )
            metric_drafts.append(
                MetricArtifact.draft(
                    content=_jsonify(record["metrics"]),
                    original_name=f"{cif_path.stem}_metrics.json",
                    step_number=inputs.step_number,
                    metadata={"example_id": record["example_id"]},
                )
            )

        return ArtifactResult(
            success=True,
            artifacts={
                self.OutputRole.structures: structure_drafts,
                self.OutputRole.metrics: metric_drafts,
            },
            metadata={"structure_paths": structure_paths, "count": len(structure_paths)},
        )


class RunLigandMPNNDesign(OperationDefinition):
    name: ClassVar[str] = "run_ligand_mpnn_design"
    description: ClassVar[str] = "Design binder sequences on chain A with LigandMPNN"

    class InputRole(StrEnum):
        structures = "structures"

    class OutputRole(StrEnum):
        structures = "structures"
        metrics = "metrics"

    inputs: ClassVar[dict[str, InputSpec]] = {
        InputRole.structures: InputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            materialize=True,
            description="RFD3-designed complexes",
        ),
    }
    outputs: ClassVar[dict[str, OutputSpec]] = {
        OutputRole.structures: OutputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            description="LigandMPNN-designed complexes",
            infer_lineage_from={"inputs": [InputRole.structures]},
        ),
        OutputRole.metrics: OutputSpec(
            artifact_type=ArtifactTypes.METRIC,
            description="Sequence recovery and binder sequence summaries",
            infer_lineage_from={"outputs": [OutputRole.structures]},
        ),
    }

    class Params(BaseModel):
        checkpoint_path: str = "ligandmpnn"
        batch_size: int = 4
        remove_waters: bool = True
        designed_chains: list[str] = Field(default_factory=lambda: [BINDER_CHAIN_ID])
        target_chain_id: str = "B"
        is_legacy_weights: bool = True
        output_dir: str | None = None

    params: Params = Params()

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        items: list[dict[str, str]] = []
        for artifact in inputs.input_artifacts[self.InputRole.structures]:
            path = artifact.materialized_path or Path(artifact.path)
            items.append(
                {
                    "path": str(path),
                    "name": artifact.original_name or Path(path).stem,
                }
            )
        return {"items": items}

    def execute(self, inputs: ExecuteInput) -> dict[str, Any]:
        output_dir = _ensure_dir(self.params.output_dir or inputs.execute_dir)
        engine = MPNNInferenceEngine(
            model_type="ligand_mpnn",
            checkpoint_path=self.params.checkpoint_path,
            is_legacy_weights=self.params.is_legacy_weights,
            out_directory=None,
            write_structures=False,
            write_fasta=False,
        )
        records: list[dict[str, Any]] = []
        for item in inputs.inputs["items"]:
            atom_array = load_atom_array(item["path"], hydrogen_policy="remove")
            outputs = engine.run(
                input_dicts=[
                    {
                        "name": item["name"],
                        "batch_size": self.params.batch_size,
                        "remove_waters": self.params.remove_waters,
                        "designed_chains": self.params.designed_chains,
                    }
                ],
                atom_arrays=[atom_array],
            )
            for output in outputs:
                design_idx = int(output.output_dict["design_idx"])
                base_path = output_dir / f"{item['name']}_mpnn_design_{design_idx}"
                output.write_structure(base_path=base_path)
                cif_path = Path(f"{base_path}.cif").resolve()
                records.append(
                    {
                        "cif_path": str(cif_path),
                        "metrics": {
                            "example_id": item["name"],
                            "design_idx": design_idx,
                            "binder_sequence": extract_chain_sequence(output.atom_array, BINDER_CHAIN_ID),
                            "target_sequence": extract_chain_sequence(
                                output.atom_array,
                                self.params.target_chain_id,
                            ),
                            "sequence_recovery": output.output_dict["sequence_recovery"],
                            "ligand_interface_sequence_recovery": output.output_dict[
                                "ligand_interface_sequence_recovery"
                            ],
                        },
                    }
                )
        return {"records": records}

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        structure_drafts: list[FileRefArtifact] = []
        metric_drafts: list[MetricArtifact] = []
        structure_paths: list[str] = []
        metric_rows: list[dict[str, Any]] = []

        for record in inputs.memory_outputs["records"]:
            cif_path = Path(record["cif_path"])
            structure_paths.append(str(cif_path))
            metric_rows.append(record["metrics"])
            structure_drafts.append(
                make_file_ref_draft(
                    cif_path,
                    step_number=inputs.step_number,
                    metadata={"design_idx": record["metrics"]["design_idx"]},
                )
            )
            metric_drafts.append(
                MetricArtifact.draft(
                    content=_jsonify(record["metrics"]),
                    original_name=f"{cif_path.stem}_metrics.json",
                    step_number=inputs.step_number,
                    metadata={"design_idx": record["metrics"]["design_idx"]},
                )
            )

        return ArtifactResult(
            success=True,
            artifacts={
                self.OutputRole.structures: structure_drafts,
                self.OutputRole.metrics: metric_drafts,
            },
            metadata={"structure_paths": structure_paths, "rows": metric_rows},
        )


class RunRF3Refold(OperationDefinition):
    name: ClassVar[str] = "run_rf3_refold"
    description: ClassVar[str] = "Refold the designed complexes with RF3"

    class InputRole(StrEnum):
        structures = "structures"

    class OutputRole(StrEnum):
        structures = "structures"
        metrics = "metrics"

    inputs: ClassVar[dict[str, InputSpec]] = {
        InputRole.structures: InputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            materialize=True,
            description="Designed complexes to refold",
        ),
    }
    outputs: ClassVar[dict[str, OutputSpec]] = {
        OutputRole.structures: OutputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            description="RF3-refolded complexes",
            infer_lineage_from={"inputs": [InputRole.structures]},
        ),
        OutputRole.metrics: OutputSpec(
            artifact_type=ArtifactTypes.METRIC,
            description="RF3 summary confidence metrics",
            infer_lineage_from={"outputs": [OutputRole.structures]},
        ),
    }

    class Params(BaseModel):
        ckpt_path: str = "rf3"
        annotate_b_factor_with_plddt: bool = True
        output_dir: str | None = None

    params: Params = Params()

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        items: list[dict[str, str]] = []
        for artifact in inputs.input_artifacts[self.InputRole.structures]:
            path = artifact.materialized_path or Path(artifact.path)
            items.append(
                {
                    "path": str(path),
                    "name": artifact.original_name or Path(path).stem,
                }
            )
        return {"items": items}

    def execute(self, inputs: ExecuteInput) -> dict[str, Any]:
        output_dir = _ensure_dir(self.params.output_dir or inputs.execute_dir)
        engine = RF3InferenceEngine(
            ckpt_path=self.params.ckpt_path,
            verbose=False,
        )
        records: list[dict[str, Any]] = []
        for item in inputs.inputs["items"]:
            atom_array = load_atom_array(item["path"], hydrogen_policy="remove")
            rf3_input = InferenceInput.from_atom_array(
                atom_array,
                example_id=item["name"],
            )
            outputs = engine.run(
                inputs=rf3_input,
                out_dir=None,
                annotate_b_factor_with_plddt=self.params.annotate_b_factor_with_plddt,
            )
            best_output = outputs[item["name"]][0]
            base_path = output_dir / f"{item['name']}_rf3_refolded"
            saved_path = Path(
                to_cif_file(
                    best_output.atom_array,
                    base_path,
                    file_type="cif",
                    include_entity_poly=False,
                )
            ).resolve()
            records.append(
                {
                    "example_id": item["name"],
                    "cif_path": str(saved_path),
                    "summary_confidences": _jsonify(best_output.summary_confidences),
                }
            )
        return {"records": records}

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        structure_drafts: list[FileRefArtifact] = []
        metric_drafts: list[MetricArtifact] = []
        structure_paths: list[str] = []
        summaries: list[dict[str, Any]] = []

        for record in inputs.memory_outputs["records"]:
            cif_path = Path(record["cif_path"])
            structure_paths.append(str(cif_path))
            summaries.append(record["summary_confidences"])
            structure_drafts.append(
                make_file_ref_draft(
                    cif_path,
                    step_number=inputs.step_number,
                    metadata={"example_id": record["example_id"]},
                )
            )
            metric_drafts.append(
                MetricArtifact.draft(
                    content=record["summary_confidences"],
                    original_name=f"{cif_path.stem}_summary_metrics.json",
                    step_number=inputs.step_number,
                    metadata={"example_id": record["example_id"]},
                )
            )

        return ArtifactResult(
            success=True,
            artifacts={
                self.OutputRole.structures: structure_drafts,
                self.OutputRole.metrics: metric_drafts,
            },
            metadata={"structure_paths": structure_paths, "summaries": summaries},
        )


class BinderAlignedRMSD(OperationDefinition):
    name: ClassVar[str] = "binder_aligned_rmsd"
    description: ClassVar[str] = "Compute tutorial-style RMSDs after aligning on the binder backbone"

    class InputRole(StrEnum):
        reference = "reference"
        mobile = "mobile"

    class OutputRole(StrEnum):
        metrics = "metrics"

    inputs: ClassVar[dict[str, InputSpec]] = {
        InputRole.reference: InputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            materialize=True,
            description="Reference structures, typically the MPNN designs",
        ),
        InputRole.mobile: InputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            materialize=True,
            description="Mobile structures, typically the RF3 outputs",
        ),
    }
    outputs: ClassVar[dict[str, OutputSpec]] = {
        OutputRole.metrics: OutputSpec(
            artifact_type=ArtifactTypes.METRIC,
            description="Binder-aligned peptide and phosphosite RMSDs",
            infer_lineage_from={"inputs": [InputRole.mobile]},
        ),
    }
    group_by: ClassVar[GroupByStrategy | None] = GroupByStrategy.ZIP

    class Params(BaseModel):
        target_chain_id: str = "B"
        ptm_residue_id: int
        ptm_resname: str = "PTR"
        binder_chain_id: str = BINDER_CHAIN_ID

    params: Params

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        items: list[dict[str, str]] = []
        for group in inputs.grouped():
            reference_artifact = group[self.InputRole.reference]
            mobile_artifact = group[self.InputRole.mobile]
            reference_path = reference_artifact.materialized_path or Path(reference_artifact.path)
            mobile_path = mobile_artifact.materialized_path or Path(mobile_artifact.path)
            items.append(
                {
                    "reference_path": str(reference_path),
                    "mobile_path": str(mobile_path),
                    "name": mobile_artifact.original_name or Path(mobile_path).stem,
                }
            )
        return {"items": items}

    def execute(self, inputs: ExecuteInput) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for item in inputs.inputs["items"]:
            reference = load_atom_array(item["reference_path"], hydrogen_policy="remove")
            mobile = load_atom_array(item["mobile_path"], hydrogen_policy="remove")
            metrics = compute_binder_aligned_rmsd_metrics(
                reference,
                mobile,
                target_chain_id=self.params.target_chain_id,
                ptm_residue_id=self.params.ptm_residue_id,
                binder_chain_id=self.params.binder_chain_id,
                ptm_resname=self.params.ptm_resname,
            )
            records.append({"name": item["name"], "metrics": metrics})
        return {"records": records}

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        metric_drafts = [
            MetricArtifact.draft(
                content=_jsonify(record["metrics"]),
                original_name=f"{record['name']}_metrics.json",
                step_number=inputs.step_number,
            )
            for record in inputs.memory_outputs["records"]
        ]
        return ArtifactResult(success=True, artifacts={self.OutputRole.metrics: metric_drafts})


class PhosphositeHBondMetrics(OperationDefinition):
    name: ClassVar[str] = "phosphosite_hbond_metrics"
    description: ClassVar[str] = "Count phosphosite hydrogen bonds on the RF3-refolded complexes"

    class InputRole(StrEnum):
        structures = "structures"

    class OutputRole(StrEnum):
        metrics = "metrics"

    inputs: ClassVar[dict[str, InputSpec]] = {
        InputRole.structures: InputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            materialize=True,
            description="Refolded complexes to score",
        ),
    }
    outputs: ClassVar[dict[str, OutputSpec]] = {
        OutputRole.metrics: OutputSpec(
            artifact_type=ArtifactTypes.METRIC,
            description="Total and phosphosite-specific hydrogen bond counts",
            infer_lineage_from={"inputs": [InputRole.structures]},
        ),
    }

    class Params(BaseModel):
        target_chain_id: str = "B"
        ptm_residue_id: int
        ptm_resname: str = "PTR"

    params: Params

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        items: list[dict[str, str]] = []
        for artifact in inputs.input_artifacts[self.InputRole.structures]:
            path = artifact.materialized_path or Path(artifact.path)
            items.append(
                {
                    "path": str(path),
                    "name": artifact.original_name or Path(path).stem,
                }
            )
        return {"items": items}

    def execute(self, inputs: ExecuteInput) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for item in inputs.inputs["items"]:
            atom_array = load_atom_array(item["path"], hydrogen_policy="remove")
            metrics = compute_phosphosite_hbond_metrics(
                atom_array,
                chain_id=self.params.target_chain_id,
                residue_id=self.params.ptm_residue_id,
                res_name=self.params.ptm_resname,
            )
            records.append({"name": item["name"], "metrics": metrics})
        return {"records": records}

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        metric_drafts = [
            MetricArtifact.draft(
                content=_jsonify(record["metrics"]),
                original_name=f"{record['name']}_metrics.json",
                step_number=inputs.step_number,
            )
            for record in inputs.memory_outputs["records"]
        ]
        return ArtifactResult(success=True, artifacts={self.OutputRole.metrics: metric_drafts})


class SelectionSASAMetrics(OperationDefinition):
    name: ClassVar[str] = "selection_sasa_metrics"
    description: ClassVar[str] = "Compute SASA burial for the phosphotyrosine and its phosphate group"

    class InputRole(StrEnum):
        structures = "structures"

    class OutputRole(StrEnum):
        metrics = "metrics"

    inputs: ClassVar[dict[str, InputSpec]] = {
        InputRole.structures: InputSpec(
            artifact_type=ArtifactTypes.FILE_REF,
            materialize=True,
            description="Refolded complexes to score",
        ),
    }
    outputs: ClassVar[dict[str, OutputSpec]] = {
        OutputRole.metrics: OutputSpec(
            artifact_type=ArtifactTypes.METRIC,
            description="SASA burial for PO4 and full PTR selections",
            infer_lineage_from={"inputs": [InputRole.structures]},
        ),
    }

    class Params(BaseModel):
        target_chain_id: str = "B"
        ptm_residue_id: int
        ptm_resname: str = "PTR"

    params: Params

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        items: list[dict[str, str]] = []
        for artifact in inputs.input_artifacts[self.InputRole.structures]:
            path = artifact.materialized_path or Path(artifact.path)
            items.append(
                {
                    "path": str(path),
                    "name": artifact.original_name or Path(path).stem,
                }
            )
        return {"items": items}

    def execute(self, inputs: ExecuteInput) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for item in inputs.inputs["items"]:
            atom_array = load_atom_array(item["path"], hydrogen_policy="remove")
            ptr_mask = (
                (atom_array.chain_id == self.params.target_chain_id)
                & (atom_array.res_id == self.params.ptm_residue_id)
                & (atom_array.res_name == self.params.ptm_resname)
            )
            po4_mask = ptr_mask & np.isin(atom_array.atom_name, PHOSPHATE_ATOMS)
            metrics = {
                "po4": compute_selection_sasa_metrics(atom_array, po4_mask),
                "ptr": compute_selection_sasa_metrics(atom_array, ptr_mask),
            }
            records.append({"name": item["name"], "metrics": metrics})
        return {"records": records}

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        metric_drafts = [
            MetricArtifact.draft(
                content=_jsonify(record["metrics"]),
                original_name=f"{record['name']}_metrics.json",
                step_number=inputs.step_number,
            )
            for record in inputs.memory_outputs["records"]
        ]
        return ArtifactResult(success=True, artifacts={self.OutputRole.metrics: metric_drafts})
