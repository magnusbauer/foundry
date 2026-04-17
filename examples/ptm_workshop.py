from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import biotite.structure as struc
import hydride
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES
from atomworks.io.utils.io_utils import to_cif_string
from biotite.structure import AtomArray

PTR_SIDECHAIN_ATOMS = (
    "CB",
    "CG",
    "CD1",
    "CD2",
    "CE1",
    "CE2",
    "CZ",
    "OH",
    "P",
    "O1P",
    "O2P",
    "O3P",
)
PHOSPHATE_ATOMS = ("P", "O1P", "O2P", "O3P")

HBOND_CUTOFF_DIST = 2.5
HBOND_CUTOFF_ANGLE = 120.0
HBOND_PH = 7.0
HBOND_REQUIRE_CROSS_RESIDUE = True

SASA_PROBE_RADIUS = 1.4
SASA_VDW_RADII = "Single"
SASA_POINT_NUMBER = 1000
SASA_POINT_DISTR = "Fibonacci"

_MOLSTAR_BLUE = {"r": 37, "g": 99, "b": 235}
_MOLSTAR_GREY = {"r": 148, "g": 163, "b": 184}
_MOLSTAR_RED = {"r": 220, "g": 38, "b": 38}
_MOLSTAR_ORANGE = {"r": 245, "g": 158, "b": 11}
_MOLSTAR_SLATE = {"r": 100, "g": 116, "b": 139}


@dataclass(slots=True)
class MolstarViewSpec:
    custom_data: dict[str, Any]
    color_data: dict[str, Any] | None
    width: int
    height: int


def extract_min_interface_pae(summary_confidences: dict[str, Any]) -> float:
    matrix = summary_confidences.get("chain_pair_pae_min") or []
    values = [
        float(value)
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
        if i != j and value is not None
    ]
    return float(min(values)) if values else np.nan


def paired_backbone_indices(
    reference: AtomArray,
    mobile: AtomArray,
    chain_id: str = "A",
) -> tuple[np.ndarray, np.ndarray]:
    ref_mask = (reference.chain_id == chain_id) & np.isin(
        reference.atom_name, PROTEIN_BACKBONE_ATOM_NAMES
    )
    mobile_mask = (mobile.chain_id == chain_id) & np.isin(
        mobile.atom_name, PROTEIN_BACKBONE_ATOM_NAMES
    )
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
        raise ValueError("No common binder backbone atoms found between structures")
    ref_paired = np.array([ref_lookup[key] for key in common_keys], dtype=int)
    mobile_paired = np.array([mobile_lookup[key] for key in common_keys], dtype=int)
    return ref_paired, mobile_paired


def binder_backbone_rmsd(
    reference: AtomArray,
    mobile: AtomArray,
    chain_id: str = "A",
) -> float:
    ref_idx, mobile_idx = paired_backbone_indices(reference, mobile, chain_id=chain_id)
    _, transform = struc.superimpose(reference[ref_idx], mobile[mobile_idx])
    mobile_fit = transform.apply(mobile)
    return float(struc.rmsd(reference[ref_idx], mobile_fit[mobile_idx]))


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
    ref_mask: np.ndarray,
    mobile_mask: np.ndarray,
) -> tuple[AtomArray, Any, float]:
    ref_idx, mobile_idx = paired_common_indices(reference, mobile, ref_mask, mobile_mask)
    _, transform = struc.superimpose(reference[ref_idx], mobile[mobile_idx])
    mobile_aligned = transform.apply(mobile)
    alignment_rmsd = float(struc.rmsd(reference[ref_idx], mobile_aligned[mobile_idx]))
    return mobile_aligned, transform, alignment_rmsd


def rmsd_for_masks(
    reference: AtomArray,
    mobile: AtomArray,
    ref_mask: np.ndarray,
    mobile_mask: np.ndarray,
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


def atom_triplet_label(atom_array: AtomArray, atom_index: int) -> str:
    return (
        f"{atom_array.chain_id[atom_index]}:"
        f"{atom_array.res_name[atom_index]}{atom_array.res_id[atom_index]}:"
        f"{atom_array.atom_name[atom_index]}"
    )


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
    donor = atom_triplet_label(atom_array, donor_idx)
    hydrogen = atom_array.atom_name[hydrogen_idx]
    acceptor = atom_triplet_label(atom_array, acceptor_idx)
    return f"{donor} -- {hydrogen} --> {acceptor}"


def prepare_atom_array_for_hbonds(atom_array: AtomArray, ph: float = HBOND_PH) -> AtomArray:
    prepared = atom_array.copy()
    if "H" in prepared.element:
        prepared = prepared[prepared.element != "H"]
    for category in list(prepared.get_annotation_categories()):
        annotation = prepared.get_annotation(category)
        if getattr(annotation, "ndim", 1) != 1:
            prepared.del_annotation(category)
    prepared.bonds = struc.connect_via_residue_names(prepared)
    prepared.charge = hydride.estimate_amino_acid_charges(prepared, ph=ph)
    prepared_with_h, _ = hydride.add_hydrogen(prepared)
    prepared_with_h.coord = hydride.relax_hydrogen(prepared_with_h)
    return prepared_with_h


def compute_phosphosite_hbond_metrics(
    atom_array: AtomArray,
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

    all_connections = []
    phosphosite_connections = []
    phosphosite_records = []
    for donor_idx, hydrogen_idx, acceptor_idx in triplets:
        if HBOND_REQUIRE_CROSS_RESIDUE and same_residue(prepared, donor_idx, acceptor_idx):
            continue

        connection = format_hbond_connection(
            prepared,
            donor_idx=donor_idx,
            hydrogen_idx=hydrogen_idx,
            acceptor_idx=acceptor_idx,
        )
        all_connections.append(connection)

        donor_match = bool(phosphosite_mask[donor_idx])
        acceptor_match = bool(phosphosite_mask[acceptor_idx])
        if donor_match or acceptor_match:
            phosphosite_connections.append(connection)
            phosphosite_records.append(
                {
                    "donor_idx": int(donor_idx),
                    "hydrogen_idx": int(hydrogen_idx),
                    "acceptor_idx": int(acceptor_idx),
                    "donor_label": atom_triplet_label(prepared, donor_idx),
                    "hydrogen_label": atom_triplet_label(prepared, hydrogen_idx),
                    "acceptor_label": atom_triplet_label(prepared, acceptor_idx),
                    "donor_acceptor_distance": float(
                        np.linalg.norm(prepared.coord[donor_idx] - prepared.coord[acceptor_idx])
                    ),
                    "hydrogen_acceptor_distance": float(
                        np.linalg.norm(prepared.coord[hydrogen_idx] - prepared.coord[acceptor_idx])
                    ),
                }
            )

    return {
        "prepared_structure": prepared,
        "total_hbonds": float(len(all_connections)),
        "phosphosite_hbonds": float(len(phosphosite_connections)),
        "connections": all_connections,
        "phosphosite_connections": phosphosite_connections,
        "phosphosite_records": phosphosite_records,
    }


def _molstar_viewer(
    spec: MolstarViewSpec,
) -> widgets.Widget:
    try:
        from ipymolstar import PDBeMolstar
    except ImportError as error:
        msg = (
            "ipymolstar is required for the workshop structure viewer. "
            "Install it in the notebook setup cell."
        )
        raise ImportError(msg) from error

    viewer = PDBeMolstar(
        height=f"{spec.height}px",
        width=f"{spec.width}px",
        hide_controls=True,
        hide_expand_icon=True,
    )
    viewer.layout = widgets.Layout(width=f"{spec.width}px", height=f"{spec.height}px")
    viewer.custom_data = spec.custom_data
    viewer.color_data = spec.color_data
    return viewer


def _molstar_spec(
    atom_array: AtomArray,
    *,
    color_params: list[dict[str, Any]] | None = None,
    width: int = 520,
    height: int = 430,
) -> MolstarViewSpec:
    return MolstarViewSpec(
        custom_data={
            "data": to_cif_string(
                atom_array,
                include_entity_poly=False,
                _allow_ambiguous_bond_annotations=True,
            ),
            "format": "mmcif",
            "binary": False,
        },
        color_data=(
            {
                "data": color_params,
                "nonSelectedColor": None,
                "keepColors": False,
                "keepRepresentations": False,
            }
            if color_params
            else None
        ),
        width=width,
        height=height,
    )


def _update_molstar_viewer(viewer: widgets.Widget, spec: MolstarViewSpec) -> None:
    viewer.layout = widgets.Layout(width=f"{spec.width}px", height=f"{spec.height}px")
    viewer.custom_data = spec.custom_data
    viewer.color_data = spec.color_data


def _chain_color_param(chain_id: str, color: dict[str, int]) -> dict[str, Any]:
    return {"auth_asym_id": chain_id, "color": color}


def _residue_highlight_param(
    chain_id: str,
    residue_id: int,
    *,
    color: dict[str, int],
    tooltip: str | None = None,
) -> dict[str, Any]:
    param: dict[str, Any] = {
        "auth_asym_id": chain_id,
        "auth_residue_number": int(residue_id),
        "color": color,
        "sideChain": True,
    }
    if tooltip:
        param["tooltip"] = tooltip
    return param


def _apply_chain_map(atom_array: AtomArray, chain_map: dict[str, str]) -> AtomArray:
    remapped = atom_array.copy()
    remapped.chain_id = np.array(
        [chain_map.get(str(chain_id), str(chain_id)) for chain_id in remapped.chain_id],
        dtype=remapped.chain_id.dtype,
    )
    return remapped


def _overlay_chain_maps(
    reference: AtomArray,
    mobile: AtomArray,
) -> tuple[dict[str, str], dict[str, str]]:
    mobile_chains = list(dict.fromkeys(str(chain_id) for chain_id in mobile.chain_id))
    reference_chains = list(dict.fromkeys(str(chain_id) for chain_id in reference.chain_id))

    fallback_ids = [char for char in "XYZUVWQRSTLMNOPKJIHGFEDCBA" if char not in mobile_chains]
    ref_map: dict[str, str] = {}
    for index, chain_id in enumerate(reference_chains):
        if index < len(fallback_ids):
            ref_map[chain_id] = fallback_ids[index]
        else:
            ref_map[chain_id] = chain_id

    mobile_map = {chain_id: chain_id for chain_id in mobile_chains}
    return ref_map, mobile_map


def _build_overlay_atom_array(
    reference: AtomArray,
    mobile: AtomArray,
) -> tuple[AtomArray, dict[str, str], dict[str, str]]:
    reference_map, mobile_map = _overlay_chain_maps(reference, mobile)
    reference_overlay = _apply_chain_map(reference, reference_map)
    mobile_overlay = _apply_chain_map(mobile, mobile_map)
    return struc.concatenate([reference_overlay, mobile_overlay]), reference_map, mobile_map


def make_browser_structure_spec(
    primary: AtomArray,
    secondary: AtomArray | None = None,
    *,
    width: int = 520,
    height: int = 430,
    ptr_chain: str = "B",
    ptr_resi: int | None = 7,
) -> MolstarViewSpec:
    if secondary is None:
        color_params = [
            _chain_color_param("A", _MOLSTAR_BLUE),
            _chain_color_param(ptr_chain, _MOLSTAR_GREY),
        ]
        if ptr_resi is not None:
            color_params.append(
                _residue_highlight_param(
                    ptr_chain,
                    ptr_resi,
                    color=_MOLSTAR_ORANGE,
                    tooltip=f"{ptr_chain}:PTR{ptr_resi}",
                )
            )
        return _molstar_spec(
            primary,
            color_params=color_params,
            width=width,
            height=height,
        )

    overlay_atom_array, reference_map, mobile_map = _build_overlay_atom_array(primary, secondary)
    color_params = []
    for chain_id in dict.fromkeys(reference_map.values()):
        color_params.append(_chain_color_param(chain_id, _MOLSTAR_SLATE))
    for chain_id in dict.fromkeys(mobile_map.values()):
        color_params.append(_chain_color_param(chain_id, _MOLSTAR_RED))
    if ptr_resi is not None:
        if ptr_chain in reference_map:
            color_params.append(
                _residue_highlight_param(
                    reference_map[ptr_chain],
                    ptr_resi,
                    color=_MOLSTAR_GREY,
                    tooltip=f"reference {ptr_chain}:PTR{ptr_resi}",
                )
            )
        if ptr_chain in mobile_map:
            color_params.append(
                _residue_highlight_param(
                    mobile_map[ptr_chain],
                    ptr_resi,
                    color=_MOLSTAR_ORANGE,
                    tooltip=f"mobile {ptr_chain}:PTR{ptr_resi}",
                )
            )
    return _molstar_spec(
        overlay_atom_array,
        color_params=color_params,
        width=width,
        height=height,
    )


def make_browser_structure_view(
    primary: AtomArray,
    secondary: AtomArray | None = None,
    *,
    width: int = 520,
    height: int = 430,
    ptr_chain: str = "B",
    ptr_resi: int | None = 7,
) -> widgets.Widget:
    return _molstar_viewer(
        make_browser_structure_spec(
            primary,
            secondary,
            width=width,
            height=height,
            ptr_chain=ptr_chain,
            ptr_resi=ptr_resi,
        )
    )


def make_structure_overlay_spec(
    reference: AtomArray,
    mobile: AtomArray,
    zoom_to_selection: dict[str, Any] | None = None,
    *,
    width: int = 700,
    height: int = 500,
    ptr_chain: str = "B",
    ptr_resi: int | None = None,
    reference_color: str = "#64748b",
    mobile_color: str = "#dc2626",
) -> MolstarViewSpec:
    del zoom_to_selection, reference_color, mobile_color
    return make_browser_structure_spec(
        reference,
        secondary=mobile,
        width=width,
        height=height,
        ptr_chain=ptr_chain,
        ptr_resi=ptr_resi,
    )


def make_structure_overlay_view(
    reference: AtomArray,
    mobile: AtomArray,
    zoom_to_selection: dict[str, Any] | None = None,
    *,
    width: int = 700,
    height: int = 500,
    ptr_chain: str = "B",
    ptr_resi: int | None = None,
    reference_color: str = "#64748b",
    mobile_color: str = "#dc2626",
) -> widgets.Widget:
    return _molstar_viewer(
        make_structure_overlay_spec(
            reference,
            mobile,
            zoom_to_selection=zoom_to_selection,
            width=width,
            height=height,
            ptr_chain=ptr_chain,
            ptr_resi=ptr_resi,
            reference_color=reference_color,
            mobile_color=mobile_color,
        )
    )


def make_hbond_view_spec(
    atom_array: AtomArray,
    hbond_records: list[dict[str, Any]],
    chain_id: str,
    residue_id: int,
    *,
    width: int = 700,
    height: int = 500,
) -> MolstarViewSpec:
    color_params: list[dict[str, Any]] = [
        _chain_color_param("A", _MOLSTAR_BLUE),
        _chain_color_param(chain_id, _MOLSTAR_GREY),
        _residue_highlight_param(
            chain_id,
            residue_id,
            color=_MOLSTAR_ORANGE,
            tooltip=f"{chain_id}:PTR{residue_id}",
        ),
    ]
    for record in hbond_records:
        donor_label = str(record["donor_label"])
        acceptor_label = str(record["acceptor_label"])
        donor_chain, donor_residue = donor_label.split(":")[:2]
        acceptor_chain, acceptor_residue = acceptor_label.split(":")[:2]
        donor_res_id = int("".join(char for char in donor_residue if char.isdigit()))
        acceptor_res_id = int("".join(char for char in acceptor_residue if char.isdigit()))
        color_params.append(
            _residue_highlight_param(
                donor_chain,
                donor_res_id,
                color=_MOLSTAR_BLUE,
                tooltip=donor_label,
            )
        )
        color_params.append(
            _residue_highlight_param(
                acceptor_chain,
                acceptor_res_id,
                color=_MOLSTAR_ORANGE,
                tooltip=acceptor_label,
            )
        )
    return _molstar_spec(
        atom_array,
        color_params=color_params,
        width=width,
        height=height,
    )


def make_hbond_view(
    atom_array: AtomArray,
    hbond_records: list[dict[str, Any]],
    chain_id: str,
    residue_id: int,
    *,
    width: int = 700,
    height: int = 500,
) -> widgets.Widget:
    return _molstar_viewer(
        make_hbond_view_spec(
            atom_array,
            hbond_records,
            chain_id,
            residue_id,
            width=width,
            height=height,
        )
    )


def project_points_to_plane(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]
    return centered @ basis.T


def plot_hbond_contact_map(
    atom_array: AtomArray,
    hbond_records: list[dict[str, Any]],
    chain_id: str,
    residue_id: int,
    phosphate_atom_names: Sequence[str],
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    if not hbond_records:
        ax.text(
            0.5,
            0.5,
            "No phosphosite H-bonds detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    residue_mask = (atom_array.chain_id == chain_id) & (atom_array.res_id == residue_id)
    phosphate_mask = residue_mask & np.isin(atom_array.atom_name, phosphate_atom_names)

    residue_indices = set(np.flatnonzero(residue_mask))
    phosphate_indices = set(np.flatnonzero(phosphate_mask))
    donor_indices = {int(record["donor_idx"]) for record in hbond_records}
    acceptor_indices = {int(record["acceptor_idx"]) for record in hbond_records}
    selected_indices = np.array(
        sorted(residue_indices | donor_indices | acceptor_indices),
        dtype=int,
    )

    projected = project_points_to_plane(atom_array.coord[selected_indices])
    point_by_index = {
        atom_idx: projected[position]
        for position, atom_idx in enumerate(selected_indices)
    }

    for record in hbond_records:
        donor_xy = point_by_index[int(record["donor_idx"])]
        acceptor_xy = point_by_index[int(record["acceptor_idx"])]
        ax.plot(
            [donor_xy[0], acceptor_xy[0]],
            [donor_xy[1], acceptor_xy[1]],
            linestyle="--",
            linewidth=1.8,
            color="#f59e0b",
            alpha=0.9,
        )

    residue_coords = np.array([point_by_index[idx] for idx in sorted(residue_indices)])
    phosphate_coords = np.array([point_by_index[idx] for idx in sorted(phosphate_indices)])
    donor_coords = np.array([point_by_index[idx] for idx in sorted(donor_indices)])
    acceptor_coords = np.array([point_by_index[idx] for idx in sorted(acceptor_indices)])

    if len(residue_coords):
        ax.scatter(
            residue_coords[:, 0],
            residue_coords[:, 1],
            s=110,
            color="#fcd34d",
            edgecolors="black",
            linewidths=0.4,
            zorder=2,
        )
    if len(phosphate_coords):
        ax.scatter(
            phosphate_coords[:, 0],
            phosphate_coords[:, 1],
            s=170,
            color="#ef4444",
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
    if len(donor_coords):
        ax.scatter(
            donor_coords[:, 0],
            donor_coords[:, 1],
            s=120,
            color="#2563eb",
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )
    if len(acceptor_coords):
        ax.scatter(
            acceptor_coords[:, 0],
            acceptor_coords[:, 1],
            s=120,
            color="#f59e0b",
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )

    label_indices = sorted(phosphate_indices | donor_indices | acceptor_indices)
    for atom_idx in label_indices:
        x_coord, y_coord = point_by_index[atom_idx]
        ax.text(
            x_coord + 0.15,
            y_coord + 0.15,
            f"{atom_array.res_name[atom_idx]}:{atom_array.atom_name[atom_idx]}",
            fontsize=8,
            zorder=5,
        )

    ax.set_title(f"Phosphosite H-bond sketch around {chain_id}{residue_id}")
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def make_sasa_view_spec(
    atom_array: AtomArray,
    chain_id: str,
    residue_id: int,
    phosphate_atom_names: Sequence[str],
    *,
    width: int = 700,
    height: int = 500,
) -> MolstarViewSpec:
    color_params = [
        _chain_color_param("A", _MOLSTAR_BLUE),
        _chain_color_param(chain_id, _MOLSTAR_GREY),
        _residue_highlight_param(
            chain_id,
            residue_id,
            color=_MOLSTAR_ORANGE,
            tooltip=f"{chain_id}:PTR{residue_id}",
        ),
    ]
    if phosphate_atom_names:
        color_params.append(
            _residue_highlight_param(
                chain_id,
                residue_id,
                color=_MOLSTAR_RED,
                tooltip="Phosphate atoms highlighted in red",
            )
        )
    return _molstar_spec(
        atom_array,
        color_params=color_params,
        width=width,
        height=height,
    )


def make_sasa_view(
    atom_array: AtomArray,
    chain_id: str,
    residue_id: int,
    phosphate_atom_names: Sequence[str],
    *,
    width: int = 700,
    height: int = 500,
) -> widgets.Widget:
    return _molstar_viewer(
        make_sasa_view_spec(
            atom_array,
            chain_id,
            residue_id,
            phosphate_atom_names,
            width=width,
            height=height,
        )
    )


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


def compute_selection_sasa_breakdown(atom_array: AtomArray, mask: np.ndarray) -> pd.DataFrame:
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

    rows = []
    selected_indices = np.flatnonzero(mask)
    for local_idx, atom_idx in enumerate(selected_indices):
        iso = float(isolated_sasa[local_idx])
        complex_val = float(full_complex_sasa[atom_idx])
        buried = float(iso - complex_val)
        rows.append(
            {
                "atom_label": atom_triplet_label(atom_array, atom_idx),
                "isolated_sasa": iso,
                "complex_sasa": complex_val,
                "buried_sasa": buried,
                "fraction_buried": float(buried / iso) if iso > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _selection_centroid(atom_array: AtomArray, mask: np.ndarray) -> np.ndarray | None:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return None
    return atom_array.coord[indices].mean(axis=0)


def _selection_min_distance(
    atom_array: AtomArray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    left_indices = np.flatnonzero(left_mask)
    right_indices = np.flatnonzero(right_mask)
    if len(left_indices) == 0 or len(right_indices) == 0:
        return np.nan
    deltas = (
        atom_array.coord[left_indices][:, None, :]
        - atom_array.coord[right_indices][None, :, :]
    )
    return float(np.sqrt(np.sum(deltas * deltas, axis=-1).min()))


def build_rfd3_metrics_df(
    records: Sequence[dict[str, Any]],
    summary_df: pd.DataFrame,
    *,
    binder_chain_id: str = "A",
    target_chain_id: str = "B",
    ptr_residue_id: int = 7,
) -> pd.DataFrame:
    browser_df = summary_df.reset_index(drop=True).copy()
    if not records:
        return browser_df

    reference = records[0]["atom_array"]
    extra_rows = []
    for record in records:
        atom_array = record["atom_array"]
        binder_mask = atom_array.chain_id == binder_chain_id
        binder_backbone_mask = binder_mask & np.isin(
            atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES
        )
        ptr_mask = (
            (atom_array.chain_id == target_chain_id)
            & (atom_array.res_id == ptr_residue_id)
            & (atom_array.res_name == "PTR")
        )
        phosphate_mask = ptr_mask & np.isin(atom_array.atom_name, PHOSPHATE_ATOMS)
        binder_centroid = _selection_centroid(atom_array, binder_backbone_mask)
        phosphate_centroid = _selection_centroid(atom_array, phosphate_mask)
        centroid_distance = (
            float(np.linalg.norm(binder_centroid - phosphate_centroid))
            if binder_centroid is not None and phosphate_centroid is not None
            else np.nan
        )
        extra_rows.append(
            {
                "atom_count": int(len(atom_array)),
                "binder_atom_count": int(np.count_nonzero(binder_mask)),
                "target_atom_count": int(np.count_nonzero(atom_array.chain_id == target_chain_id)),
                "binder_backbone_rmsd_to_first": binder_backbone_rmsd(
                    reference,
                    atom_array,
                    chain_id=binder_chain_id,
                ),
                "binder_min_ptr_distance": _selection_min_distance(
                    atom_array,
                    binder_mask,
                    ptr_mask,
                ),
                "binder_backbone_centroid_to_ptr": centroid_distance,
            }
        )

    return pd.concat([browser_df, pd.DataFrame(extra_rows)], axis=1)


def build_rf3_metrics_df(
    records: Sequence[dict[str, Any]],
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    browser_df = summary_df.reset_index(drop=True).copy()
    if browser_df.empty:
        return browser_df
    browser_df["overall_plddt_pct"] = browser_df["overall_plddt"] * 100.0
    browser_df["min_pae"] = [
        extract_min_interface_pae(record["summary_confidences"])
        for record in records
    ]
    return browser_df


def _metric_label(name: str, metric_labels: dict[str, str] | None) -> str:
    if metric_labels and name in metric_labels:
        return metric_labels[name]
    return name.replace("_", " ")


def _format_cell_value(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _numeric_metric_columns(
    metrics_df: pd.DataFrame,
    *,
    exclude: Sequence[str] = (),
) -> list[str]:
    excluded = set(exclude)
    return [
        column
        for column in metrics_df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(metrics_df[column])
    ]


def _build_metric_scatter(
    metrics_df: pd.DataFrame,
    *,
    label_column: str,
    x_column: str,
    y_column: str,
    z_column: str | None,
    selected_index: int,
    title: str,
    metric_labels: dict[str, str] | None = None,
    plot_height: int = 390,
) -> go.Figure:
    x_label = _metric_label(x_column, metric_labels)
    y_label = _metric_label(y_column, metric_labels)
    z_label = _metric_label(z_column, metric_labels) if z_column else None

    fig = go.Figure()

    hover_lines = [
        "%{text}<br>",
        f"{x_label}: ",
        "%{x:.3f}<br>",
        f"{y_label}: ",
        "%{y:.3f}",
    ]
    if z_column:
        # Use coloraxis so the colorbar is part of the layout — reliable in FigureWidget.
        marker: dict[str, Any] = {
            "size": 10,
            "color": metrics_df[z_column],
            "coloraxis": "coloraxis",
            "line": {"width": 1, "color": "white"},
        }
        customdata = metrics_df[[z_column]].to_numpy()
        hover_lines.extend(["<br>", f"{z_label}: ", "%{customdata[0]:.3f}"])
    else:
        marker = {
            "size": 10,
            "color": "#60a5fa",
            "line": {"width": 1, "color": "white"},
        }
        customdata = None
    hover_lines.append("<extra></extra>")

    fig.add_trace(
        go.Scatter(
            x=metrics_df[x_column],
            y=metrics_df[y_column],
            mode="markers",
            text=metrics_df[label_column],
            customdata=customdata,
            marker=marker,
            hovertemplate="".join(hover_lines),
            showlegend=False,
        )
    )

    # Selection ring — always add as trace 1 so the index stays stable.
    selected_row = metrics_df.iloc[selected_index]
    sel_visible = bool(pd.notna(selected_row[x_column]) and pd.notna(selected_row[y_column]))
    fig.add_trace(
        go.Scatter(
            x=[selected_row[x_column]] if sel_visible else [None],
            y=[selected_row[y_column]] if sel_visible else [None],
            mode="markers",
            marker={
                "size": 22,
                "color": "rgba(249, 115, 22, 0.2)",
                "symbol": "circle",
                "line": {"width": 3, "color": "#f97316"},
            },
            hoverinfo="skip",
            showlegend=False,
            visible=sel_visible,
        )
    )

    layout_kw: dict[str, Any] = {
        "title": title,
        "xaxis_title": x_label,
        "yaxis_title": y_label,
        "height": plot_height,
        "margin": dict(l=90 if z_column else 0, r=28, t=45, b=30 if z_column else 0),
    }
    if z_column:
        layout_kw["coloraxis"] = {
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {
                "title": {"text": z_label, "side": "bottom"},
                "thickness": 14,
                "len": 0.9,
                "x": 0,
                "xanchor": "right",
                "xpad": 5,
            },
        }
    fig.update_layout(**layout_kw)
    return fig


def _make_figure_widget(
    fig: go.Figure,
    *,
    pixel_width: int | None = None,
) -> tuple[go.FigureWidget, widgets.Widget]:
    if pixel_width is not None:
        fig = go.Figure(fig)
        fig.update_layout(autosize=False, width=pixel_width)
    fw = go.FigureWidget(fig)
    css_width = f"{pixel_width}px" if pixel_width is not None else "46%"
    panel = widgets.Box([fw], layout=widgets.Layout(width=css_width))
    return fw, panel


def make_studio_metric_browser(
    *,
    records: Sequence[dict[str, Any]],
    metrics_df: pd.DataFrame,
    structure_factory: Callable[[dict[str, Any]], Any],
    label_column: str,
    title: str,
    metric_columns: Sequence[str] | None = None,
    default_x: str | None = None,
    default_y: str | None = None,
    default_z: str | None = None,
    metric_labels: dict[str, str] | None = None,
    note_html: str | None = None,
    table_columns: Sequence[str] | None = None,
    plot_height: int = 390,
    plot_width: int = 420,
) -> widgets.Widget:
    browser_df = metrics_df.reset_index(drop=True).copy()
    if len(records) != len(browser_df):
        raise ValueError("records and metrics_df must have the same number of rows")

    resolved_metric_columns = list(
        metric_columns or _numeric_metric_columns(browser_df, exclude=[label_column])
    )
    if len(resolved_metric_columns) < 2:
        raise ValueError("Need at least two numeric metric columns to build a scatter browser")

    if default_x not in resolved_metric_columns:
        default_x = resolved_metric_columns[0]
    if default_y not in resolved_metric_columns or default_y == default_x:
        default_y = next(
            (column for column in resolved_metric_columns if column != default_x),
            resolved_metric_columns[0],
        )
    if default_z not in resolved_metric_columns:
        default_z = next(
            (
                column
                for column in resolved_metric_columns
                if column not in {default_x, default_y}
            ),
            default_y,
        )

    metric_options = [
        (_metric_label(column, metric_labels), column)
        for column in resolved_metric_columns
    ]
    z_metric_options = [("None", None), *metric_options]
    del table_columns

    current_index = {"value": 0}
    prev_button = widgets.Button(description="Previous")
    next_button = widgets.Button(description="Next")
    status_html = widgets.HTML()
    x_dropdown = widgets.Dropdown(options=metric_options, value=default_x, description="X")
    y_dropdown = widgets.Dropdown(options=metric_options, value=default_y, description="Y")
    z_dropdown = widgets.Dropdown(options=z_metric_options, value=default_z, description="Z")
    initial_structure = structure_factory(records[0])
    structure_cache: dict[int, Any] = {0: initial_structure}
    structure_output: widgets.Output | None = None
    structure_viewer: widgets.Widget | None = None
    if isinstance(initial_structure, MolstarViewSpec):
        structure_viewer = _molstar_viewer(initial_structure)
        structure_panel: widgets.Widget = structure_viewer
    else:
        structure_output = widgets.Output(layout=widgets.Layout(width="54%"))
        with structure_output:
            display(initial_structure)
        structure_panel = structure_output
    initial_plot = _build_metric_scatter(
        browser_df,
        label_column=label_column,
        x_column=default_x,
        y_column=default_y,
        z_column=default_z,
        selected_index=0,
        title=title,
        metric_labels=metric_labels,
        plot_height=plot_height,
    )
    figure_widget, plot_panel = _make_figure_widget(initial_plot, pixel_width=plot_width)

    def resolve_structure(index: int) -> Any:
        if index not in structure_cache:
            structure_cache[index] = structure_factory(records[index])
        return structure_cache[index]

    def render(*_args: Any) -> None:
        index = current_index["value"]
        row = browser_df.iloc[index]
        x_column = x_dropdown.value
        y_column = y_dropdown.value
        z_column = z_dropdown.value
        x_label = _metric_label(x_column, metric_labels)
        y_label = _metric_label(y_column, metric_labels)
        z_label = _metric_label(z_column, metric_labels) if z_column else None

        status_parts = [
            f"<b>{index + 1}/{len(records)}</b>",
            f"<code>{row[label_column]}</code>",
            f"{x_label}={_format_cell_value(row[x_column])}",
            f"{y_label}={_format_cell_value(row[y_column])}",
        ]
        if z_column is not None:
            status_parts.append(f"{z_label}={_format_cell_value(row[z_column])}")
        status_html.value = " &nbsp; ".join(status_parts)
        prev_button.disabled = index == 0
        next_button.disabled = index == len(records) - 1

        structure = resolve_structure(index)
        if structure_viewer is not None and isinstance(structure, MolstarViewSpec):
            _update_molstar_viewer(structure_viewer, structure)
        elif structure_output is not None:
            with structure_output:
                clear_output(wait=True)
                display(structure)

        # Update the FigureWidget in place so the plot doesn't flash or lose zoom.
        selected_row = browser_df.iloc[index]
        sel_visible = bool(pd.notna(selected_row[x_column]) and pd.notna(selected_row[y_column]))
        hover_lines = [
            "%{text}<br>",
            f"{x_label}: %{{x:.3f}}<br>",
            f"{y_label}: %{{y:.3f}}",
        ]
        if z_column:
            hover_lines.append(f"<br>{z_label}: %{{customdata[0]:.3f}}")
        hover_lines.append("<extra></extra>")

        with figure_widget.batch_update():
            figure_widget.data[0].x = browser_df[x_column].tolist()
            figure_widget.data[0].y = browser_df[y_column].tolist()
            figure_widget.data[0].text = browser_df[label_column].tolist()
            figure_widget.data[0].hovertemplate = "".join(hover_lines)

            if z_column:
                figure_widget.data[0].customdata = browser_df[[z_column]].to_numpy()
                figure_widget.data[0].marker = go.scatter.Marker(
                    size=10,
                    color=browser_df[z_column].tolist(),
                    coloraxis="coloraxis",
                    line=dict(width=1, color="white"),
                )
                figure_widget.layout.coloraxis = go.layout.Coloraxis(
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=go.layout.coloraxis.ColorBar(
                        title=dict(text=z_label, side="bottom"),
                        thickness=14,
                        len=0.9,
                        x=0,
                        xanchor="right",
                        xpad=5,
                    ),
                )
                figure_widget.layout.margin = dict(l=90, r=28, t=45, b=30)
            else:
                figure_widget.data[0].customdata = None
                figure_widget.data[0].marker = go.scatter.Marker(
                    size=10,
                    color="#60a5fa",
                    line=dict(width=1, color="white"),
                )
                figure_widget.layout.coloraxis = go.layout.Coloraxis(showscale=False)
                figure_widget.layout.margin = dict(l=0, r=28, t=45, b=0)

            figure_widget.data[1].x = [selected_row[x_column]] if sel_visible else [None]
            figure_widget.data[1].y = [selected_row[y_column]] if sel_visible else [None]
            figure_widget.data[1].visible = sel_visible

            figure_widget.layout.xaxis.title.text = x_label
            figure_widget.layout.yaxis.title.text = y_label
            figure_widget.layout.autosize = False
            figure_widget.layout.width = plot_width

    def _on_scatter_click(trace: Any, points: Any, selector: Any) -> None:
        if points.point_inds:
            current_index["value"] = points.point_inds[0]
            render()

    figure_widget.data[0].on_click(_on_scatter_click)

    def step(delta: int) -> None:
        current_index["value"] = min(
            max(current_index["value"] + delta, 0),
            len(records) - 1,
        )
        render()

    prev_button.on_click(lambda _: step(-1))
    next_button.on_click(lambda _: step(1))
    x_dropdown.observe(render, names="value")
    y_dropdown.observe(render, names="value")
    z_dropdown.observe(render, names="value")

    children: list[Any] = [
        widgets.HBox([prev_button, next_button, status_html]),
        widgets.HBox([x_dropdown, y_dropdown, z_dropdown]),
    ]
    if note_html:
        children.append(widgets.HTML(note_html))
    children.append(widgets.HBox(
        [structure_panel, plot_panel],
        layout=widgets.Layout(justify_content="space-between"),
    ))

    browser = widgets.VBox(children)
    render()
    return browser


def make_structure_browser(
    *,
    records: Sequence[dict[str, Any]],
    metrics_df: pd.DataFrame,
    structure_factory: Callable[[dict[str, Any]], Any],
    label_column: str,
    title: str | None = None,
    metric_columns: Sequence[str] | None = None,
    info_columns: Sequence[str] | None = None,
    metric_labels: dict[str, str] | None = None,
    note_html: str | None = None,
) -> widgets.Widget:
    del title, metric_columns
    browser_df = metrics_df.reset_index(drop=True).copy()
    if len(records) != len(browser_df):
        raise ValueError("records and metrics_df must have the same number of rows")

    resolved_info_columns = list(
        info_columns
        or [
            column
            for column in browser_df.columns
            if column != label_column
        ][:4]
    )

    current_index = {"value": 0}
    prev_button = widgets.Button(description="Previous")
    next_button = widgets.Button(description="Next")
    status_html = widgets.HTML()
    initial_structure = structure_factory(records[0])
    structure_cache: dict[int, Any] = {0: initial_structure}
    structure_output: widgets.Output | None = None
    structure_viewer: widgets.Widget | None = None
    if isinstance(initial_structure, MolstarViewSpec):
        structure_viewer = _molstar_viewer(initial_structure)
        structure_panel: widgets.Widget = widgets.Box([structure_viewer])
    else:
        structure_output = widgets.Output()
        with structure_output:
            display(initial_structure)
        structure_panel = structure_output

    def resolve_structure(index: int) -> Any:
        if index not in structure_cache:
            structure_cache[index] = structure_factory(records[index])
        return structure_cache[index]

    def render(*_args: Any) -> None:
        index = current_index["value"]
        row = browser_df.iloc[index]
        status_parts = [
            f"<b>{index + 1}/{len(records)}</b>",
            f"<code>{row[label_column]}</code>",
        ]
        for column in resolved_info_columns:
            status_parts.append(
                f"{_metric_label(column, metric_labels)}={_format_cell_value(row[column])}"
            )
        status_html.value = " &nbsp; ".join(status_parts)
        prev_button.disabled = index == 0
        next_button.disabled = index == len(records) - 1

        structure = resolve_structure(index)
        if structure_viewer is not None and isinstance(structure, MolstarViewSpec):
            _update_molstar_viewer(structure_viewer, structure)
        elif structure_output is not None:
            with structure_output:
                clear_output(wait=True)
                display(structure)

    def step(delta: int) -> None:
        current_index["value"] = min(
            max(current_index["value"] + delta, 0),
            len(records) - 1,
        )
        render()

    prev_button.on_click(lambda _: step(-1))
    next_button.on_click(lambda _: step(1))

    children: list[Any] = [widgets.HBox([prev_button, next_button, status_html])]
    if note_html:
        children.append(widgets.HTML(note_html))
    children.append(structure_panel)

    browser = widgets.VBox(children)
    render()
    return browser


__all__ = [
    "HBOND_CUTOFF_ANGLE",
    "HBOND_CUTOFF_DIST",
    "HBOND_PH",
    "HBOND_REQUIRE_CROSS_RESIDUE",
    "PHOSPHATE_ATOMS",
    "PTR_SIDECHAIN_ATOMS",
    "SASA_POINT_DISTR",
    "SASA_POINT_NUMBER",
    "SASA_PROBE_RADIUS",
    "SASA_VDW_RADII",
    "align_mobile_on_binder_backbone",
    "atom_triplet_label",
    "binder_backbone_rmsd",
    "build_rf3_metrics_df",
    "build_rfd3_metrics_df",
    "compute_phosphosite_hbond_metrics",
    "compute_selection_sasa_breakdown",
    "compute_selection_sasa_metrics",
    "extract_min_interface_pae",
    "format_hbond_connection",
    "make_browser_structure_spec",
    "make_browser_structure_view",
    "make_hbond_view",
    "make_sasa_view",
    "make_structure_browser",
    "make_studio_metric_browser",
    "make_structure_overlay_spec",
    "make_structure_overlay_view",
    "paired_backbone_indices",
    "paired_common_indices",
    "plot_hbond_contact_map",
    "prepare_atom_array_for_hbonds",
    "project_points_to_plane",
    "rmsd_for_masks",
    "same_residue",
]
