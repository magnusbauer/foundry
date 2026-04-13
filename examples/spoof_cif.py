from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import biotite.structure as struc
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from atomworks.constants import DICT_THREE_TO_ONE, UNKNOWN_AA
from atomworks.io.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import get_residue_starts

DEFAULT_DISPLAY_MAP = {
    **DICT_THREE_TO_ONE,
    "PTR": "pY",
    "SEP": "pS",
    "TPO": "pT",
    "M3L": "K(me3)",
}
DEFAULT_UNKNOWN = DICT_THREE_TO_ONE[UNKNOWN_AA]


def tokenize_polymer_sequence(sequence: str) -> list[str]:
    """Split a mixed canonical / parenthesized sequence into residue tokens."""
    tokens: list[str] = []
    idx = 0
    while idx < len(sequence):
        char = sequence[idx]
        if char.isspace():
            idx += 1
            continue
        if char == "(":
            end = sequence.find(")", idx)
            if end == -1:
                raise ValueError(f"Unclosed PTM token in sequence: {sequence!r}")
            token = sequence[idx + 1 : end].strip().upper()
            if not token:
                raise ValueError(f"Empty PTM token in sequence: {sequence!r}")
            tokens.append(token)
            idx = end + 1
            continue
        if not char.isalpha():
            raise ValueError(
                f"Unsupported character {char!r} in sequence {sequence!r}. "
                "Use one-letter amino acids and PTMs in parentheses, e.g. (PTR)."
            )
        tokens.append(char.upper())
        idx += 1
    if not tokens:
        raise ValueError("Sequence is empty.")
    return tokens


def find_ptm_positions(tokens: list[str], ptm_resnames: set[str] | None = None) -> dict[int, str]:
    """Return 1-indexed PTM positions from a tokenized sequence."""
    if ptm_resnames is None:
        return {idx + 1: token for idx, token in enumerate(tokens) if len(token) > 1}
    allowed = {name.upper() for name in ptm_resnames}
    return {idx + 1: token for idx, token in enumerate(tokens) if token.upper() in allowed}


def sequence_length(sequence: str) -> int:
    """Count residues while treating each parenthesized PTM as one residue."""
    return len(tokenize_polymer_sequence(sequence))


def _spoof_cif_from_dictionary(item: dict[str, Any], out_dir: Path) -> tuple[Path, Any]:
    """Build a CIF file from a cifutils component dictionary."""
    if "name" not in item or "components" not in item:
        raise ValueError("Input dictionary must contain 'name' and 'components'.")

    atom_array, component_list = components_to_atom_array(
        item["components"],
        return_components=True,
        bonds=item.get("bonds"),
    )
    msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(
        component_list
    )

    cif_path = out_dir / f"{item['name']}.cif"
    save_path = Path(
        to_cif_file(
            atom_array,
            cif_path,
            extra_categories={"msa_paths_by_chain_id": msa_paths_by_chain_id}
            if msa_paths_by_chain_id
            else None,
            file_type="cif",
        )
    )
    return save_path, atom_array


def clean_cif_file(cif_path: str | Path) -> Path:
    """Replace textual NaNs so downstream parsers do not choke on the spoofed CIF."""
    cif_path = Path(cif_path)
    cif_path.write_text(cif_path.read_text().replace("nan", "0"))
    return cif_path


def default_ptr_binder_conditioning(
    chain_id: str,
    residue_id: int,
) -> dict[str, dict[str, str] | str | bool]:
    """Return a workshop-friendly RFD3 conditioning block for phosphotyrosine."""
    residue_key = f"{chain_id}{residue_id}"
    return {
        "dialect": 2,
        "infer_ori_strategy": "hotspots",
        "redesign_motif_sidechains": False,
        "select_fixed_atoms": False,
        "select_hotspots": {
            residue_key: "P,O1P,O2P,O3P,OH",
        },
        "select_buried": {
            residue_key: "P,O1P,O2P,O3P",
        },
        "select_hbond_acceptor": {
            residue_key: "O1P,O2P,O3P",
        },
    }


def build_rfd3_input(
    *,
    name: str,
    cif_path: str | Path,
    binder_length: int,
    target_length: int,
    target_chain_id: str,
    ptm_residue_id: int,
) -> dict[str, dict[str, Any]]:
    """Create a single-example RFD3 JSON payload for peptide binder design."""
    cif_path = Path(cif_path).resolve()
    spec: dict[str, Any] = {
        "input": str(cif_path),
        "contig": f"{binder_length}-{binder_length},/0,{target_chain_id}1-{target_length}",
        "length": f"{binder_length + target_length}-{binder_length + target_length}",
    }
    spec.update(
        default_ptr_binder_conditioning(
            chain_id=target_chain_id,
            residue_id=ptm_residue_id,
        )
    )
    return {name: spec}


def spoof_cif_from_sequence(
    *,
    name: str,
    sequence: str,
    out_dir: str | Path,
    chain_id: str = "B",
) -> tuple[Path, Any]:
    """Write a spoofed CIF for a polymer sequence that may include PTMs."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cif_path, atom_array = _spoof_cif_from_dictionary(
        item={
            "name": name,
            "components": [{"seq": sequence, "chain_id": chain_id}],
        },
        out_dir=out_dir,
    )
    clean_cif_file(cif_path)
    return cif_path, atom_array


def build_ptm_binder_workshop_inputs(
    *,
    name: str,
    sequence: str,
    binder_length: int,
    out_dir: str | Path,
    target_chain_id: str = "B",
    ptm_resname: str = "PTR",
) -> dict[str, Any]:
    """Create the spoofed CIF plus an RFD3 JSON input file for the workshop."""
    tokens = tokenize_polymer_sequence(sequence)
    ptm_positions = find_ptm_positions(tokens, {ptm_resname})
    if not ptm_positions:
        raise ValueError(f"Sequence {sequence!r} does not contain {ptm_resname}.")

    ptm_residue_id = next(iter(ptm_positions))
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_path, atom_array = spoof_cif_from_sequence(
        name=name,
        sequence=sequence,
        out_dir=out_dir,
        chain_id=target_chain_id,
    )
    rfd3_payload = build_rfd3_input(
        name=name,
        cif_path=cif_path,
        binder_length=binder_length,
        target_length=len(tokens),
        target_chain_id=target_chain_id,
        ptm_residue_id=ptm_residue_id,
    )
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(rfd3_payload, indent=2))

    return {
        "name": name,
        "sequence": sequence,
        "tokens": tokens,
        "sequence_length": len(tokens),
        "ptm_positions": ptm_positions,
        "ptm_residue_id": ptm_residue_id,
        "cif_path": cif_path,
        "json_path": json_path,
        "atom_array": atom_array,
        "rfd3_input": rfd3_payload,
    }


def residue_label(atom_array, atom_index: int) -> str:
    """Format a chain / residue label for one atom."""
    return (
        f"{atom_array.chain_id[atom_index]}{int(atom_array.res_id[atom_index])}:"
        f"{atom_array.res_name[atom_index]}"
    )


def atom_label(atom_array, atom_index: int) -> str:
    """Format a chain / residue / atom label for one atom."""
    return f"{residue_label(atom_array, atom_index)}:{atom_array.atom_name[atom_index]}"


def bond_table_for_residue(
    atom_array,
    *,
    chain_id: str,
    residue_id: int,
    include_neighbors: bool = True,
) -> pd.DataFrame:
    """Summarize bonds connected to one residue."""
    if atom_array.bonds is None:
        raise ValueError("AtomArray does not contain bond annotations.")

    rows: list[dict[str, Any]] = []
    focus_mask = (atom_array.chain_id == chain_id) & (atom_array.res_id == residue_id)
    if not focus_mask.any():
        raise ValueError(f"Residue {chain_id}{residue_id} not found.")

    for atom_i, atom_j, bond_type in atom_array.bonds.as_array():
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        in_focus_i = bool(focus_mask[atom_i])
        in_focus_j = bool(focus_mask[atom_j])
        if include_neighbors:
            keep = in_focus_i or in_focus_j
        else:
            keep = in_focus_i and in_focus_j
        if not keep:
            continue
        rows.append(
            {
                "atom_1": atom_label(atom_array, atom_i),
                "atom_2": atom_label(atom_array, atom_j),
                "residue_1": residue_label(atom_array, atom_i),
                "residue_2": residue_label(atom_array, atom_j),
                "bond_type": int(bond_type),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(
        by=["residue_1", "atom_1", "residue_2", "atom_2"],
        ignore_index=True,
    )


def plot_residue_bond_graph(
    atom_array,
    *,
    chain_id: str,
    ax=None,
):
    """Plot a residue-level bond graph for one chain."""
    if atom_array.bonds is None:
        raise ValueError("AtomArray does not contain bond annotations.")

    token_starts = get_residue_starts(atom_array)
    nodes: list[tuple[str, int, str]] = []
    for start in token_starts:
        if atom_array.chain_id[start] != chain_id:
            continue
        nodes.append(
            (
                str(atom_array.chain_id[start]),
                int(atom_array.res_id[start]),
                str(atom_array.res_name[start]),
            )
        )

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node)

    for atom_i, atom_j, _ in atom_array.bonds.as_array():
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if atom_array.chain_id[atom_i] != chain_id or atom_array.chain_id[atom_j] != chain_id:
            continue
        node_i = (
            str(atom_array.chain_id[atom_i]),
            int(atom_array.res_id[atom_i]),
            str(atom_array.res_name[atom_i]),
        )
        node_j = (
            str(atom_array.chain_id[atom_j]),
            int(atom_array.res_id[atom_j]),
            str(atom_array.res_name[atom_j]),
        )
        if node_i != node_j:
            graph.add_edge(node_i, node_j)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, len(nodes) * 0.8), 2.5))

    pos = {node: (node[1], 0.0) for node in nodes}
    labels = {node: f"{node[1]}:{node[2]}" for node in nodes}
    colors = ["#d97706" if node[2] == "PTR" else "#2563eb" for node in nodes]

    nx.draw_networkx(
        graph,
        pos=pos,
        labels=labels,
        node_color=colors,
        node_size=1400,
        font_size=9,
        ax=ax,
    )
    ax.set_title(f"Residue bond graph for chain {chain_id}")
    ax.set_axis_off()
    return ax


def plot_local_atom_bond_graph(
    atom_array,
    *,
    chain_id: str,
    residue_id: int,
    ax=None,
):
    """Plot the atom-level bond graph for a PTM residue and its peptide neighbors."""
    if atom_array.bonds is None:
        raise ValueError("AtomArray does not contain bond annotations.")

    local_table = bond_table_for_residue(
        atom_array,
        chain_id=chain_id,
        residue_id=residue_id,
        include_neighbors=True,
    )
    if local_table.empty:
        raise ValueError(f"No bonds found for {chain_id}{residue_id}.")

    graph = nx.Graph()
    focus_key = f"{chain_id}{residue_id}:"

    def node_color(node_name: str) -> str:
        if node_name.startswith(focus_key):
            return "#d97706"
        if node_name.startswith(f"{chain_id}{residue_id - 1}:") or node_name.startswith(
            f"{chain_id}{residue_id + 1}:"
        ):
            return "#93c5fd"
        return "#cbd5e1"

    for row in local_table.itertuples(index=False):
        graph.add_edge(row.atom_1, row.atom_2, bond_type=row.bond_type)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    pos = nx.spring_layout(graph, seed=7, k=0.9)
    colors = [node_color(node) for node in graph.nodes]
    nx.draw_networkx(
        graph,
        pos=pos,
        node_color=colors,
        node_size=1400,
        font_size=8,
        ax=ax,
    )
    ax.set_title(f"Local atom bond graph around {chain_id}{residue_id}")
    ax.set_axis_off()
    return ax


def extract_chain_sequence(
    atom_array,
    chain_id: str,
    *,
    residue_map: dict[str, str] | None = None,
) -> str:
    """Return a compact sequence string for one chain."""
    residue_map = residue_map or DEFAULT_DISPLAY_MAP
    res_starts = get_residue_starts(atom_array)
    tokens: list[str] = []
    for start in res_starts:
        if atom_array.chain_id[start] != chain_id:
            continue
        res_name = str(atom_array.res_name[start]).upper()
        tokens.append(residue_map.get(res_name, DEFAULT_UNKNOWN))
    return "".join(tokens)


def chain_summary(atom_array) -> pd.DataFrame:
    """Build a quick chain-level summary table for display in notebooks."""
    rows: list[dict[str, Any]] = []
    token_starts = get_residue_starts(atom_array)
    for chain_id in pd.unique(atom_array.chain_id):
        chain_mask = atom_array.chain_id == chain_id
        token_count = sum(atom_array.chain_id[start] == chain_id for start in token_starts)
        rows.append(
            {
                "chain_id": str(chain_id),
                "n_atoms": int(chain_mask.sum()),
                "n_residues": int(token_count),
                "sequence": extract_chain_sequence(atom_array, str(chain_id)),
            }
        )
    return pd.DataFrame(rows).sort_values("chain_id", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spoof a CIF and matching RFD3 JSON input for a PTM peptide target."
    )
    parser.add_argument("--name", default="ptm_target")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--binder-length", type=int, default=100)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--target-chain-id", default="B")
    args = parser.parse_args()

    result = build_ptm_binder_workshop_inputs(
        name=args.name,
        sequence=args.sequence,
        binder_length=args.binder_length,
        out_dir=args.out_dir,
        target_chain_id=args.target_chain_id,
    )

    summary = {
        "name": result["name"],
        "sequence": result["sequence"],
        "tokens": result["tokens"],
        "ptm_positions": result["ptm_positions"],
        "sequence_length": result["sequence_length"],
        "cif_path": str(result["cif_path"]),
        "json_path": str(result["json_path"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
