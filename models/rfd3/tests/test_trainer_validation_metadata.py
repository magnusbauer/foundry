import numpy as np
import pytest
import torch
from atomworks.io.tools.inference import components_to_atom_array
from biotite.structure import concatenate
from biotite.structure.residues import get_residue_starts

from rfd3.constants import VIRTUAL_ATOM_ELEMENT_NAME, association_schemes
from rfd3.inference.input_parsing import create_atom_array_from_design_specification
from rfd3.trainer import rfd3 as rfd3_module
from rfd3.trainer.trainer_utils import _cleanup_virtual_atoms_and_assign_atom_name_elements
from rfd3.trainer.rfd3 import AADesignTrainer


def _create_ptm_atom_array():
    components = [
        {
            "seq": "AG(PTR)(SEP)SA",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "A",
        },
    ]
    atom_array = components_to_atom_array(components)
    atom_array.coord = (
        np.arange(len(atom_array) * 3, dtype=np.float32).reshape(len(atom_array), 3)
        * 0.1
    )
    return atom_array


def _build_example(*, include_specification: bool, include_src_component: bool = True):
    atom_array_input = _create_ptm_atom_array()
    atom_array, _ = create_atom_array_from_design_specification(
        atom_array_input=atom_array_input,
        input=None,
        contig="5-5,A3-4,5-5",
        length="12-12",
        dialect=2,
    )
    atom_array = atom_array.copy()
    if "token_id" not in atom_array.get_annotation_categories():
        residue_starts = get_residue_starts(atom_array)
        token_ids = np.zeros(atom_array.array_length(), dtype=int)
        for token_id, start in enumerate(residue_starts):
            end = (
                residue_starts[token_id + 1]
                if token_id + 1 < len(residue_starts)
                else atom_array.array_length()
            )
            token_ids[start:end] = token_id
        atom_array.set_annotation("token_id", token_ids)
    if "is_protein" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "is_protein",
            np.ones(atom_array.array_length(), dtype=bool),
        )
    if "is_ligand" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "is_ligand",
            np.zeros(atom_array.array_length(), dtype=bool),
        )
    if include_src_component and "src_component" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "src_component",
            np.array([""] * atom_array.array_length(), dtype=object),
        )
    elif not include_src_component and "src_component" in atom_array.get_annotation_categories():
        atom_array.del_annotation("src_component")
    if "gt_atom_name" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("gt_atom_name", atom_array.atom_name.copy())
    for annotation in ["active_donor", "active_acceptor"]:
        if annotation in atom_array.get_annotation_categories():
            atom_array.del_annotation(annotation)

    example = {
        "atom_array": atom_array,
        "feats": {},
        "example_id": "trainer-validation-metadata",
    }
    if include_specification:
        example["specification"] = {"example": "smoke-inference", "dialect": 2}
    return example


def _build_network_output(atom_array):
    n_tokens = int(np.max(atom_array.token_id)) + 1
    return {
        "X_L": atom_array.coord[None].astype(np.float32, copy=True),
        "sequence_indices_I": torch.zeros((1, n_tokens), dtype=torch.long),
        "sequence_logits_I": torch.zeros((1, n_tokens, 32), dtype=torch.float32),
    }


def _make_test_trainer():
    trainer = object.__new__(AADesignTrainer)
    trainer.allow_sequence_outputs = True
    trainer.cleanup_guideposts = False
    trainer.cleanup_virtual_atoms = False
    trainer.read_sequence_from_sequence_head = False
    trainer.output_full_json = True
    trainer.compute_non_clash_metrics_for_diffused_region_only = False
    trainer.association_scheme = "atom14"
    trainer.seed = None
    return trainer


@pytest.mark.fast
def test_build_predicted_atom_array_stack_without_specification():
    trainer = _make_test_trainer()
    example = _build_example(include_specification=False)

    predicted_atom_array_stack, prediction_metadata = (
        trainer._build_predicted_atom_array_stack(
            _build_network_output(example["atom_array"]),
            example,
        )
    )

    assert len(predicted_atom_array_stack) == 1
    assert 0 in prediction_metadata
    assert "specification" not in prediction_metadata[0]
    assert "task" not in prediction_metadata[0]


@pytest.mark.fast
def test_build_predicted_atom_array_stack_with_specification():
    trainer = _make_test_trainer()
    example = _build_example(include_specification=True)

    predicted_atom_array_stack, prediction_metadata = (
        trainer._build_predicted_atom_array_stack(
            _build_network_output(example["atom_array"]),
            example,
        )
    )

    assert len(predicted_atom_array_stack) == 1
    assert prediction_metadata[0]["task"] == "smoke-inference"
    assert prediction_metadata[0]["specification"]["dialect"] == 2


@pytest.mark.fast
def test_build_predicted_atom_array_stack_without_src_component():
    trainer = _make_test_trainer()
    example = _build_example(
        include_specification=False,
        include_src_component=False,
    )

    predicted_atom_array_stack, prediction_metadata = (
        trainer._build_predicted_atom_array_stack(
            _build_network_output(example["atom_array"]),
            example,
        )
    )

    assert len(predicted_atom_array_stack) == 1
    assert prediction_metadata[0]["diffused_index_map"] == {}


@pytest.mark.fast
def test_cleanup_virtual_atoms_handles_reused_res_ids_across_chains():
    atom_array = components_to_atom_array(
        [
            {
                "seq": "Y",
                "chain_type": "polypeptide(l)",
                "is_polymer": True,
                "chain_id": "A",
            },
            {
                "seq": "A",
                "chain_type": "polypeptide(l)",
                "is_polymer": True,
                "chain_id": "B",
            },
        ]
    )
    atom_array.coord = (
        np.arange(len(atom_array) * 3, dtype=np.float32).reshape(len(atom_array), 3)
        * 0.1
    )
    atom_array.set_annotation("gt_atom_name", atom_array.atom_name.copy())
    atom_array.set_annotation(
        "is_motif_atom_unindexed",
        np.zeros(atom_array.array_length(), dtype=bool),
    )

    # Make chain A an unknown-sequence residue and pad it to the dense scheme length
    # while keeping chain B fixed-sequence. This reproduces the multichain cleanup path
    # that previously merged A1 and B1 when only res_id boundaries were used.
    is_fixed_seq = atom_array.chain_id == "B"
    atom_array.set_annotation("is_motif_atom_with_fixed_seq", is_fixed_seq)

    chain_a = atom_array[atom_array.chain_id == "A"]
    chain_b = atom_array[atom_array.chain_id == "B"]

    n_pad = len(association_schemes["dense"]["TYR"]) - len(chain_a)
    assert n_pad > 0
    pad_atoms = chain_a[:n_pad].copy()
    pad_atoms.atom_name = np.array(["VX"] * n_pad, dtype=pad_atoms.atom_name.dtype)
    pad_atoms.gt_atom_name = pad_atoms.atom_name.copy()
    pad_atoms.element = np.array(
        [VIRTUAL_ATOM_ELEMENT_NAME] * n_pad,
        dtype=pad_atoms.element.dtype,
    )

    padded_atom_array = concatenate([chain_a, pad_atoms, chain_b])

    cleaned = _cleanup_virtual_atoms_and_assign_atom_name_elements(
        padded_atom_array,
        association_scheme="dense",
    )

    assert np.all(cleaned.element != VIRTUAL_ATOM_ELEMENT_NAME)
    chain_switch = np.where(cleaned.chain_id[1:] != cleaned.chain_id[:-1])[0]
    assert len(chain_switch) == 1
    assert cleaned.chain_id[0] == "A"
    assert cleaned.chain_id[-1] == "B"


@pytest.mark.fast
def test_build_predicted_atom_array_stack_tolerates_cleanup_failure(monkeypatch):
    trainer = _make_test_trainer()
    trainer.cleanup_virtual_atoms = True
    trainer.association_scheme = "dense"
    example = _build_example(include_specification=False)

    def _raise_cleanup_error(atom_array, association_scheme="atom14"):
        raise ValueError("cleanup failed")

    monkeypatch.setattr(
        rfd3_module,
        "_cleanup_virtual_atoms_and_assign_atom_name_elements",
        _raise_cleanup_error,
    )

    predicted_atom_array_stack, prediction_metadata = (
        trainer._build_predicted_atom_array_stack(
            _build_network_output(example["atom_array"]),
            example,
        )
    )

    assert len(predicted_atom_array_stack) == 1
    assert prediction_metadata[0]["metrics"]["cleanup_virtual_atoms_failed"] == 1.0
