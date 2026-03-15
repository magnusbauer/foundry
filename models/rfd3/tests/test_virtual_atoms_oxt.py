import numpy as np
from biotite import structure as struc

from rfd3.constants import ATOM14_ATOM_NAMES
from rfd3.transforms.virtual_atoms import PadTokensWithVirtualAtoms


def test_pad_tokens_with_virtual_atoms_removes_residual_oxt():
    atom_array = struc.info.residue("ILE")
    atom_array = atom_array[atom_array.element != "H"]
    atom_array.coord = np.zeros((atom_array.array_length(), 3), dtype=np.float32)
    atom_array.occupancy = np.ones(atom_array.array_length(), dtype=np.float32)
    atom_array.set_annotation(
        "token_id", np.zeros(atom_array.array_length(), dtype=np.int32)
    )
    atom_array.set_annotation(
        "is_protein", np.ones(atom_array.array_length(), dtype=bool)
    )
    atom_array.set_annotation(
        "atomize", np.zeros(atom_array.array_length(), dtype=bool)
    )
    atom_array.set_annotation(
        "is_motif_atom_with_fixed_seq",
        np.ones(atom_array.array_length(), dtype=bool),
    )
    atom_array.set_annotation(
        "is_motif_atom_unindexed", np.zeros(atom_array.array_length(), dtype=bool)
    )

    transform = PadTokensWithVirtualAtoms(
        n_atoms_per_token=14,
        atom_to_pad_from="CA",
        association_scheme="atom14",
    )

    output = transform.forward(
        {
            "atom_array": atom_array,
            "is_inference": False,
            "example_id": "oxt-regression",
        }
    )
    transformed = output["atom_array"]

    assert "OXT" not in transformed.atom_name
    assert transformed.array_length() == atom_array.array_length() - 1
    assert np.all(np.isin(transformed.atom_name, ATOM14_ATOM_NAMES))
