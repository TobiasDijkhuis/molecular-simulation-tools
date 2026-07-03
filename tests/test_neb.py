import pytest
from ase import Atoms
import numpy as np

from ase.mep.neb import interpolate
from molecular_simulation_tools.neb import get_images_for_neb, idpp_interpolate_subset


def test_get_images_for_neb_with_too_few_images_raises():
    initial_atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
    final_atoms = Atoms(symbols="H", positions=[[0, 0, 1]])
    n_images = 2
    with pytest.raises(ValueError):
        get_images_for_neb(initial_atoms, final_atoms, n_images)


def test_get_images_for_neb():
    initial_atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
    final_atoms = Atoms(symbols="H", positions=[[0, 0, 1]])
    n_images = 5
    neb_images = get_images_for_neb(initial_atoms, final_atoms, n_images)

    assert len(neb_images) == n_images
    assert all(neb_image == initial_atoms for neb_image in neb_images[:-1])
    assert neb_images[-1] == final_atoms

    assert id(neb_images[0]) != id(initial_atoms)
    assert id(neb_images[-1]) != id(final_atoms)


idpp_interpolate_subset_data = [
    (
        Atoms("OH2", positions=np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]])),
        Atoms("OH2", positions=np.array([[0, 0, 1], [0, 0, 0], [0, 0, 2]])),
        5,
        [0, 1],
    ),
    (
        Atoms("OH2", positions=np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]])),
        Atoms("OH2", positions=np.array([[0, 0, 1], [0, 0, 0], [0, 0, 2]])),
        5,
        [0, 1, 2],
    ),
]


@pytest.mark.parametrize(
    "initial_image, final_image, n_images, indices",
    idpp_interpolate_subset_data,
)
def test_idpp_interpolate_subset(
    initial_image,
    final_image,
    n_images,
    indices,
):
    images = get_images_for_neb(initial_image, final_image, n_images)

    interpolated_images = idpp_interpolate_subset(
        images, indices, kwargs={"traj": None, "log": None}
    )

    linear_indices = [
        idx for idx in np.arange(len(initial_image)) if idx not in indices
    ]

    linear_interpolated_images = [image.copy() for image in images]
    interpolate(linear_interpolated_images)
    for image_idx, (idpp_image, linear_image) in enumerate(
        zip(interpolated_images, linear_interpolated_images, strict=True)
    ):
        assert np.allclose(
            idpp_image.positions[linear_indices, :],
            linear_image.positions[linear_indices, :],
        )
        if image_idx > 0 and image_idx < n_images - 1:
            assert not np.allclose(
                idpp_image.positions[indices, :], linear_image.positions[indices, :]
            )
