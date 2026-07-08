import math

from lattice.core.vectors import cosine


def test_cosine_identical_vectors():
    assert math.isclose(cosine((1.0, 2.0), (1.0, 2.0)), 1.0)


def test_cosine_orthogonal_vectors():
    assert cosine((1.0, 0.0), (0.0, 1.0)) == 0.0


def test_cosine_zero_vector_is_zero():
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0
