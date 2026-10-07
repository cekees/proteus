"""TC_base: per-object state."""
from proteus.TransportCoefficients import TC_base


def test_coefficients_objects_do_not_share_their_sparse_diffusion_info():
    # Transport writes a default layout into coefficients.sdInfo for each
    # diffusion term it lacks, sized by the mesh's dimension. Shared through a
    # mutable default argument, a 1D model's layout reached a later 2D model.
    a = TC_base(nc=1, diffusion={0: {0: {0: 'constant'}}}, potential={0: {0: 'u'}})
    b = TC_base(nc=1, diffusion={0: {0: {0: 'constant'}}}, potential={0: {0: 'u'}})
    a.sdInfo[(0, 0)] = "1D layout"
    assert b.sdInfo == {}
