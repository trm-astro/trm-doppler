"""
Python glue connecting the Rust-based ``trm.doppler._doppler_rs`` extension
(built from the ``rust/`` workspace via PyO3/maturin; see ``rust/README.md``)
to this package's pure-Python :class:`~trm.doppler.map.Map` and
:class:`~trm.doppler.data.Data` classes.

This provides ``comdat``, ``comdef``, ``datcom`` and ``memit``, matching the
names and calling conventions of the equivalent functions in the retired
C++ ``doppler.cc`` extension, but implemented with plain Python attribute
access instead of the Python C API. Unlike the old ``read_map``/
``read_data``/``update_map``/``update_data`` C++ helpers, the
flattening/unflattening of ``Map``/``Data`` objects into the flat arrays
expected by ``_doppler_rs`` is done here in Python, where it is far less
code and much easier to maintain.

``trm.doppler._doppler_rs`` is built as part of this package itself (see the
root ``pyproject.toml``'s ``[tool.maturin]`` section, which points at
``rust/doppler-py``); a normal ``pip install .`` / ``pip install -e .`` from
the repository root builds it automatically.
"""

import numpy as np

from .core import DopplerError

try:
    from . import _doppler_rs
except ImportError as exc:  # pragma: no cover - exercised only when unbuilt
    _doppler_rs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

__all__ = ['comdat', 'comdef', 'datcom', 'memit']


def _require_rust():
    if _doppler_rs is None:
        raise DopplerError(
            "trm.doppler._rust: the compiled Rust extension "
            "'trm.doppler._doppler_rs' could not be imported; (re)install "
            "this package with 'pip install -e .' from the repository "
            f"root to build it. Original import error: {_IMPORT_ERROR}"
        )



def _image_nxyz(image):
    """
    Returns (nx, ny, nz) for an Image, matching the numpy array-shape
    convention used throughout this package: 2D image arrays have shape
    (ny, nx); 3D image arrays have shape (nz, ny, nx).
    """
    shape = image.data.shape
    if len(shape) == 2:
        ny, nx = shape
        nz = 1
    else:
        nz, ny, nx = shape
    return nx, ny, nz


def _flatten_map(dmap):
    """
    Flattens a Map's Images into the flat arguments expected by the
    op/tr/default_image/memit Rust bindings, returning a dict that also
    carries enough information (``shapes``) to unflatten a result back
    into the Map's Images afterwards.
    """
    images = []
    nxyz, vxy, vz = [], [], []
    wavel, gamma, scale, itype = [], [], [], []
    def_option, def_bias = [], []
    def_fwhmxy, def_fwhmz, def_squeeze, def_sqfwhm = [], [], [], []
    shapes = []

    for image in dmap.data:
        nx, ny, nz = _image_nxyz(image)
        shapes.append(image.data.shape)
        images.append(np.ascontiguousarray(image.data, dtype=np.float32).ravel())

        nxyz.append((nx, ny, nz))
        vxy.append(float(image.vxy))
        vz.append(float(image.vz) if image.vz is not None else 0.0)
        wavel.append([float(w) for w in np.asarray(image.wave, dtype=np.float64)])
        gamma.append([float(g) for g in np.asarray(image.gamma, dtype=np.float32)])
        scale.append([float(s) for s in np.asarray(image.scale, dtype=np.float32)])
        itype.append(int(image.itype))

        default = image.default
        def_option.append(int(default.option))
        def_bias.append(float(default.bias))
        def_fwhmxy.append(float(getattr(default, 'fwhmxy', 0.0)))
        def_fwhmz.append(float(getattr(default, 'fwhmz', 0.0)))
        def_squeeze.append(float(getattr(default, 'squeeze', 0.0)))
        def_sqfwhm.append(float(getattr(default, 'sqfwhm', 0.0)))

    image_flat = (
        np.concatenate(images) if images else np.array([], dtype=np.float32)
    )

    return dict(
        image=image_flat, nxyz=nxyz, vxy=vxy, vz=vz, wavel=wavel, gamma=gamma,
        scale=scale, itype=itype, tzero=float(dmap.tzero), period=float(dmap.period),
        quad=float(dmap.quad), vfine=float(dmap.vfine), sfac=float(dmap.sfac),
        def_option=def_option, def_bias=def_bias, def_fwhmxy=def_fwhmxy,
        def_fwhmz=def_fwhmz, def_squeeze=def_squeeze, def_sqfwhm=def_sqfwhm,
        shapes=shapes,
    )


def _unflatten_into_map(dmap, image_flat, shapes):
    """Writes a flat image array (as returned by `tr`/`memit`) back into
    each Image's `data` attribute, in place."""
    offset = 0
    for image, shape in zip(dmap.data, shapes):
        npix = int(np.prod(shape))
        image.data = np.asarray(
            image_flat[offset:offset + npix], dtype=np.float32
        ).reshape(shape)
        offset += npix


def _flatten_data(ddata):
    """
    Flattens a Data's Spectra into the flat arguments expected by the
    op/tr/memit Rust bindings, returning a dict that also carries enough
    information (``shapes``) to unflatten a result back into the Data's
    Spectra afterwards.
    """
    flux_parts, ferr_parts, wave_parts = [], [], []
    nwave, nspec = [], []
    time, expose, nsub, fwhm = [], [], [], []
    shapes = []

    for spectra in ddata.data:
        shapes.append(spectra.flux.shape)
        ns, nw = spectra.flux.shape
        flux_parts.append(np.ascontiguousarray(spectra.flux, dtype=np.float32).ravel())
        ferr_parts.append(np.ascontiguousarray(spectra.ferr, dtype=np.float32).ravel())
        wave_parts.append(np.ascontiguousarray(spectra.wave, dtype=np.float64).ravel())
        nwave.append(int(nw))
        nspec.append(int(ns))
        time.append([float(t) for t in np.asarray(spectra.time, dtype=np.float64)])
        expose.append([float(e) for e in np.asarray(spectra.expose, dtype=np.float32)])
        nsub.append([int(n) for n in np.asarray(spectra.nsub, dtype=np.int32)])
        fwhm.append(float(spectra.fwhm))

    flux = np.concatenate(flux_parts) if flux_parts else np.array([], dtype=np.float32)
    ferr = np.concatenate(ferr_parts) if ferr_parts else np.array([], dtype=np.float32)
    wave = np.concatenate(wave_parts) if wave_parts else np.array([], dtype=np.float64)

    return dict(
        flux=flux, ferr=ferr, wave=wave, nwave=nwave, nspec=nspec,
        time=time, expose=expose, nsub=nsub, fwhm=fwhm, shapes=shapes,
    )


def _unflatten_into_data(ddata, flux_flat, shapes):
    """Writes a flat flux array back into each Spectra's `flux` attribute,
    in place."""
    offset = 0
    for spectra, shape in zip(ddata.data, shapes):
        npix = int(np.prod(shape))
        spectra.flux = np.asarray(
            flux_flat[offset:offset + npix], dtype=np.float32
        ).reshape(shape)
        offset += npix


def comdat(dmap, ddata):
    """
    comdat(map, data)

    Computes the data equivalent to `map` using `data` as the template
    (its wavelength scales, times, exposures, sub-division factors and
    FWHM resolutions). On exit, `data`'s fluxes are overwritten in place
    with the computed values.
    """
    _require_rust()
    m = _flatten_map(dmap)
    d = _flatten_data(ddata)
    try:
        flux = _doppler_rs.op(
            m['image'], m['nxyz'], m['vxy'], m['vz'], m['wavel'], m['gamma'],
            m['scale'], m['itype'], m['tzero'], m['period'], m['quad'],
            m['vfine'], m['sfac'], d['wave'], d['nwave'], d['nspec'],
            d['time'], d['expose'], d['nsub'], d['fwhm'],
        )
    except Exception as err:
        raise DopplerError(f"doppler.comdat: {err}") from err
    _unflatten_into_data(ddata, flux, d['shapes'])


def comdef(dmap):
    """
    comdef(map)

    Computes the default image for each Image in `map` (uniform,
    Gaussian-blurred 2D, or Gaussian-blurred 3D with optional squeeze,
    according to each Image's `default` attribute). On exit, each Image's
    `data` is overwritten in place with its default.
    """
    _require_rust()
    for image in dmap.data:
        nx, ny, nz = _image_nxyz(image)
        default = image.default
        try:
            new_data = _doppler_rs.default_image(
                np.ascontiguousarray(image.data, dtype=np.float32).ravel(),
                nx, ny, nz,
                float(image.vxy),
                float(image.vz) if image.vz is not None else 0.0,
                int(default.option), float(default.bias),
                float(getattr(default, 'fwhmxy', 0.0)),
                float(getattr(default, 'fwhmz', 0.0)),
                float(getattr(default, 'squeeze', 0.0)),
                float(getattr(default, 'sqfwhm', 0.0)),
            )
        except Exception as err:
            raise DopplerError(f"doppler.comdef: {err}") from err
        image.data = np.asarray(new_data, dtype=np.float32).reshape(image.data.shape)


def datcom(ddata, dmap):
    """
    datcom(data, map)

    Computes the transpose operation to `comdat`, for testing: projects
    `data`'s fluxes back onto image space. On exit, `map`'s Images are
    overwritten in place with the (additive) back-projection.
    """
    _require_rust()
    m = _flatten_map(dmap)
    d = _flatten_data(ddata)
    try:
        image = _doppler_rs.tr(
            m['nxyz'], m['vxy'], m['vz'], m['wavel'], m['gamma'], m['scale'],
            m['itype'], m['tzero'], m['period'], m['quad'], m['vfine'], m['sfac'],
            d['flux'], d['wave'], d['nwave'], d['nspec'], d['time'], d['expose'],
            d['nsub'], d['fwhm'],
        )
    except Exception as err:
        raise DopplerError(f"doppler.datcom: {err}") from err
    _unflatten_into_map(dmap, image, m['shapes'])


def memit(dmap, ddata, niter, caim, tlim=1.e-4, rmax=0.2):
    """
    memit(map, data, niter, caim, tlim=1e-4, rmax=0.2)

    Carries out up to `niter` MEM iterations on `map` given `data`. On
    exit, `map`'s Images are overwritten in place with the reconstructed
    image. Returns `(constraint, test, entropy, iterations)` diagnostics
    for the final iteration performed (`iterations` <= `niter` if
    convergence -- `test < tlim and constraint <= caim` -- was reached
    early); the original C++ `memit` discarded these, so this return value
    can be ignored for drop-in compatibility.

    Arguments::

      map   : input Map, changed on output

      data  : input Data

      niter : number of iterations

      caim  : reduced chi**2 to aim for

      tlim  : limit on the gradient-misfit diagnostic below which
              iterations cease

      rmax  : maximum fractional change to allow per iteration

    Note: MEM iteration itself is currently provided by a temporary
    placeholder (see ``rust/memsys-stub``) pending the external `memsys`
    crate publishing its agreed library API; see ``rust/README.md``. The
    returned diagnostics should not yet be treated as scientifically
    authoritative.
    """
    _require_rust()
    if niter < 1:
        raise DopplerError('doppler.memit: niter < 1')
    if caim <= 0.:
        raise DopplerError('doppler.memit: caim <= 0')
    if tlim >= 0.5:
        raise DopplerError('doppler.memit: tlim >= 0.5')
    if rmax >= 0.5:
        raise DopplerError('doppler.memit: rmax >= 0.5')

    m = _flatten_map(dmap)
    d = _flatten_data(ddata)
    try:
        image, constraint, test, entropy, iterations = _doppler_rs.memit(
            m['image'], m['nxyz'], m['vxy'], m['vz'], m['wavel'], m['gamma'],
            m['scale'], m['itype'], m['tzero'], m['period'], m['quad'],
            m['vfine'], m['sfac'], m['def_option'], m['def_bias'],
            m['def_fwhmxy'], m['def_fwhmz'], m['def_squeeze'], m['def_sqfwhm'],
            d['flux'], d['ferr'], d['wave'], d['nwave'], d['nspec'], d['time'],
            d['expose'], d['nsub'], d['fwhm'], int(niter), float(caim),
            float(tlim), float(rmax),
        )
    except Exception as err:
        raise DopplerError(f"doppler.memit: {err}") from err
    _unflatten_into_map(dmap, image, m['shapes'])
    return constraint, test, entropy, iterations
