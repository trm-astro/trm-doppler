#!/usr/bin/env python

"""
Doppler tomography package

Core classes & routines
"""

import numpy as np

# Version number. To be updated
# if data formats change.
VERSION = 20210826

# FWHM/sigma for a gaussian
EFAC = 2.354820045

# Speed of light in km/s
CKMS = 299792.458


def sameDims(arr1, arr2):
    """
    Checks that two numpy arrays have the same number of dimensions
    and the same dimensions
    """
    if arr1.ndim != arr2.ndim:
        return False

    for d1, d2 in zip(arr1.shape, arr2.shape):
        if d1 != d2:
            return False

    return True


def afits(fname):
    """
    Appends .fits to a filename if it does not end
    with .fits, .fit, .fits.gz or .fit.gz
    """

    if (
        fname.endswith(".fits")
        or fname.endswith(".fits")
        or fname.endswith(".fits.gz")
        or fname.endswith(".fit.gz")
    ):
        return fname
    else:
        return fname + ".fits"


def acfg(fname):
    """
    Appends .cfg to a filename if it does not end with it.
    """

    if fname.endswith(".cfg"):
        return fname
    else:
        return fname + ".cfg"


def meshgrid(nxy, vxy, nz=1, dvz=0):
    """Carries out near-equivalent to numpy's meshgrid function for
    Doppler images. i.e. it returns grids of vx,vy or vx,vy,vz which
    have C-style dimensions (nxy,nxy) or (nz,nxy,nxy) and values along
    the vx,vy,vz axes equal to the vx, vy, vz values for each
    point. These arrays are then useful for creating and manipulating
    images.

    Example: suppose we want to set an image "image" equal to a gaussian spot
    centred at Vx=100,Vy=500,Vz=100 to an array called "image", then the
    following would do this (nxy etc assumed set already, nz > 1)::

      vx,vy,vz = meshgrid(nxy, nz, vxy, dvz)

      # set an RMS and height for the spot
      sigma  = 50.
      height = 5.
      array  = height*np.exp(-((vx-100)**2+(vy-500)**2+(vz-100)**2)/sigma**2/2.)

    Arguments::

      nxy : the number of pixels along the Vx & Vy sides.
      nz  : the number of planes in Vz
      vxy : the km/s/pixel in Vx-Vy (square pixels)
      dvz : the km/s spacing in Vz

    Returns::

      vx, vy : if nz > 1, Vx and Vy coordinates of every point in 2D image
             Each array is 2D

      vx, vy, vz : if nz == 1, Vx, Vy and Vz coordinates of every point in 3D
                image. Each array is 3D

    """

    # total km/s range in Vx-Vy plane
    vrange = vxy * (nxy - 1) / 2.0

    # create 1D arrays of coords in Vx-Vy
    if nz == 1:
        # 2D case
        vy, vx = np.mgrid[
            -vrange : vrange : eval(f"{nxy}j"),
            -vrange : vrange : eval(f"{nxy}j"),
        ]
        return (vx, vy)
    else:
        nx = ny = nxy
        vzrange = dvz * (nz - 1) / 2.0
        vz, vy, vx = np.mgrid[
            -vzrange : vzrange : eval(f"{nz}j"),
            -vrange : vrange : eval(f"{nxy}j"),
            -vrange : vrange : eval(f"{nxy}j"),
        ]
        return (vx, vy, vz)


def centroid(xpos, fwhm, y, emission=True, e=None):
    """(xcen,{xerr}) = centroid(xpos, fwhm, y, e=None) -- computes weighted centroid of a feature in a 1D array.

    This uses the classic Young & Schneider cross-correlation with a gaussian method. Note that this routine will
    use all of the arrays passed to it. If it runs slowly you may be be able to speed it up by passing sub-sections
    of the arrays.

    Arguments:

    xpos     -- initial position, with centre of the first pixel defined as 0.0. This must initially be
                within the range 1 to len(y)-2, i.e. more than 1 from either end. The routine fails if it cannot
                locate an extremum of the cross-correlation only shifts
                by 3*fwhm from this position at the most.
    fwhm     -- the full width at half maximum in pixels of the gaussian which must be >=2.
    emission -- True for emission lines, False for absorption
    y        -- the array of values containing the feature. e are 1-sigma uncertainties. Must have at least 3 pixels.
    e        -- 1-sigma uncertainties if available. These are only used to estimate the uncertainty in the answer

    Returns:

    xcen     -- centroid
    xerr     -- 1-sigma uncertainty in centroid if e is not None

    Raises DopplerError exceptions if there are problems.
    """

    from scipy.optimize import brentq

    def comperr(xd, y, e, sigma):
        """Computes uncertainty in final position from median"""
        xdsq = np.power(xd, 2)
        vc = np.sum(xdsq * np.exp(-xdsq) * np.power(e, 2))
        ddc = np.sum(y * (xdsq - 1) * np.exp(-xdsq / 2))
        return sigma * np.sqrt(vc) / abs(ddc)

    def dcorr(xd, y):
        """Returns derivative of correlation with gaussian

        xd -- xd offsets from centre of gaussian, scaled by RMS
        y  -- the corresponding y values
        """
        return np.sum(y * xd * np.exp(-np.power(xd, 2) / 2))

    # check inputs
    if fwhm < 2:
        raise DopplerError("centroid: fwhm = " + str(fwhm) + " < 2")
    if xpos < 1 or xpos > len(y) - 2:
        raise DopplerError(
            "centroid: xpos = "
            + str(xpos)
            + " is out of allowed range 1 to "
            + str(len(y) - 2)
        )
    if len(y) < 3:
        raise DopplerError("median: length of y array < 3")
    if e is not None and len(e) != len(y):
        raise DopplerError("median: length of e array does not match the y array")

    sigma = fwhm / 2.3548
    x = np.arange(len(y), dtype=float)
    xd = (x - xpos) / sigma
    dc1 = dcorr(xd, y)
    if dc1 == 0:
        if e is not None:
            return (xpos, comperr(xd, y, e, sigma))
        return xpos

    # Try to bracket the peak by looking for closest switch in sign
    # of the derivative of the x-corr.

    x1 = xpos
    x2 = xpos
    dc2 = dc1
    found_switch = False

    # move in direction defined by sign of gradient.
    if (dc1 > 0.0 and emission) or (dc1 < 0.0 and not emission):
        cshift = 0.25
    else:
        cshift = -0.25

    shift = cshift
    while abs(shift) < 3.1:
        x2 = xpos + fwhm * shift
        xd = (x - x2) / sigma
        dc2 = dcorr(xd, y)
        if dc2 == 0:
            if e != None:
                return (x2, comperr(xd, y, e, sigma))
            else:
                return x2
        if (dc1 > 0 and dc2 < 0) or (dc1 < 0 and dc2 > 0):
            found_switch = True
            break
        shift += cshift

    if not found_switch:
        raise DopplerError(
            "centroid: could not find peak within 3*fwhm of initial position"
        )

    # reorder if necessary
    if x1 > x2:
        (x1, x2) = (x2, x1)
        (dc1, dc2) = (dc2, dc1)

    if x1 < 0.0 or x2 < 1.0 or x1 > len(y) - 2 or x2 > len(y) - 1:
        raise DopplerError(
            "centroid: bracketting limits = "
            + str(x1)
            + ", "
            + str(x2)
            + " out of range."
        )

    # now find root using Brent's method

    def bfunc(xpos, x, y, sigma):
        """Function for the root finder brentq"""
        return dcorr((x - xpos) / sigma, y)

    (xm, r) = brentq(bfunc, x1, x2, args=(x, y, sigma), full_output=True)
    if not r.converged:
        raise DopplerError("centroid: brentq did not converge")

    if e is not None:
        xd = (x - xm) / sigma
        return (xm, comperr(xd, y, e, sigma))
    else:
        return xm


class DopplerError(Exception):
    pass
