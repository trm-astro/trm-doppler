trm.doppler is a Python implementation of Doppler tomography. Its
performance-critical core (image<->data projection, FFT-based blurring and
the MEM iteration driver) is implemented in Rust, exposed to Python via
PyO3; see ``rust/README.md`` for details of that implementation.

Pre-requisites for installation:

  * A Rust toolchain (https://rustup.rs) and `maturin
    <https://www.maturin.rs/>`_, used to build the compiled
    ``trm.doppler._doppler_rs`` extension as part of the normal Python
    install (no separate C++ build step, and no external ``fftw``/
    ``cpp-mem`` libraries are required any more).

Python:

  astropy -- general astronomy package which is needed
             for FITS.

  numpy, scipy -- standard scientific Python packages.

Some scripts need other packages. e.g. The volume rendering script
vrend.py needs 's2plot' (but I intend to explore other options for
3D visualiation so don't reply on this).

This is the Python3 version. I am making no attempt to be backwards
compatible with Python2. I recommend installing with 'pip' as in

pip install . --user

run in the directory in which the pyproject.toml file is to be found
rather than 'python setup.py install --user' because 'pip' tracks the
files installed allowing you to uninstall them.

