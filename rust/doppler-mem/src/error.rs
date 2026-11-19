use thiserror::Error;

/// Errors from [`crate::run_memit`] and its helpers.
#[derive(Debug, Error)]
pub enum DopplerMemError {
    #[error("mismatched array lengths: {0}")]
    LengthMismatch(String),

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    #[error(transparent)]
    Core(#[from] doppler_core::DopplerError),
}
