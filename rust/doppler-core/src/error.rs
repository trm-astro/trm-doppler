use thiserror::Error;

/// Errors that can arise while validating or running the `op`/`tr`
/// projections.
#[derive(Debug, Error)]
pub enum DopplerError {
    #[error("mismatched array lengths: {0}")]
    LengthMismatch(String),

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}
