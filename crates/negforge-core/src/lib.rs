//! NEGForge: a 1D ballistic-MOSFET electrostatics + NEGF transport model.
//!
//! See the crate modules for the physics of each piece, and the top-level
//! repository README for architecture, build instructions, and limitations.

pub mod charge;
pub mod constants;
pub mod device;
pub mod error;
pub mod negf;
pub mod resources;
pub mod selfconsistent;
pub mod sweep;
mod tridiag;

pub use device::{Device, DeviceParams};
pub use error::{NegForgeError, Result};
pub use negf::GreenFunctionResult;
pub use selfconsistent::{SelfConsistentOptions, SelfConsistentResult};
pub use sweep::IvPoint;
