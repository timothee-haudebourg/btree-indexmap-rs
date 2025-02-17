//! BTree-based map data structure with stable indexes.
mod map;
pub use map::*;

mod r#impl;
#[allow(unused_imports)]
pub use r#impl::*;
