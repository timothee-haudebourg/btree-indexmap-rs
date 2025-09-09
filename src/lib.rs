//! BTree-based map data structure with stable indexes.
pub use equivalent::Comparable;

pub mod map;
pub mod multi_map;

pub use map::BTreeIndexMap;
pub use multi_map::BTreeIndexMultiMap;

mod r#impl;
#[allow(unused_imports)]
pub use r#impl::*;
