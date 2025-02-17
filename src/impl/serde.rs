use std::marker::PhantomData;

use serde::{ser::SerializeMap, Deserialize, Serialize};

use crate::BTreeIndexMap;

impl<K: Serialize, V: Serialize> Serialize for BTreeIndexMap<K, V> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer
	{
		let mut map = serializer.serialize_map(Some(self.len()))?;
		
		for (key, value) in self {
			map.serialize_entry(key, value)?;
		}
		
		map.end()
	}
}

impl<'de, K: Ord + Deserialize<'de>, V: Deserialize<'de>> Deserialize<'de> for BTreeIndexMap<K, V> {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>
	{
		struct Visitor<K, V>(PhantomData<(K, V)>);

		impl<'de, K, V> serde::de::Visitor<'de> for Visitor<K, V> where K: Ord + Deserialize<'de>, V: Deserialize<'de> {
			type Value = BTreeIndexMap<K, V>;

			fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
				write!(formatter, "a map")
			}

			fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
			where
				A: serde::de::MapAccess<'de>
			{
				let mut result = BTreeIndexMap::new();

				while let Some((key, value)) = map.next_entry()? {
					result.insert(key, value);
				}

				Ok(result)
			}
		}

		deserializer.deserialize_map(Visitor(PhantomData))
	}
}