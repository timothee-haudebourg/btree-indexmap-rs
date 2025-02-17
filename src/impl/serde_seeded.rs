use crate::BTreeIndexMap;
use serde::ser::SerializeMap;
use serde_seeded::{de::Seed, ser::Seeded, DeserializeSeeded, SerializeSeeded};

impl<K: SerializeSeeded<Q>, V: SerializeSeeded<Q>, Q> SerializeSeeded<Q> for BTreeIndexMap<K, V> {
	fn serialize_seeded<S>(&self, seed: &Q, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		let mut map = serializer.serialize_map(Some(self.len()))?;
		for (key, value) in self {
			map.serialize_entry(&Seeded::new(seed, key), &Seeded::new(seed, value))?;
		}
		map.end()
	}
}

impl<'de, Q, K, V> DeserializeSeeded<'de, Q> for BTreeIndexMap<K, V>
where
	Q: ?Sized,
	K: Ord + DeserializeSeeded<'de, Q>,
	V: DeserializeSeeded<'de, Q>,
{
	fn deserialize_seeded<D>(seed: &Q, deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		struct Visitor<'a, Q: ?Sized, K, V>(Seed<'a, Q, K>, Seed<'a, Q, V>);

		impl<'de, 'a, Q, K, V> serde::de::Visitor<'de> for Visitor<'a, Q, K, V>
		where
			Q: ?Sized,
			K: Ord + DeserializeSeeded<'de, Q>,
			V: DeserializeSeeded<'de, Q>,
		{
			type Value = BTreeIndexMap<K, V>;

			fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
				write!(formatter, "a sequence")
			}

			fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
			where
				A: serde::de::MapAccess<'de>,
			{
				let mut result = BTreeIndexMap::new();

				while let Some((key, value)) = map.next_entry_seed(self.0, self.1)? {
					result.insert(key, value);
				}

				Ok(result)
			}
		}

		deserializer.deserialize_seq(Visitor(Seed::new(seed), Seed::new(seed)))
	}
}

pub mod unseeded_btree_index_map_key {
	use serde::{ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};
	use serde_seeded::{de::Seed, ser::Seeded, DeserializeSeeded, SerializeSeeded};
	use std::{collections::BTreeMap, marker::PhantomData};

	use crate::BTreeIndexMap;

	pub fn serialize_seeded<K, V, Q, S>(
		value: &BTreeIndexMap<K, V>,
		seed: &Q,
		serializer: S,
	) -> Result<S::Ok, S::Error>
	where
		K: Serialize,
		V: SerializeSeeded<Q>,
		S: Serializer,
	{
		let mut s = serializer.serialize_map(Some(value.len()))?;

		for (key, value) in value {
			s.serialize_entry(key, &Seeded::new(seed, value))?;
		}

		s.end()
	}

	pub fn deserialize_seeded<'de, K, V, Q, D>(
		seed: &Q,
		deserializer: D,
	) -> Result<BTreeIndexMap<K, V>, D::Error>
	where
		K: Ord + Deserialize<'de>,
		V: DeserializeSeeded<'de, Q>,
		D: Deserializer<'de>,
	{
		struct Visitor<'seed, Q, K, V>(&'seed Q, PhantomData<BTreeMap<K, V>>);

		impl<'de, 'seed, Q, K, V> ::serde::de::Visitor<'de> for Visitor<'seed, Q, K, V>
		where
			K: Ord + Deserialize<'de>,
			V: DeserializeSeeded<'de, Q>,
		{
			type Value = BTreeIndexMap<K, V>;

			fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
				write!(formatter, "a map")
			}

			fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
			where
				A: serde::de::MapAccess<'de>,
			{
				let mut result = BTreeIndexMap::new();

				while let Some(key) = map.next_key()? {
					let value = map.next_value_seed(Seed::new(self.0))?;
					result.insert(key, value);
				}

				Ok(result)
			}
		}

		deserializer.deserialize_map(Visitor(seed, PhantomData))
	}
}
