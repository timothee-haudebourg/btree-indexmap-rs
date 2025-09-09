use core::fmt;
use equivalent::Comparable;
use raw_btree::RawBTree;
use std::{
	cmp::Ordering,
	hash::{Hash, Hasher},
	iter::FusedIterator,
	mem::MaybeUninit,
	ops::Index,
};

#[derive(Clone)]
struct Indexes {
	first: usize,
	rest: Vec<usize>,
}

impl Indexes {
	fn new(i: usize) -> Self {
		Self {
			first: i,
			rest: Vec::new(),
		}
	}

	fn insert(&mut self, mut i: usize) {
		match i.cmp(&self.first) {
			Ordering::Equal => return,
			Ordering::Less => {
				std::mem::swap(&mut self.first, &mut i);
			}
			_ => (),
		}

		if let Err(offset) = self.rest.binary_search(&i) {
			self.rest.insert(offset, i);
		}
	}

	fn remove(&mut self, i: usize) -> Result<bool, ()> {
		if self.first == i {
			if self.rest.is_empty() {
				Err(())
			} else {
				self.first = self.rest.remove(0);
				Ok(true)
			}
		} else {
			match self.rest.binary_search(&i) {
				Ok(offset) => {
					self.rest.insert(offset, i);
					Ok(true)
				}
				Err(_) => Ok(false),
			}
		}
	}

	fn swap_last(&mut self, j: usize) {
		match self.rest.pop() {
			Some(i) => {
				self.insert(i);
			}
			None => self.first = j,
		}
	}

	fn len(&self) -> usize {
		1 + self.rest.len()
	}

	fn iter(&self) -> IndexesIter {
		IndexesIter {
			first: Some(self.first),
			rest: self.rest.iter(),
		}
	}

	fn iter_mut(&mut self) -> IndexesIterMut {
		IndexesIterMut {
			first: Some(&mut self.first),
			rest: self.rest.iter_mut(),
		}
	}
}

impl IntoIterator for Indexes {
	type Item = usize;
	type IntoIter = IndexesIntoIter;

	fn into_iter(self) -> Self::IntoIter {
		IndexesIntoIter {
			first: Some(self.first),
			rest: self.rest.into_iter(),
		}
	}
}

#[derive(Default)]
pub struct IndexesIter<'a> {
	first: Option<usize>,
	rest: std::slice::Iter<'a, usize>,
}

impl<'a> Iterator for IndexesIter<'a> {
	type Item = usize;

	fn size_hint(&self) -> (usize, Option<usize>) {
		let len = self.rest.len() + self.first.is_some().then_some(1).unwrap_or_default();
		(len, Some(len))
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.first.take().or_else(|| self.rest.next().copied())
	}
}

impl<'a> ExactSizeIterator for IndexesIter<'a> {}

#[derive(Default)]
struct IndexesIterMut<'a> {
	first: Option<&'a mut usize>,
	rest: std::slice::IterMut<'a, usize>,
}

impl<'a> Iterator for IndexesIterMut<'a> {
	type Item = &'a mut usize;

	fn size_hint(&self) -> (usize, Option<usize>) {
		let len = self.rest.len() + self.first.is_some().then_some(1).unwrap_or_default();
		(len, Some(len))
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.first.take().or_else(|| self.rest.next())
	}
}

impl<'a> ExactSizeIterator for IndexesIterMut<'a> {}

#[derive(Default)]
struct IndexesIntoIter {
	first: Option<usize>,
	rest: std::vec::IntoIter<usize>,
}

impl Iterator for IndexesIntoIter {
	type Item = usize;

	fn size_hint(&self) -> (usize, Option<usize>) {
		let len = self.rest.len() + self.first.is_some().then_some(1).unwrap_or_default();
		(len, Some(len))
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.first.take().or_else(|| self.rest.next())
	}
}

impl DoubleEndedIterator for IndexesIntoIter {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.rest.next_back().or_else(|| self.first.take())
	}
}

impl ExactSizeIterator for IndexesIntoIter {}

pub struct GetIndexedEntries<'a, K, V> {
	indexes: IndexesIter<'a>,
	entries: &'a [(K, V)],
}

impl<'a, K, V> Iterator for GetIndexedEntries<'a, K, V> {
	type Item = (usize, &'a K, &'a V);

	fn next(&mut self) -> Option<Self::Item> {
		self.indexes.next().map(|i| {
			let (k, v) = &self.entries[i];
			(i, k, v)
		})
	}
}

pub struct GetEntries<'a, K, V>(GetIndexedEntries<'a, K, V>);

impl<'a, K, V> Iterator for GetEntries<'a, K, V> {
	type Item = (&'a K, &'a V);

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, k, v)| (k, v))
	}
}

pub struct Get<'a, K, V>(GetIndexedEntries<'a, K, V>);

impl<'a, K, V> Iterator for Get<'a, K, V> {
	type Item = &'a V;

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, _, v)| v)
	}
}

pub struct GetIndexed<'a, K, V>(GetIndexedEntries<'a, K, V>);

impl<'a, K, V> Iterator for GetIndexed<'a, K, V> {
	type Item = (usize, &'a V);

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(i, _, v)| (i, v))
	}
}

pub struct GetIndexedEntriesMut<'a, K, V> {
	indexes: IndexesIter<'a>,
	entries: &'a mut [(K, V)],
}

impl<'a, K, V> Iterator for GetIndexedEntriesMut<'a, K, V> {
	type Item = (usize, &'a K, &'a mut V);

	fn next(&mut self) -> Option<Self::Item> {
		self.indexes.next().map(|i| {
			let (k, v) = &mut self.entries[i];
			let (k, v) = unsafe {
				// SAFEY: values are not aliased, and cannot be borrowed more
				//        than once.
				std::mem::transmute::<(&mut K, &mut V), (&'a K, &'a mut V)>((k, v))
			};

			(i, k, v)
		})
	}
}

pub struct GetEntriesMut<'a, K, V>(GetIndexedEntriesMut<'a, K, V>);

impl<'a, K, V> Iterator for GetEntriesMut<'a, K, V> {
	type Item = (&'a K, &'a mut V);

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, k, v)| (k, v))
	}
}

pub struct GetMut<'a, K, V>(GetIndexedEntriesMut<'a, K, V>);

impl<'a, K, V> Iterator for GetMut<'a, K, V> {
	type Item = &'a mut V;

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, _, v)| v)
	}
}

#[derive(Clone)]
pub struct BTreeIndexMultiMap<K, V> {
	entries: Vec<(K, V)>,
	map: RawBTree<Indexes>,
}

impl<K, V> Default for BTreeIndexMultiMap<K, V> {
	fn default() -> Self {
		Self::new()
	}
}

impl<K, V> BTreeIndexMultiMap<K, V> {
	pub fn new() -> Self {
		Self {
			entries: Vec::new(),
			map: RawBTree::new(),
		}
	}

	pub fn with_capacity(cap: usize) -> Self {
		Self {
			entries: Vec::with_capacity(cap),
			map: RawBTree::new(),
		}
	}

	pub fn capacity(&self) -> usize {
		self.entries.capacity()
	}

	pub fn as_entries(&self) -> &[(K, V)] {
		&self.entries
	}

	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}

	pub fn len(&self) -> usize {
		self.entries.len()
	}

	pub fn first(&self) -> Option<(&K, &V)> {
		self.entries.first().map(|(k, v)| (k, v))
	}

	pub fn last(&self) -> Option<(&K, &V)> {
		self.entries.last().map(|(k, v)| (k, v))
	}

	pub fn contains_key<Q>(&self, key: &Q) -> bool
	where
		Q: ?Sized + Comparable<K>,
	{
		self.map.get(outer_cmp(&self.entries), key).is_some()
	}

	pub fn key_occurences<Q>(&self, key: &Q) -> usize
	where
		Q: ?Sized + Comparable<K>,
	{
		self.map
			.get(outer_cmp(&self.entries), key)
			.map(|i| i.len())
			.unwrap_or_default()
	}

	pub fn indexes_of<Q>(&self, key: &Q) -> IndexesIter
	where
		Q: ?Sized + Comparable<K>,
	{
		self.map
			.get(outer_cmp(&self.entries), key)
			.map(Indexes::iter)
			.unwrap_or_default()
	}

	pub fn index_of<Q>(&self, key: &Q) -> Option<usize>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.indexes_of(key).next()
	}

	pub fn get_indexed_entries<Q>(&self, key: &Q) -> GetIndexedEntries<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		GetIndexedEntries {
			indexes: self
				.map
				.get(outer_cmp(&self.entries), key)
				.map(Indexes::iter)
				.unwrap_or_default(),
			entries: &self.entries,
		}
	}

	pub fn get_or_insert_indexed_entries_with(
		&mut self,
		key: K,
		f: impl FnOnce() -> V,
	) -> GetIndexedEntries<K, V>
	where
		K: Ord,
	{
		match self.map.address_of(outer_cmp(&self.entries), &key) {
			Ok(addr) => GetIndexedEntries {
				indexes: unsafe {
					// SAFETY: no `self.map` has been deallocated since we found
					//         `addr`.
					self.map.get_at(addr)
				}
				.unwrap()
				.iter(),
				entries: &self.entries,
			},
			Err(_) => {
				let (i, _) = self.push_back(key, f());
				GetIndexedEntries {
					indexes: self
						.map
						.get(inner_index_cmp(&self.entries), &i)
						.unwrap()
						.iter(),
					entries: &self.entries,
				}
			}
		}
	}

	pub fn get_or_insert_indexed_entries(&mut self, key: K, value: V) -> GetIndexedEntries<K, V>
	where
		K: Ord,
	{
		self.get_or_insert_indexed_entries_with(key, || value)
	}

	pub fn get_entries<Q>(&self, key: &Q) -> GetEntries<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		GetEntries(self.get_indexed_entries(key))
	}

	pub fn get_or_insert_entries_with(&mut self, key: K, f: impl FnOnce() -> V) -> GetEntries<K, V>
	where
		K: Ord,
	{
		GetEntries(self.get_or_insert_indexed_entries_with(key, f))
	}

	pub fn get_or_insert_entries(&mut self, key: K, value: V) -> GetEntries<K, V>
	where
		K: Ord,
	{
		self.get_or_insert_entries_with(key, || value)
	}

	pub fn get<Q>(&self, key: &Q) -> Get<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		Get(self.get_indexed_entries(key))
	}

	pub fn get_or_insert_with(&mut self, key: K, f: impl FnOnce() -> V) -> Get<K, V>
	where
		K: Ord,
	{
		Get(self.get_or_insert_indexed_entries_with(key, f))
	}

	pub fn get_or_insert(&mut self, key: K, value: V) -> Get<K, V>
	where
		K: Ord,
	{
		self.get_or_insert_with(key, || value)
	}

	pub fn get_indexed<Q>(&self, key: &Q) -> GetIndexed<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		GetIndexed(self.get_indexed_entries(key))
	}

	pub fn get_or_insert_indexed_with(&mut self, key: K, f: impl FnOnce() -> V) -> GetIndexed<K, V>
	where
		K: Ord,
	{
		GetIndexed(self.get_or_insert_indexed_entries_with(key, f))
	}

	pub fn get_or_insert_indexed(&mut self, key: K, value: V) -> GetIndexed<K, V>
	where
		K: Ord,
	{
		self.get_or_insert_indexed_with(key, || value)
	}

	pub fn get_indexed_entries_mut<Q>(&mut self, key: &Q) -> GetIndexedEntriesMut<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		GetIndexedEntriesMut {
			indexes: self
				.map
				.get(outer_cmp(&self.entries), key)
				.map(Indexes::iter)
				.unwrap_or_default(),
			entries: &mut self.entries,
		}
	}

	pub fn get_or_insert_indexed_entries_mut_with(
		&mut self,
		key: K,
		f: impl FnOnce() -> V,
	) -> GetIndexedEntriesMut<K, V>
	where
		K: Ord,
	{
		match self.map.address_of(outer_cmp(&self.entries), &key) {
			Ok(addr) => GetIndexedEntriesMut {
				indexes: unsafe {
					// SAFETY: no `self.map` has been deallocated since we found
					//         `addr`.
					self.map.get_at(addr)
				}
				.unwrap()
				.iter(),
				entries: &mut self.entries,
			},
			Err(_) => {
				let (i, _) = self.push_back(key, f());
				GetIndexedEntriesMut {
					indexes: self
						.map
						.get(inner_index_cmp(&self.entries), &i)
						.unwrap()
						.iter(),
					entries: &mut self.entries,
				}
			}
		}
	}

	pub fn get_or_insert_indexed_entries_mut(
		&mut self,
		key: K,
		value: V,
	) -> GetIndexedEntriesMut<K, V>
	where
		K: Ord,
	{
		self.get_or_insert_indexed_entries_mut_with(key, || value)
	}

	pub fn get_entries_mut<Q>(&mut self, key: &Q) -> GetEntriesMut<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		GetEntriesMut(self.get_indexed_entries_mut(key))
	}

	pub fn get_or_insert_entries_mut_with(
		&mut self,
		key: K,
		f: impl FnOnce() -> V,
	) -> GetEntriesMut<K, V>
	where
		K: Ord,
	{
		GetEntriesMut(self.get_or_insert_indexed_entries_mut_with(key, f))
	}

	pub fn get_or_insert_entries_mut(&mut self, key: K, value: V) -> GetEntriesMut<K, V>
	where
		K: Ord,
	{
		self.get_or_insert_entries_mut_with(key, || value)
	}

	pub fn get_mut<Q>(&mut self, key: &Q) -> GetMut<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		GetMut(self.get_indexed_entries_mut(key))
	}

	pub fn get_or_insert_mut_with(&mut self, key: K, f: impl FnOnce() -> V) -> GetMut<K, V>
	where
		K: Ord,
	{
		GetMut(self.get_or_insert_indexed_entries_mut_with(key, f))
	}

	pub fn get_or_insert_mut(&mut self, key: K, value: V) -> GetMut<K, V>
	where
		K: Ord,
	{
		self.get_or_insert_mut_with(key, || value)
	}

	pub fn entry(&mut self, key: K) -> Entry<K, V>
	where
		K: Ord,
	{
		if let Some(indices) = self.map.get(outer_cmp(&self.entries), &key) {
			return Entry::Occupied(OccupiedEntry {
				entries: &mut self.entries,
				indices,
			});
		}

		Entry::Vacant(VacantEntry {
			map: unsafe {
				// SAFETY: The compiler thinks `self` is still borrowed at this
				// point because `indices` has type `&'_ Indices` where `'_`
				// includes both the outer scope lifetime and the `self.map`
				// lifetime. However we know that `indices` is dropped before
				// this line is ever reached.
				std::mem::transmute_copy(&self)
			},
			key,
		})
	}

	pub fn push_back(&mut self, key: K, value: V) -> (usize, bool)
	where
		K: Ord,
	{
		let i = self.entries.len();
		self.entries.push((key, value));

		let unique = match self.map.get_mut(inner_index_cmp(&self.entries), &i) {
			Some(indexes) => {
				indexes.rest.push(i);
				false
			}
			None => {
				self.map.insert(inner_cmp(&self.entries), Indexes::new(i));
				true
			}
		};

		(i, unique)
	}

	/// # Panics
	///
	/// Panics if `index > len`.
	pub fn insert_at(&mut self, index: usize, key: K, value: V) -> bool
	where
		K: Ord,
	{
		for js in self.map.iter_mut() {
			for j in js.iter_mut() {
				if *j >= index {
					*j += 1;
				}
			}
		}

		self.entries.insert(index, (key, value));

		match self.map.get_mut(inner_index_cmp(&self.entries), &index) {
			Some(indexes) => {
				indexes.insert(index);
				false
			}
			None => {
				self.map
					.insert(inner_cmp(&self.entries), Indexes::new(index));
				true
			}
		}
	}

	pub fn push_front(&mut self, key: K, value: V) -> bool
	where
		K: Ord,
	{
		self.insert_at(0, key, value)
	}

	pub fn shift_insert_full(&mut self, key: K, value: V) -> (usize, ShiftInsert<K, V>)
	where
		K: Ord,
	{
		match self.map.get_mut(outer_cmp(&self.entries), &key) {
			Some(indices) => {
				let i = indices.first;
				let mut indices = std::mem::replace(indices, Indexes::new(i))
					.into_iter()
					.enumerate();
				indices.next();

				let first = std::mem::replace(&mut self.entries[i].1, value);

				(
					i,
					ShiftInsert {
						first: Some(first),
						rest: ShiftRemove(ShiftRemoveIndexedEntries { indices, map: self }),
					},
				)
			}
			None => {
				let (i, _) = self.push_back(key, value);
				(
					i,
					ShiftInsert {
						first: None,
						rest: ShiftRemove(ShiftRemoveIndexedEntries {
							indices: Default::default(),
							map: self,
						}),
					},
				)
			}
		}
	}

	pub fn shift_insert(&mut self, key: K, value: V) -> ShiftInsert<K, V>
	where
		K: Ord,
	{
		self.shift_insert_full(key, value).1
	}

	pub fn swap_insert_full(&mut self, key: K, value: V) -> (usize, SwapInsert<K, V>)
	where
		K: Ord,
	{
		match self.map.get_mut(outer_cmp(&self.entries), &key) {
			Some(indices) => {
				let i = indices.first;
				let mut indices = std::mem::replace(indices, Indexes::new(i)).into_iter();
				indices.next();

				let first = std::mem::replace(&mut self.entries[i].1, value);

				(
					i,
					SwapInsert {
						first: Some(first),
						rest: SwapRemove(SwapRemoveIndexedEntries { indices, map: self }),
					},
				)
			}
			None => {
				let (i, _) = self.push_back(key, value);
				(
					i,
					SwapInsert {
						first: None,
						rest: SwapRemove(SwapRemoveIndexedEntries {
							indices: Default::default(),
							map: self,
						}),
					},
				)
			}
		}
	}

	pub fn swap_insert(&mut self, key: K, value: V) -> SwapInsert<K, V>
	where
		K: Ord,
	{
		self.swap_insert_full(key, value).1
	}

	pub fn shift_insert_front(&mut self, key: K, value: V) -> ShiftInsertFront<K, V>
	where
		K: Ord,
	{
		ShiftInsertFront {
			remove: self.shift_remove(&key),
			entry: Some((key, value)),
		}
	}

	pub fn swap_insert_front(&mut self, key: K, value: V) -> SwapInsertFront<K, V>
	where
		K: Ord,
	{
		SwapInsertFront {
			remove: self.swap_remove(&key),
			entry: Some((key, value)),
		}
	}

	pub fn shift_insert_back_full(&mut self, key: K, value: V) -> (usize, ShiftInsertBack<K, V>)
	where
		K: Ord,
	{
		let len = self.len();
		let remove = self.shift_remove(&key);
		let i = len - remove.len();

		(
			i,
			ShiftInsertBack {
				remove,
				entry: Some((key, value)),
			},
		)
	}

	pub fn shift_insert_back(&mut self, key: K, value: V) -> ShiftInsertBack<K, V>
	where
		K: Ord,
	{
		ShiftInsertBack {
			remove: self.shift_remove(&key),
			entry: Some((key, value)),
		}
	}

	/// Alias of [`Self::shift_insert_back`].
	pub fn insert(&mut self, key: K, value: V) -> ShiftInsertBack<K, V>
	where
		K: Ord,
	{
		self.shift_insert_back(key, value)
	}

	pub fn swap_insert_back_full(&mut self, key: K, value: V) -> (usize, SwapInsertBack<K, V>)
	where
		K: Ord,
	{
		let len = self.len();
		let remove = self.swap_remove(&key);
		let i = len - remove.len();

		(
			i,
			SwapInsertBack {
				remove,
				entry: Some((key, value)),
			},
		)
	}

	pub fn swap_insert_back(&mut self, key: K, value: V) -> SwapInsertBack<K, V>
	where
		K: Ord,
	{
		SwapInsertBack {
			remove: self.swap_remove(&key),
			entry: Some((key, value)),
		}
	}

	pub fn shift_remove_at(&mut self, i: usize) -> Option<(K, V)>
	where
		K: Ord,
	{
		self.map
			.address_of(inner_index_cmp(&self.entries), &i)
			.ok()
			.map(|addr| {
				let removed = unsafe { self.map.get_mut_at(addr).unwrap().remove(i) };

				if removed.is_err() {
					unsafe {
						// SAFETY: the node hasn't moved and hasn't been
						//         deallocated since we found its address.
						self.map.remove_at(addr).unwrap();
					}
				}

				let entry = self.entries.remove(i);

				for js in self.map.iter_mut() {
					for j in js.iter_mut() {
						if *j > i {
							*j -= 1;
						}
					}
				}

				entry
			})
	}

	pub fn swap_remove_at(&mut self, i: usize) -> Option<(K, V)>
	where
		K: Ord,
	{
		self.map
			.address_of(inner_index_cmp(&self.entries), &i)
			.ok()
			.map(|addr| {
				let removed = unsafe { self.map.get_mut_at(addr).unwrap().remove(i) };

				if removed.is_err() {
					unsafe {
						// SAFETY: the node hasn't moved and hasn't been
						//         deallocated since we found its address.
						self.map.remove_at(addr).unwrap();
					}
				}

				self.swap_already_removed_at(i)
			})
	}

	fn swap_already_removed_at(&mut self, i: usize) -> (K, V)
	where
		K: Ord,
	{
		let mut entry = self.entries.pop().unwrap();

		if !self.entries.is_empty() {
			self.map
				.get_mut(outer_cmp(&self.entries), &entry.0)
				.unwrap()
				.swap_last(i);

			std::mem::swap(&mut self.entries[i], &mut entry);
		}

		entry
	}

	pub fn shift_remove_indexed_entries<Q>(&mut self, key: &Q) -> ShiftRemoveIndexedEntries<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		let indices = self
			.map
			.remove(outer_cmp(&self.entries), key)
			.map(Indexes::into_iter)
			.unwrap_or_default()
			.enumerate();

		ShiftRemoveIndexedEntries { indices, map: self }
	}

	pub fn shift_remove_entries<Q>(&mut self, key: &Q) -> ShiftRemoveEntries<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		ShiftRemoveEntries(self.shift_remove_indexed_entries(key))
	}

	pub fn shift_remove<Q>(&mut self, key: &Q) -> ShiftRemove<K, V>
	where
		Q: ?Sized + Comparable<K>,
	{
		ShiftRemove(self.shift_remove_indexed_entries(key))
	}

	pub fn swap_remove_indexed_entries<Q>(&mut self, key: &Q) -> SwapRemoveIndexedEntries<K, V>
	where
		Q: ?Sized + Comparable<K>,
		K: Ord,
	{
		let indices = self
			.map
			.remove(outer_cmp(&self.entries), key)
			.map(Indexes::into_iter)
			.unwrap_or_default();

		SwapRemoveIndexedEntries { indices, map: self }
	}

	pub fn swap_remove_entries<Q>(&mut self, key: &Q) -> SwapRemoveEntries<K, V>
	where
		Q: ?Sized + Comparable<K>,
		K: Ord,
	{
		SwapRemoveEntries(self.swap_remove_indexed_entries(key))
	}

	pub fn swap_remove<Q>(&mut self, key: &Q) -> SwapRemove<K, V>
	where
		Q: ?Sized + Comparable<K>,
		K: Ord,
	{
		SwapRemove(self.swap_remove_indexed_entries(key))
	}

	pub fn iter(&self) -> Iter<K, V> {
		Iter(self.entries.iter())
	}

	pub fn iter_mut(&mut self) -> IterMut<K, V> {
		IterMut(self.entries.iter_mut())
	}

	pub fn iter_sorted(&self) -> IterSorted<K, V> {
		IterSorted {
			entries: &self.entries,
			inner: self.map.iter(),
			current: None,
		}
	}

	pub fn iter_mut_sorted(&mut self) -> IterMutSorted<K, V> {
		IterMutSorted {
			entries: &mut self.entries,
			inner: self.map.iter(),
			current: None,
		}
	}

	pub fn into_iter_sorted(self) -> IntoIterSorted<K, V> {
		IntoIterSorted {
			entries: unsafe {
				// SAFETY: `(K, V)` and `MaybeUninit<(K, V)>` have the exact
				//         same representation.
				std::mem::transmute::<Vec<(K, V)>, Vec<MaybeUninit<(K, V)>>>(self.entries)
			},
			inner: self.map.into_iter(),
			current: None,
		}
	}

	pub fn keys(&self) -> Keys<K, V> {
		Keys(self.entries.iter())
	}

	pub fn values(&self) -> Values<K, V> {
		Values(self.entries.iter())
	}

	pub fn values_mut(&mut self) -> ValuesMut<K, V> {
		ValuesMut(self.entries.iter_mut())
	}

	/// Sort the entries by key name.
	///
	/// The relative order of entries with the same key is unchanged.
	pub fn sort(&mut self)
	where
		K: Ord,
	{
		let this = std::mem::take(self);
		self.extend(this.into_iter_sorted())
	}
}

pub struct ShiftInsertFront<'a, K: Ord, V> {
	remove: ShiftRemove<'a, K, V>,
	entry: Option<(K, V)>,
}

impl<'a, K: Ord, V> Iterator for ShiftInsertFront<'a, K, V> {
	type Item = V;

	fn next(&mut self) -> Option<Self::Item> {
		self.remove.next()
	}
}

impl<'a, K: Ord, V> Drop for ShiftInsertFront<'a, K, V> {
	fn drop(&mut self) {
		while let Some(_) = self.remove.0.next() {}
		if let Some((key, value)) = self.entry.take() {
			self.remove.0.map.push_front(key, value);
		}
	}
}

pub struct SwapInsertFront<'a, K: Ord, V> {
	remove: SwapRemove<'a, K, V>,
	entry: Option<(K, V)>,
}

impl<'a, K: Ord, V> Iterator for SwapInsertFront<'a, K, V> {
	type Item = V;

	fn next(&mut self) -> Option<Self::Item> {
		self.remove.next()
	}
}

impl<'a, K: Ord, V> Drop for SwapInsertFront<'a, K, V> {
	fn drop(&mut self) {
		while let Some(_) = self.remove.0.next() {}
		if let Some((key, value)) = self.entry.take() {
			self.remove.0.map.push_front(key, value);
		}
	}
}

pub struct ShiftInsertBack<'a, K: Ord, V> {
	remove: ShiftRemove<'a, K, V>,
	entry: Option<(K, V)>,
}

impl<'a, K: Ord, V> Iterator for ShiftInsertBack<'a, K, V> {
	type Item = V;

	fn next(&mut self) -> Option<Self::Item> {
		self.remove.next()
	}
}

impl<'a, K: Ord, V> Drop for ShiftInsertBack<'a, K, V> {
	fn drop(&mut self) {
		while let Some(_) = self.remove.0.next() {}
		if let Some((key, value)) = self.entry.take() {
			self.remove.0.map.push_back(key, value);
		}
	}
}

pub struct SwapInsertBack<'a, K: Ord, V> {
	remove: SwapRemove<'a, K, V>,
	entry: Option<(K, V)>,
}

impl<'a, K: Ord, V> Iterator for SwapInsertBack<'a, K, V> {
	type Item = V;

	fn next(&mut self) -> Option<Self::Item> {
		self.remove.next()
	}
}

impl<'a, K: Ord, V> Drop for SwapInsertBack<'a, K, V> {
	fn drop(&mut self) {
		while let Some(_) = self.remove.0.next() {}
		if let Some((key, value)) = self.entry.take() {
			self.remove.0.map.push_back(key, value);
		}
	}
}

pub struct ShiftInsert<'a, K, V> {
	first: Option<V>,
	rest: ShiftRemove<'a, K, V>,
}

impl<'a, K, V> Iterator for ShiftInsert<'a, K, V> {
	type Item = V;

	fn next(&mut self) -> Option<Self::Item> {
		self.first.take().or_else(|| self.rest.next())
	}
}

pub struct SwapInsert<'a, K: Ord, V> {
	first: Option<V>,
	rest: SwapRemove<'a, K, V>,
}

impl<'a, K: Ord, V> Iterator for SwapInsert<'a, K, V> {
	type Item = V;

	fn next(&mut self) -> Option<Self::Item> {
		self.first.take().or_else(|| self.rest.next())
	}
}

pub struct SwapRemoveIndexedEntries<'a, K: Ord, V> {
	indices: IndexesIntoIter,
	map: &'a mut BTreeIndexMultiMap<K, V>,
}

impl<'a, K: Ord, V> Iterator for SwapRemoveIndexedEntries<'a, K, V> {
	type Item = (usize, K, V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.indices.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.indices.next().map(|i| {
			let (k, v) = self.map.swap_already_removed_at(i);
			(i, k, v)
		})
	}
}

impl<'a, K: Ord, V> ExactSizeIterator for SwapRemoveIndexedEntries<'a, K, V> {}

impl<'a, K: Ord, V> Drop for SwapRemoveIndexedEntries<'a, K, V> {
	fn drop(&mut self) {
		// Consume the rest of the iterator to make sure everything is removed.
		for _ in self {}
	}
}

pub struct SwapRemoveEntries<'a, K: Ord, V>(SwapRemoveIndexedEntries<'a, K, V>);

impl<'a, K: Ord, V> Iterator for SwapRemoveEntries<'a, K, V> {
	type Item = (K, V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, k, v)| (k, v))
	}
}

impl<'a, K: Ord, V> ExactSizeIterator for SwapRemoveEntries<'a, K, V> {}

pub struct SwapRemove<'a, K: Ord, V>(SwapRemoveIndexedEntries<'a, K, V>);

impl<'a, K: Ord, V> Iterator for SwapRemove<'a, K, V> {
	type Item = V;

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, _, v)| v)
	}
}

impl<'a, K: Ord, V> ExactSizeIterator for SwapRemove<'a, K, V> {}

pub struct ShiftRemoveIndexedEntries<'a, K, V> {
	indices: std::iter::Enumerate<IndexesIntoIter>,
	map: &'a mut BTreeIndexMultiMap<K, V>,
}

impl<'a, K, V> Iterator for ShiftRemoveIndexedEntries<'a, K, V> {
	type Item = (usize, K, V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.indices.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.indices.next().map(|(shift, i)| {
			let shifted_i = i - shift;
			let (k, v) = self.map.entries.remove(shifted_i);

			// Shift larger indices.
			for js in self.map.map.iter_mut() {
				for j in js.iter_mut() {
					if *j >= shifted_i {
						*j -= 1;
					}
				}
			}

			(i, k, v)
		})
	}
}

impl<'a, K, V> ExactSizeIterator for ShiftRemoveIndexedEntries<'a, K, V> {}

impl<'a, K, V> Drop for ShiftRemoveIndexedEntries<'a, K, V> {
	fn drop(&mut self) {
		// Consume the rest of the iterator to make sure everything is removed.
		for _ in self {}
	}
}

pub struct ShiftRemoveEntries<'a, K, V>(ShiftRemoveIndexedEntries<'a, K, V>);

impl<'a, K, V> Iterator for ShiftRemoveEntries<'a, K, V> {
	type Item = (K, V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, k, v)| (k, v))
	}
}

impl<'a, K, V> ExactSizeIterator for ShiftRemoveEntries<'a, K, V> {}

pub struct ShiftRemove<'a, K, V>(ShiftRemoveIndexedEntries<'a, K, V>);

impl<'a, K, V> Iterator for ShiftRemove<'a, K, V> {
	type Item = V;

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, _, v)| v)
	}
}

impl<'a, K, V> ExactSizeIterator for ShiftRemove<'a, K, V> {}

fn inner_index_cmp<K, V>(
	entries: &[(K, V)],
) -> impl use<'_, K, V> + Fn(&Indexes, &usize) -> Ordering
where
	K: Ord,
{
	|i, &j| entries[i.first].0.cmp(&entries[j].0)
}

fn inner_cmp<K, V>(entries: &[(K, V)]) -> impl use<'_, K, V> + Fn(&Indexes, &Indexes) -> Ordering
where
	K: Ord,
{
	|i, j| entries[i.first].0.cmp(&entries[j.first].0)
}

fn outer_cmp<K, V, Q>(entries: &[(K, V)]) -> impl use<'_, K, V, Q> + Fn(&Indexes, &Q) -> Ordering
where
	Q: ?Sized + Comparable<K>,
{
	|i, q| q.compare(&entries[i.first].0).reverse()
}

impl<K, V, Q> Index<&Q> for BTreeIndexMultiMap<K, V>
where
	Q: ?Sized + Comparable<K>,
{
	type Output = V;

	fn index(&self, key: &Q) -> &Self::Output {
		self.get(key).next().expect("no entry found for key")
	}
}

pub type IntoIter<K, V> = std::vec::IntoIter<(K, V)>;

impl<K, V> IntoIterator for BTreeIndexMultiMap<K, V> {
	type IntoIter = IntoIter<K, V>;
	type Item = (K, V);

	fn into_iter(self) -> Self::IntoIter {
		self.entries.into_iter()
	}
}

impl<K: PartialEq, V: PartialEq> PartialEq for BTreeIndexMultiMap<K, V> {
	fn eq(&self, other: &Self) -> bool {
		self.iter_sorted().eq(other.iter_sorted())
	}
}

impl<K: Eq, V: Eq> Eq for BTreeIndexMultiMap<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for BTreeIndexMultiMap<K, V> {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		self.iter_sorted().partial_cmp(other.iter_sorted())
	}
}

impl<K: Ord, V: Ord> Ord for BTreeIndexMultiMap<K, V> {
	fn cmp(&self, other: &Self) -> Ordering {
		self.iter_sorted().cmp(other.iter_sorted())
	}
}

impl<K: Hash, V: Hash> Hash for BTreeIndexMultiMap<K, V> {
	fn hash<H: Hasher>(&self, state: &mut H) {
		state.write_usize(self.len());
		for item in self.iter_sorted() {
			item.hash(state);
		}
	}
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for BTreeIndexMultiMap<K, V> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_set().entries(self.iter()).finish()
	}
}

impl<K: Ord, V> Extend<(K, V)> for BTreeIndexMultiMap<K, V> {
	fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
		for (key, value) in iter {
			self.push_back(key, value);
		}
	}
}

impl<K: Ord, V> FromIterator<(K, V)> for BTreeIndexMultiMap<K, V> {
	fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
		let mut result = Self::new();
		result.extend(iter);
		result
	}
}

pub struct Iter<'a, K, V>(std::slice::Iter<'a, (K, V)>);

impl<'a, K, V> Iterator for Iter<'a, K, V> {
	type Item = (&'a K, &'a V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(k, v)| (k, v))
	}
}

impl<K, V> DoubleEndedIterator for Iter<'_, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(k, v)| (k, v))
	}
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

impl<'a, K, V> IntoIterator for &'a BTreeIndexMultiMap<K, V> {
	type IntoIter = Iter<'a, K, V>;
	type Item = (&'a K, &'a V);

	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}

pub struct IterMut<'a, K, V>(std::slice::IterMut<'a, (K, V)>);

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
	type Item = (&'a K, &'a mut V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(k, v)| (&*k, v))
	}
}

impl<K, V> DoubleEndedIterator for IterMut<'_, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(k, v)| (&*k, v))
	}
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}

impl<K, V> FusedIterator for IterMut<'_, K, V> {}

impl<'a, K, V> IntoIterator for &'a mut BTreeIndexMultiMap<K, V> {
	type IntoIter = IterMut<'a, K, V>;
	type Item = (&'a K, &'a mut V);

	fn into_iter(self) -> Self::IntoIter {
		self.iter_mut()
	}
}

pub struct IterSorted<'a, K, V> {
	entries: &'a [(K, V)],
	inner: raw_btree::Iter<'a, Indexes>,
	current: Option<IndexesIter<'a>>,
}

impl<'a, K, V> Iterator for IterSorted<'a, K, V> {
	type Item = (&'a K, &'a V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match &mut self.current {
				Some(indexes) => match indexes.next() {
					Some(i) => {
						let (k, v) = &self.entries[i];
						break Some((k, v));
					}
					None => self.current = None,
				},
				None => match self.inner.next() {
					Some(indexes) => self.current = Some(indexes.iter()),
					None => break None,
				},
			}
		}
	}
}

impl<K, V> ExactSizeIterator for IterSorted<'_, K, V> {}

impl<K, V> FusedIterator for IterSorted<'_, K, V> {}

pub struct IterMutSorted<'a, K, V> {
	entries: &'a mut [(K, V)],
	inner: raw_btree::Iter<'a, Indexes>,
	current: Option<IndexesIter<'a>>,
}

impl<'a, K, V> Iterator for IterMutSorted<'a, K, V> {
	type Item = (&'a K, &'a mut V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match &mut self.current {
				Some(indexes) => match indexes.next() {
					Some(i) => {
						let (k, v) = unsafe {
							std::mem::transmute::<&mut (K, V), &mut (K, V)>(&mut self.entries[i])
						};
						break Some((k, v));
					}
					None => self.current = None,
				},
				None => match self.inner.next() {
					Some(indexes) => self.current = Some(indexes.iter()),
					None => break None,
				},
			}
		}
	}
}

impl<K, V> ExactSizeIterator for IterMutSorted<'_, K, V> {}

impl<K, V> FusedIterator for IterMutSorted<'_, K, V> {}

pub struct IntoIterSorted<K, V> {
	entries: Vec<MaybeUninit<(K, V)>>,
	inner: raw_btree::IntoIter<Indexes>,
	current: Option<IndexesIntoIter>,
}

impl<K, V> Iterator for IntoIterSorted<K, V> {
	type Item = (K, V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match &mut self.current {
				Some(indexes) => match indexes.next() {
					Some(i) => {
						let (k, v) = unsafe { self.entries[i].assume_init_read() };
						break Some((k, v));
					}
					None => self.current = None,
				},
				None => match self.inner.next() {
					Some(indexes) => self.current = Some(indexes.into_iter()),
					None => break None,
				},
			}
		}
	}
}

impl<K, V> ExactSizeIterator for IntoIterSorted<K, V> {}

impl<K, V> FusedIterator for IntoIterSorted<K, V> {}

impl<K, V> Drop for IntoIterSorted<K, V> {
	fn drop(&mut self) {
		// Make sure the remaining entries are consumed.
		self.last();
	}
}

pub struct Keys<'a, K, V>(std::slice::Iter<'a, (K, V)>);

impl<'a, K, V> Iterator for Keys<'a, K, V> {
	type Item = &'a K;

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(k, _)| k)
	}
}

impl<K, V> DoubleEndedIterator for Keys<'_, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(k, _)| k)
	}
}

impl<K, V> ExactSizeIterator for Keys<'_, K, V> {}

impl<K, V> FusedIterator for Keys<'_, K, V> {}

pub struct Values<'a, K, V>(std::slice::Iter<'a, (K, V)>);

impl<'a, K, V> Iterator for Values<'a, K, V> {
	type Item = &'a V;

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, v)| v)
	}
}

impl<K, V> DoubleEndedIterator for Values<'_, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(_, v)| v)
	}
}

impl<K, V> ExactSizeIterator for Values<'_, K, V> {}

impl<K, V> FusedIterator for Values<'_, K, V> {}

pub struct ValuesMut<'a, K, V>(std::slice::IterMut<'a, (K, V)>);

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
	type Item = &'a mut V;

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|(_, v)| v)
	}
}

impl<K, V> DoubleEndedIterator for ValuesMut<'_, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(_, v)| v)
	}
}

impl<K, V> ExactSizeIterator for ValuesMut<'_, K, V> {}

impl<K, V> FusedIterator for ValuesMut<'_, K, V> {}

pub enum Entry<'a, K, V> {
	Vacant(VacantEntry<'a, K, V>),
	Occupied(OccupiedEntry<'a, K, V>),
}

pub struct VacantEntry<'a, K, V> {
	map: &'a mut BTreeIndexMultiMap<K, V>,
	key: K,
}

impl<'a, K, V> VacantEntry<'a, K, V> {
	pub fn key(&self) -> &K {
		&self.key
	}

	pub fn into_key(self) -> K {
		self.key
	}

	pub fn insert_full(self, value: V) -> (usize, &'a K, &'a mut V)
	where
		K: Ord,
	{
		let (i, _) = self.map.push_back(self.key, value);
		let (k, v) = &mut self.map.entries[i];
		(i, &*k, v)
	}

	pub fn insert_entry(self, value: V) -> (&'a K, &'a mut V)
	where
		K: Ord,
	{
		let (_, k, v) = self.insert_full(value);
		(k, v)
	}

	pub fn insert(self, value: V) -> &'a mut V
	where
		K: Ord,
	{
		self.insert_full(value).2
	}
}

pub struct OccupiedEntry<'a, K, V> {
	entries: &'a mut Vec<(K, V)>,
	indices: &'a Indexes,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
	pub fn key(&self) -> &K {
		&self.entries[self.indices.first].0
	}

	pub fn get_full(&self) -> GetIndexedEntries<'_, K, V> {
		GetIndexedEntries {
			indexes: self.indices.iter(),
			entries: &self.entries,
		}
	}

	pub fn get_entry(&self) -> GetEntries<'_, K, V> {
		GetEntries(self.get_full())
	}

	pub fn get(&self) -> Get<'_, K, V> {
		Get(self.get_full())
	}

	pub fn get_mut_full(&mut self) -> GetIndexedEntriesMut<'_, K, V> {
		GetIndexedEntriesMut {
			indexes: self.indices.iter(),
			entries: self.entries,
		}
	}

	pub fn get_mut_entry(&mut self) -> GetEntriesMut<'_, K, V> {
		GetEntriesMut(self.get_mut_full())
	}

	pub fn get_mut(&mut self) -> GetMut<'_, K, V> {
		GetMut(self.get_mut_full())
	}

	pub fn into_mut_full(self) -> GetIndexedEntriesMut<'a, K, V> {
		GetIndexedEntriesMut {
			indexes: self.indices.iter(),
			entries: self.entries,
		}
	}

	pub fn into_mut_entry(self) -> GetEntriesMut<'a, K, V> {
		GetEntriesMut(self.into_mut_full())
	}

	pub fn into_mut(self) -> GetMut<'a, K, V> {
		GetMut(self.into_mut_full())
	}

	pub fn insert(&mut self, value: V) -> V {
		std::mem::replace(&mut self.entries[self.indices.first].1, value)
	}
}
