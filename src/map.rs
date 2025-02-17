use core::fmt;
use raw_btree::RawBTree;
use std::{
	cmp::Ordering,
	hash::{Hash, Hasher},
	iter::FusedIterator,
	mem::MaybeUninit,
	ops::Index,
};

pub use equivalent::Comparable;

#[derive(Clone)]
pub struct BTreeIndexMap<K, V> {
	entries: Vec<(K, V)>,
	map: RawBTree<usize>,
}

impl<K, V> Default for BTreeIndexMap<K, V> {
	fn default() -> Self {
		Self::new()
	}
}

impl<K, V> BTreeIndexMap<K, V> {
	pub fn new() -> Self {
		Self {
			entries: Vec::new(),
			map: RawBTree::new(),
		}
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

	pub fn get_full<Q>(&self, key: &Q) -> Option<(usize, &K, &V)>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.map
			.get(outer_cmp(&self.entries), key)
			.copied()
			.map(|i| {
				let (k, v) = &self.entries[i];
				(i, k, v)
			})
	}

	pub fn get_entry<Q>(&self, key: &Q) -> Option<(&K, &V)>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.get_full(key).map(|(_, k, v)| (k, v))
	}

	pub fn get<Q>(&self, key: &Q) -> Option<&V>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.get_full(key).map(|(_, _, v)| v)
	}

	pub fn get_mut_full<Q>(&mut self, key: &Q) -> Option<(usize, &K, &mut V)>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.map
			.get(outer_cmp(&self.entries), key)
			.copied()
			.map(|i| {
				let (k, v) = &mut self.entries[i];
				(i, &*k, v)
			})
	}

	pub fn get_mut_entry<Q>(&mut self, key: &Q) -> Option<(&K, &mut V)>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.get_mut_full(key).map(|(_, k, v)| (k, v))
	}

	pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.get_mut_full(key).map(|(_, _, v)| v)
	}

	pub fn entry(&mut self, key: K) -> Entry<K, V>
	where
		K: Ord,
	{
		match self.map.get(outer_cmp(&self.entries), &key).copied() {
			Some(i) => Entry::Occupied(OccupiedEntry { map: self, i }),
			None => Entry::Vacant(VacantEntry { map: self, key }),
		}
	}

	pub fn insert_full(&mut self, key: K, value: V) -> (usize, Option<V>)
	where
		K: Ord,
	{
		match self.map.get(outer_cmp(&self.entries), &key).copied() {
			Some(i) => (i, Some(std::mem::replace(&mut self.entries[i].1, value))),
			None => {
				let i = self.force_insert_full(key, value);
				(i, None)
			}
		}
	}

	fn force_insert_full(&mut self, key: K, value: V) -> usize
	where
		K: Ord,
	{
		let i = self.entries.len();
		self.entries.push((key, value));
		self.map.insert(inner_cmp(&self.entries), i);
		i
	}

	pub fn insert(&mut self, key: K, value: V) -> Option<V>
	where
		K: Ord,
	{
		self.insert_full(key, value).1
	}

	pub fn shift_remove_full<Q>(&mut self, key: &Q) -> Option<(usize, K, V)>
	where
		Q: ?Sized + Comparable<K>,
	{
		let i = self.map.remove(outer_cmp(&self.entries), key)?;

		let (k, v) = self.entries.remove(i);

		// Shift larger indices.
		for j in self.map.iter_mut() {
			if *j > i {
				*j -= 1;
			}
		}

		Some((i, k, v))
	}

	pub fn shift_remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.shift_remove_full(key).map(|(_, k, v)| (k, v))
	}

	pub fn shift_remove<Q>(&mut self, key: &Q) -> Option<V>
	where
		Q: ?Sized + Comparable<K>,
	{
		self.shift_remove_full(key).map(|(_, _, v)| v)
	}

	pub fn swap_remove_full<Q>(&mut self, key: &Q) -> Option<(usize, K, V)>
	where
		Q: ?Sized + Comparable<K>,
		K: Ord,
	{
		let i = self.map.remove(outer_cmp(&self.entries), key)?;

		let j = self.entries.len() - 1;

		if j > i {
			// Remove the last entry's index from the map.
			self.map.remove(inner_cmp(&self.entries), &j);
		}

		let (k, v) = self.entries.swap_remove(i);
		Some((i, k, v))
	}

	pub fn swap_remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
	where
		Q: ?Sized + Comparable<K>,
		K: Ord,
	{
		self.swap_remove_full(key).map(|(_, k, v)| (k, v))
	}

	pub fn swap_remove<Q>(&mut self, key: &Q) -> Option<V>
	where
		Q: ?Sized + Comparable<K>,
		K: Ord,
	{
		self.swap_remove_full(key).map(|(_, _, v)| v)
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
		}
	}

	pub fn iter_mut_sorted(&mut self) -> IterMutSorted<K, V> {
		IterMutSorted {
			entries: &mut self.entries,
			inner: self.map.iter(),
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
}

fn inner_cmp<K, V>(entries: &[(K, V)]) -> impl use<'_, K, V> + Fn(&usize, &usize) -> Ordering
where
	K: Ord,
{
	|&i, &j| entries[i].0.cmp(&entries[j].0)
}

fn outer_cmp<K, V, Q>(entries: &[(K, V)]) -> impl use<'_, K, V, Q> + Fn(&usize, &Q) -> Ordering
where
	Q: ?Sized + Comparable<K>,
{
	|&i, q| q.compare(&entries[i].0).reverse()
}

impl<K, V, Q> Index<&Q> for BTreeIndexMap<K, V>
where
	Q: ?Sized + Comparable<K>,
{
	type Output = V;

	fn index(&self, key: &Q) -> &Self::Output {
		self.get(key).expect("no entry found for key")
	}
}

impl<K, V> IntoIterator for BTreeIndexMap<K, V> {
	type IntoIter = std::vec::IntoIter<(K, V)>;
	type Item = (K, V);

	fn into_iter(self) -> Self::IntoIter {
		self.entries.into_iter()
	}
}

impl<K: PartialEq, V: PartialEq> PartialEq for BTreeIndexMap<K, V> {
	fn eq(&self, other: &Self) -> bool {
		self.iter_sorted().eq(other.iter_sorted())
	}
}

impl<K: Eq, V: Eq> Eq for BTreeIndexMap<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for BTreeIndexMap<K, V> {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		self.iter_sorted().partial_cmp(other.iter_sorted())
	}
}

impl<K: Ord, V: Ord> Ord for BTreeIndexMap<K, V> {
	fn cmp(&self, other: &Self) -> Ordering {
		self.iter_sorted().cmp(other.iter_sorted())
	}
}

impl<K: Hash, V: Hash> Hash for BTreeIndexMap<K, V> {
	fn hash<H: Hasher>(&self, state: &mut H) {
		state.write_usize(self.len());
		for item in self.iter_sorted() {
			item.hash(state);
		}
	}
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for BTreeIndexMap<K, V> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_set().entries(self.iter()).finish()
	}
}

impl<K: Ord, V> Extend<(K, V)> for BTreeIndexMap<K, V> {
	fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
		for (key, value) in iter {
			self.insert(key, value);
		}
	}
}

impl<K: Ord, V> FromIterator<(K, V)> for BTreeIndexMap<K, V> {
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

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(k, v)| (k, v))
	}
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

impl<'a, K, V> IntoIterator for &'a BTreeIndexMap<K, V> {
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

impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(k, v)| (&*k, v))
	}
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {}

impl<'a, K, V> FusedIterator for IterMut<'a, K, V> {}

impl<'a, K, V> IntoIterator for &'a mut BTreeIndexMap<K, V> {
	type IntoIter = IterMut<'a, K, V>;
	type Item = (&'a K, &'a mut V);

	fn into_iter(self) -> Self::IntoIter {
		self.iter_mut()
	}
}

pub struct IterSorted<'a, K, V> {
	entries: &'a [(K, V)],
	inner: raw_btree::Iter<'a, usize>,
}

impl<'a, K, V> Iterator for IterSorted<'a, K, V> {
	type Item = (&'a K, &'a V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.inner.next().copied().map(|i| {
			let (k, v) = &self.entries[i];
			(k, v)
		})
	}
}

impl<'a, K, V> DoubleEndedIterator for IterSorted<'a, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.inner.next_back().copied().map(|i| {
			let (k, v) = &self.entries[i];
			(k, v)
		})
	}
}

impl<'a, K, V> ExactSizeIterator for IterSorted<'a, K, V> {}

impl<'a, K, V> FusedIterator for IterSorted<'a, K, V> {}

pub struct IterMutSorted<'a, K, V> {
	entries: &'a mut [(K, V)],
	inner: raw_btree::Iter<'a, usize>,
}

impl<'a, K, V> Iterator for IterMutSorted<'a, K, V> {
	type Item = (&'a K, &'a mut V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.inner.next().copied().map(|i| {
			let (k, v) = &mut self.entries[i];
			(
				unsafe {
					// SAFETY: `k` is cannot be borrowed again as long as the
					//         iterator lives.
					std::mem::transmute::<&mut K, &'a K>(k)
				},
				unsafe {
					// SAFETY: `v` is cannot be borrowed again as long as the
					//         iterator lives.
					std::mem::transmute::<&mut V, &'a mut V>(v)
				},
			)
		})
	}
}

impl<'a, K, V> DoubleEndedIterator for IterMutSorted<'a, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.inner.next_back().copied().map(|i| {
			let (k, v) = &mut self.entries[i];
			(
				unsafe {
					// SAFETY: `k` is cannot be borrowed again as long as the
					//         iterator lives.
					std::mem::transmute::<&mut K, &'a K>(k)
				},
				unsafe {
					// SAFETY: `v` is cannot be borrowed again as long as the
					//         iterator lives.
					std::mem::transmute::<&mut V, &'a mut V>(v)
				},
			)
		})
	}
}

impl<'a, K, V> ExactSizeIterator for IterMutSorted<'a, K, V> {}

impl<'a, K, V> FusedIterator for IterMutSorted<'a, K, V> {}

pub struct IntoIterSorted<K, V> {
	entries: Vec<MaybeUninit<(K, V)>>,
	inner: raw_btree::IntoIter<usize>,
}

impl<K, V> Iterator for IntoIterSorted<K, V> {
	type Item = (K, V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	fn next(&mut self) -> Option<Self::Item> {
		self.inner.next().map(|i| {
			unsafe {
				// SAFETY: each index is visited exactly once, so there is no
				//         risk of reading the same entry twice.
				self.entries[i].assume_init_read()
			}
		})
	}
}

impl<K, V> DoubleEndedIterator for IntoIterSorted<K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.inner.next_back().map(|i| {
			unsafe {
				// SAFETY: each index is visited exactly once, so there is no
				//         risk of reading the same entry twice.
				self.entries[i].assume_init_read()
			}
		})
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

impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(k, _)| k)
	}
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {}

impl<'a, K, V> FusedIterator for Keys<'a, K, V> {}

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

impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(_, v)| v)
	}
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {}

impl<'a, K, V> FusedIterator for Values<'a, K, V> {}

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

impl<'a, K, V> DoubleEndedIterator for ValuesMut<'a, K, V> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.0.next_back().map(|(_, v)| v)
	}
}

impl<'a, K, V> ExactSizeIterator for ValuesMut<'a, K, V> {}

impl<'a, K, V> FusedIterator for ValuesMut<'a, K, V> {}

pub enum Entry<'a, K, V> {
	Vacant(VacantEntry<'a, K, V>),
	Occupied(OccupiedEntry<'a, K, V>),
}

pub struct VacantEntry<'a, K, V> {
	map: &'a mut BTreeIndexMap<K, V>,
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
		let i = self.map.force_insert_full(self.key, value);
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
	map: &'a mut BTreeIndexMap<K, V>,
	i: usize,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
	pub fn key(&self) -> &K {
		&self.map.entries[self.i].0
	}

	pub fn get_full(&self) -> (usize, &K, &V) {
		let (k, v) = &self.map.entries[self.i];
		(self.i, k, v)
	}

	pub fn get_entry(&self) -> (&K, &V) {
		let (k, v) = &self.map.entries[self.i];
		(k, v)
	}

	pub fn get(&self) -> &V {
		&self.map.entries[self.i].1
	}

	pub fn get_mut_full(&mut self) -> (usize, &K, &mut V) {
		let (k, v) = &mut self.map.entries[self.i];
		(self.i, &*k, v)
	}

	pub fn get_mut_entry(&mut self) -> (&K, &mut V) {
		let (k, v) = &mut self.map.entries[self.i];
		(&*k, v)
	}

	pub fn get_mut(&mut self) -> &mut V {
		&mut self.map.entries[self.i].1
	}

	pub fn into_mut_full(self) -> (usize, &'a K, &'a mut V) {
		let (k, v) = &mut self.map.entries[self.i];
		(self.i, &*k, v)
	}

	pub fn into_mut_entry(self) -> (&'a K, &'a mut V) {
		let (k, v) = &mut self.map.entries[self.i];
		(&*k, v)
	}

	pub fn into_mut(self) -> &'a mut V {
		&mut self.map.entries[self.i].1
	}

	pub fn insert(&mut self, value: V) -> V {
		std::mem::replace(&mut self.map.entries[self.i].1, value)
	}
}
