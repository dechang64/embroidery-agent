//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! From organoid-fl — our own HNSW implementation with:
//!   - Custom Euclidean distance metric (f32 → u32 via to_bits)
//!   - Const generic M/M0 parameters
//!   - Bidirectional ID mapping (string ↔ internal int)
//!   - Reusable searcher across queries
//!
//! Used for embroidery pattern style fingerprint search (768-dim DINOv2 features).

use std::collections::HashMap;

/// Euclidean distance metric for f32 vectors.
/// Converts float distances to u32 via `to_bits()` for compatibility with `space::Metric`.
#[derive(Clone, Debug, Default)]
pub struct Euclidean;

impl space::Metric<Vec<f32>> for Euclidean {
    type Unit = u32;

    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u32 {
        let mut sum = 0.0f32;
        let len = a.len().min(b.len());
        for i in 0..len {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum.sqrt().to_bits()
    }
}

/// HNSW index wrapper that maps between internal integer IDs and our string IDs.
///
/// Uses const generic parameters:
/// - `M`: max connections per layer (default 16)
/// - `M0`: max connections at layer 0 (default 32)
pub struct HnswIndex<const M: usize = 16, const M0: usize = 32> {
    dimension: usize,
    /// Maps internal HNSW integer IDs → our string IDs
    id_to_string: HashMap<usize, String>,
    /// Maps our string IDs → internal HNSW integer IDs
    string_to_id: HashMap<String, usize>,
    /// The HNSW graph structure
    graph: hnsw::Hnsw<Euclidean, Vec<f32>, rand::rngs::StdRng, M, M0>,
    /// Searcher reused across queries
    searcher: hnsw::Searcher<u32>,
    /// Next available internal ID
    next_internal_id: usize,
}

impl<const M: usize, const M0: usize> HnswIndex<M, M0> {
    /// Create a new HNSW index.
    pub fn new(dimension: usize) -> Self {
        let graph = hnsw::Hnsw::new(Euclidean);
        Self {
            dimension,
            id_to_string: HashMap::new(),
            string_to_id: HashMap::new(),
            graph,
            searcher: hnsw::Searcher::new(),
            next_internal_id: 0,
        }
    }

    /// Insert a vector with the given string ID.
    /// Returns the internal HNSW index assigned to this vector.
    pub fn insert(&mut self, id: &str, values: &[f32]) -> usize {
        let internal_id = self.next_internal_id;
        self.next_internal_id += 1;

        self.id_to_string.insert(internal_id, id.to_string());
        self.string_to_id.insert(id.to_string(), internal_id);

        let _stored_idx = self.graph.insert(values.to_vec(), &mut self.searcher);

        internal_id
    }

    /// Search for k nearest neighbors of the query vector.
    /// Returns a list of (string_id, distance) sorted by distance.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Vec<(String, f32)> {
        if self.id_to_string.is_empty() {
            return Vec::new();
        }

        let n = self.id_to_string.len();
        let actual_k = k.min(n);
        let buf_size = ef.min(n);

        let mut neighbors = vec![space::Neighbor { index: 0, distance: u32::MAX }; buf_size];

        let results = self.graph.nearest(
            &query.to_vec(),
            buf_size,
            &mut self.searcher,
            &mut neighbors,
        );

        results[..actual_k.min(results.len())]
            .iter()
            .filter_map(|neighbor| {
                self.id_to_string.get(&neighbor.index).map(|id| {
                    let distance = f32::from_bits(neighbor.distance);
                    (id.clone(), distance)
                })
            })
            .collect()
    }

    /// Remove a vector by string ID (soft delete).
    pub fn remove(&mut self, id: &str) {
        if let Some(internal_id) = self.string_to_id.remove(id) {
            self.id_to_string.remove(&internal_id);
        }
    }

    /// Number of active vectors in the index.
    pub fn len(&self) -> usize {
        self.string_to_id.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.string_to_id.is_empty()
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of layers in the HNSW graph.
    pub fn layers(&self) -> usize {
        self.graph.layers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut index = HnswIndex::<16, 32>::new(3);

        index.insert("v1", &[1.0, 0.0, 0.0]);
        index.insert("v2", &[0.0, 1.0, 0.0]);
        index.insert("v3", &[0.9, 0.1, 0.0]);
        index.insert("v4", &[0.0, 0.0, 1.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 2, 50);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "v1");
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_hnsw_remove() {
        let mut index = HnswIndex::<16, 32>::new(3);

        index.insert("v1", &[1.0, 0.0, 0.0]);
        index.insert("v2", &[0.0, 1.0, 0.0]);

        assert_eq!(index.len(), 2);
        index.remove("v1");
        assert_eq!(index.len(), 1);

        let results = index.search(&[1.0, 0.0, 0.0], 5, 50);
        assert!(!results.iter().any(|(id, _)| id == "v1"));
    }

    #[test]
    fn test_hnsw_empty_search() {
        let mut index = HnswIndex::<16, 32>::new(3);
        let results = index.search(&[1.0, 0.0, 0.0], 5, 50);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_layers() {
        let mut index = HnswIndex::<16, 32>::new(3);
        for i in 0..100 {
            let v = [(i as f32) / 100.0; 3];
            index.insert(&format!("v{}", i), &v);
        }
        assert!(index.layers() >= 2);
    }
}
