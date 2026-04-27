use anyhow::Result;
use crate::hnsw_index::HnswIndex;
use std::collections::HashMap;

/// 向量数据库 — 管理刺绣图案特征向量和风格嵌入
/// Uses organoid-fl HNSW with const generic M/M0 and custom Euclidean metric.
pub struct VectorDb {
    index: HnswIndex<16, 32>,
    vectors: HashMap<String, Vec<f32>>,
    metadata: HashMap<String, HashMap<String, String>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub meta: HashMap<String, String>,
}

impl VectorDb {
    pub fn new(dimension: usize) -> Self {
        Self {
            index: HnswIndex::new(dimension),
            vectors: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: &str, vector: &[f32], meta: Option<HashMap<String, String>>) -> Result<()> {
        self.index.insert(id, vector);
        self.vectors.insert(id.to_string(), vector.to_vec());
        if let Some(m) = meta {
            self.metadata.insert(id.to_string(), m);
        }
        Ok(())
    }

    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let ef = std::cmp::max(k * 4, 50);
        let raw = self.index.search(query, k, ef);
        Ok(raw.into_iter().map(|(id, distance)| {
            let meta = self.metadata.get(&id).cloned().unwrap_or_default();
            SearchResult { id, distance, meta }
        }).collect())
    }

    pub fn remove(&mut self, id: &str) {
        self.index.remove(id);
        self.vectors.remove(id);
        self.metadata.remove(id);
    }

    pub fn get(&self, id: &str) -> Option<&Vec<f32>> { self.vectors.get(id) }
    pub fn len(&self) -> usize { self.vectors.len() }
    pub fn is_empty(&self) -> bool { self.vectors.is_empty() }
}
