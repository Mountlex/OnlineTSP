use std::{collections::HashSet, fmt::Debug, iter::FromIterator};

use dyn_clone::DynClone;

use crate::Cost;

use super::{Node, NodeSet};

/// An edge in a graph.
#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct Edge {
    cost: Cost,
    source: Node,
    sink: Node,
}

impl Edge {
    pub fn new(source: Node, sink: Node, cost: Cost) -> Self {
        Edge { source, sink, cost }
    }

    pub fn reversed(&self) -> Edge {
        Edge::new(self.sink, self.source, self.cost)
    }

    pub fn cost(&self) -> Cost {
        self.cost
    }

    pub fn source(&self) -> Node {
        self.source
    }

    pub fn sink(&self) -> Node {
        self.sink
    }

    pub fn neighbor(&self, other: Node) -> Option<Node> {
        if other == self.sink() {
            Some(self.source())
        } else if other == self.source() {
            Some(self.sink())
        } else {
            None
        }
    }
}

/// A set of edges.
#[derive(Clone, Debug, Default)]
pub struct EdgeSet(HashSet<Edge>);

impl EdgeSet {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn empty() -> Self {
        Self(HashSet::new())
    }

    pub fn sink_nodeset(&self) -> NodeSet {
        self.0.iter().map(|e| e.sink()).collect()
    }

    pub fn insert(&mut self, e: Edge) -> bool {
        self.0.insert(e)
    }

    pub fn contains(&self, e: &Edge) -> bool {
        self.0.contains(e)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn apply(&mut self, edge_cond: Box<dyn EdgeCond>) {
        let new_set = self
            .0
            .iter()
            .filter(|e| (*edge_cond).check(e))
            .copied()
            .collect();
        self.0 = new_set;
    }

    pub fn diff(&self, other: &EdgeSet) -> EdgeSet {
        self.0.difference(&other.0).copied().collect()
    }

    pub fn contains_sink(&self, sink: &Node) -> bool {
        self.get_edge_to_sink(sink).is_some()
    }

    pub fn get_edge_to_sink(&self, sink: &Node) -> Option<&Edge> {
        for edge in self.0.iter() {
            if edge.sink() == *sink {
                return Some(edge);
            }
        }
        None
    }

    pub fn to_sink_sorted_vec(self) -> Vec<Edge> {
        let mut vec: Vec<Edge> = self.into_iter().collect();
        vec.sort_by_key(|e| e.sink());
        vec
    }
}

impl From<HashSet<Edge>> for EdgeSet {
    fn from(set: HashSet<Edge>) -> Self {
        EdgeSet(set)
    }
}

impl FromIterator<Edge> for EdgeSet {
    fn from_iter<T: IntoIterator<Item = Edge>>(iter: T) -> Self {
        EdgeSet(iter.into_iter().collect())
    }
}

impl IntoIterator for EdgeSet {
    type Item = Edge;
    type IntoIter = std::collections::hash_set::IntoIter<Edge>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a EdgeSet {
    type Item = &'a Edge;
    type IntoIter = std::collections::hash_set::Iter<'a, Edge>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

dyn_clone::clone_trait_object!(EdgeCond);

pub trait EdgeCond: DynClone + Debug {
    fn check(&self, edge: &Edge) -> bool;
}

impl<F> EdgeCond for F
where
    F: Fn(&Edge) -> bool + Clone + Debug,
{
    fn check(&self, x: &Edge) -> bool {
        self(x)
    }
}
