

use rustc_hash::FxHashMap;

use crate::{Cost, Weighted};

use super::{Adjacency, Cut, CutIter, Edge, Graph, GraphSize, Neighbors, Node, NodeSet, Nodes};

#[derive(Clone, Debug)]
pub struct MultiGraph {
    adj_list: FxHashMap<Node, Vec<Edge>>,
}

impl MultiGraph {
    pub fn new() -> Self {
        MultiGraph {
            adj_list: FxHashMap::default(),
        }
    }

    pub fn singleton(node: Node) -> Self {
        let mut adj = FxHashMap::<Node, Vec<Edge>>::default();
        adj.insert(node, vec![]);
        MultiGraph { adj_list: adj }
    }

    pub fn add_edge(&mut self, source: Node, sink: Node, cost: Cost) {
        self.add_edge_raw(Edge::new(source, sink, cost));
    }

    pub fn add_edge_raw(&mut self, edge: Edge) {
        let adj_source = self.adj_list.entry(edge.source()).or_default();
        adj_source.push(edge);

        let adj_sink = self.adj_list.entry(edge.sink()).or_default();
        adj_sink.push(edge.reversed());
    }

    pub fn remove_edge(&mut self, source: Node, sink: Node, cost: Cost) {
        self.remove_edge_raw(Edge::new(source, sink, cost));
    }

    pub fn remove_edge_raw(&mut self, edge: Edge) {
        let adj_source = self.adj_list.entry(edge.source()).or_default();
        if let Some(pos) = adj_source.iter().position(|e| *e == edge) {
            adj_source.remove(pos);
        }

        let adj_sink = self.adj_list.entry(edge.sink()).or_default();
        if let Some(pos) = adj_sink.iter().position(|e| *e == edge.reversed()) {
            adj_sink.remove(pos);
        }
    }

    pub fn m(&self) -> usize {
        self.adj_list
            .values()
            .map(|edges| edges.len())
            .sum::<usize>()
            / 2
    }

    pub fn from_graph<'a, G>(graph: &'a G) -> Self
    where
        G: Nodes<'a> + Adjacency<'a>,
    {
        let adj_list = graph
            .nodes()
            .into_iter()
            .map(|n1| {
                let adj = graph.adjacent(n1).collect::<Vec<Edge>>();
                (n1, adj)
            })
            .collect::<FxHashMap<Node, Vec<Edge>>>();
        Self { adj_list }
    }

    pub fn node_iter(&self) -> impl Iterator<Item = &Node> {
        self.adj_list.keys()
    }
}

impl<'a> Cut<'a> for MultiGraph {
    type CutIter = CutIter<'a, MultiGraph>;
    fn cut(&'a self, nodes: &NodeSet) -> Self::CutIter {
        CutIter::new(self, nodes)
    }
}

impl GraphSize for MultiGraph {
    fn n(&self) -> usize {
        self.adj_list.len()
    }
}

pub struct AdjacencyIter<'a> {
    adj_iter: Option<std::slice::Iter<'a, Edge>>,
}

impl<'a> AdjacencyIter<'a> {
    fn new(adj_iter: Option<std::slice::Iter<'a, Edge>>) -> Self {
        Self { adj_iter }
    }
}

impl<'a> Iterator for AdjacencyIter<'a> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.adj_iter
            .as_mut()
            .and_then(|edges| edges.next().copied())
    }
}

impl<'a> Adjacency<'a> for MultiGraph {
    type AdjacencyIter = AdjacencyIter<'a>;

    fn adjacent(&'a self, node: Node) -> Self::AdjacencyIter {
        AdjacencyIter::new(self.adj_list.get(&node).map(|m| m.iter()))
    }
}

pub struct NeighborIter<'a> {
    adj_iter: Option<std::slice::Iter<'a, Edge>>,
}

impl<'a> NeighborIter<'a> {
    fn new(adj_iter: Option<std::slice::Iter<'a, Edge>>) -> Self {
        Self { adj_iter }
    }
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.adj_iter
            .as_mut()
            .and_then(|iter| iter.next().map(|e| e.sink()))
    }
}

impl<'a> Neighbors<'a> for MultiGraph {
    type NeighborIter = NeighborIter<'a>;

    fn neighbors(&'a self, node: Node) -> Self::NeighborIter {
        NeighborIter::new(self.adj_list.get(&node).map(|m| m.iter()))
    }
}

impl<'a> Nodes<'a> for MultiGraph {
    type NodeIter = std::iter::Copied<std::collections::hash_map::Keys<'a, Node, Vec<Edge>>>;

    fn nodes(&'a self) -> Self::NodeIter {
        self.adj_list.keys().copied()
    }
}

impl Weighted for MultiGraph {
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        self.adj_list.get(&node1).and_then(|edges| {
            edges
                .into_iter()
                .find(|e| e.sink() == node2)
                .map(|e| e.cost())
        })
    }
}

impl<'a> Graph<'a> for MultiGraph {
    fn contains_node(&self, node: Node) -> bool {
        self.adj_list.contains_key(&node)
    }

    fn contains_edge(&self, node1: Node, node2: Node) -> bool {
        self.adj_list
            .get(&node1)
            .and_then(|edges| edges.into_iter().find(|e| e.sink() == node2))
            .is_some()
    }
}
