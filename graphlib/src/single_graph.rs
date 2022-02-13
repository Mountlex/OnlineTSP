use std::fmt::Display;

use rustc_hash::FxHashMap;

use crate::{dijkstra::dijkstra, Cost, Weighted};

use super::{
    Adjacency, Cut, CutIter, Edge, Edges, ExpRoundable, Graph, GraphSize, Neighbors, Node, NodeSet,
    Nodes, ShortestPaths, TotalWeight,
};

/// Adjacency-list based graph representation for undirected weighted graphs.
#[derive(Debug, Clone)]
pub struct AdjListGraph {
    /// This object maps each node to its neighbors which corresponding edge costs.
    adj_list: FxHashMap<Node, FxHashMap<Node, Cost>>,
}

impl AdjListGraph {
    pub fn new() -> Self {
        AdjListGraph {
            adj_list: FxHashMap::default(),
        }
    }

    pub fn from_edges<'a>(edges: impl IntoIterator<Item = &'a Edge>) -> Self {
        let mut graph = Self::new();
        for edge in edges {
            graph.add_edge_raw(*edge);
        }
        graph
    }

    pub fn add_node(&mut self, node: Node) {
        if !self.adj_list.contains_key(&node) {
            self.adj_list.insert(node, FxHashMap::default());
        }
    }

    pub fn add_edge(&mut self, source: Node, sink: Node, cost: Cost) {
        self.add_edge_raw(Edge::new(source, sink, cost));
    }

    pub fn remove_virtual_node(
        &mut self,
        virtual_node: Node,
    ) {
        let neighbors: Vec<Node> = self.neighbors(virtual_node).collect();
        println!("{:?}", neighbors);
        assert_eq!(neighbors.len(), 2);
        let edge_cost = self.edge_cost(neighbors[0], virtual_node).unwrap() + self.edge_cost(neighbors[1], virtual_node).unwrap();
        for &neighbor in &neighbors {
            self.remove_edge(virtual_node, neighbor);
        }
        self.add_edge(neighbors[0], neighbors[1], edge_cost);
        self.adj_list.remove(&virtual_node);
    }

    pub fn split_edge_at(&mut self, new_node: Node, source: Node, sink: Node, at: Cost) -> Node {
        let cost = self.edge_cost(source, sink).unwrap();
        assert!(Cost::new(0) < at);
        assert!(cost > at);

        self.add_node(new_node);
        self.remove_edge(source, sink);
        self.add_edge(source, new_node, at);
        self.add_edge(new_node, sink, cost - at);

        return new_node;
    }

    fn add_edge_raw(&mut self, edge: Edge) {
        let adj_source = self.adj_list.entry(edge.source()).or_default();
        adj_source.insert(edge.sink(), edge.cost());

        let adj_sink = self.adj_list.entry(edge.sink()).or_default();
        adj_sink.insert(edge.source(), edge.cost());
    }

    pub fn remove_edge(&mut self, source: Node, sink: Node) {
        let adj_source = self.adj_list.entry(source).or_default();
        adj_source.remove(&sink);

        let adj_sink = self.adj_list.entry(sink).or_default();
        adj_sink.remove(&source);
    }

    pub fn m(&self) -> usize {
        self.adj_list
            .values()
            .map(|edges| edges.len())
            .sum::<usize>()
            / 2
    }

    pub fn degree(&self) -> usize {
        self.adj_list
            .values()
            .map(|edges| edges.len())
            .max()
            .unwrap_or_default()
    }
}

impl Display for AdjListGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "graph {{")?;
        for n1 in self.nodes() {
            for (n2, _) in self.adj_list.get(&n1).unwrap() {
                if n1 < *n2 {
                    writeln!(f, "{} -- {}", n1, n2)?;
                }
            }
        }
        writeln!(f, "}}")
    }
}

impl ShortestPaths for AdjListGraph {
    fn shortest_path_cost(&self, n1: Node, n2: Node) -> Cost {
        dijkstra(self, n1, n2)
    }
}

impl<'a> Cut<'a> for AdjListGraph {
    type CutIter = CutIter<'a, AdjListGraph>;
    fn cut(&'a self, nodes: &NodeSet) -> Self::CutIter {
        CutIter::new(self, nodes)
    }
}

impl GraphSize for AdjListGraph {
    fn n(&self) -> usize {
        self.adj_list.len()
    }
}

pub struct AdjacencyIter<'a> {
    node: Node,
    adj_iter: Option<std::collections::hash_map::Iter<'a, Node, Cost>>,
}

impl<'a> AdjacencyIter<'a> {
    fn new(node: Node, adj_iter: Option<std::collections::hash_map::Iter<'a, Node, Cost>>) -> Self {
        Self { node, adj_iter }
    }
}

impl<'a> Iterator for AdjacencyIter<'a> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.node;
        self.adj_iter
            .as_mut()
            .and_then(|values| values.next().map(|(n, c)| Edge::new(node, *n, *c)))
    }
}

impl<'a> Adjacency<'a> for AdjListGraph {
    type AdjacencyIter = AdjacencyIter<'a>;

    fn adjacent(&'a self, node: Node) -> Self::AdjacencyIter {
        AdjacencyIter::new(node, self.adj_list.get(&node).map(|m| m.iter()))
    }
}

pub struct NeighborIter<'a> {
    adj_iter: Option<std::collections::hash_map::Keys<'a, Node, Cost>>,
}

impl<'a> NeighborIter<'a> {
    fn new(adj_iter: Option<std::collections::hash_map::Keys<'a, Node, Cost>>) -> Self {
        Self { adj_iter }
    }
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.adj_iter
            .as_mut()
            .and_then(|keys| keys.next().map(|n| *n))
    }
}

impl<'a> Neighbors<'a> for AdjListGraph {
    type NeighborIter = NeighborIter<'a>;

    fn neighbors(&'a self, node: Node) -> Self::NeighborIter {
        NeighborIter::new(self.adj_list.get(&node).map(|m| m.keys()))
    }
}

impl<'a> Nodes<'a> for AdjListGraph {
    type NodeIter =
        std::iter::Copied<std::collections::hash_map::Keys<'a, Node, FxHashMap<Node, Cost>>>;

    fn nodes(&'a self) -> Self::NodeIter {
        self.adj_list.keys().copied()
    }
}

pub struct EdgeIter<'a> {
    base_iter: std::collections::hash_map::Iter<'a, Node, FxHashMap<Node, Cost>>,
    sink_iter: Option<(Node, std::collections::hash_map::Iter<'a, Node, Cost>)>,
}

impl<'a> Iterator for EdgeIter<'a> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((source, sink_iter)) = &mut self.sink_iter {
                if let Some((sink, cost)) = sink_iter.next() {
                    if *source < *sink {
                        return Some(Edge::new(*source, *sink, *cost));
                    } else {
                        continue;
                    }
                }
            }
            // sink_iter empty or not initialized
            if let Some((node, map)) = self.base_iter.next() {
                self.sink_iter = Some((*node, map.iter()))
            } else {
                break;
            }
        }
        None
    }
}

impl<'a> Edges<'a> for AdjListGraph {
    type EdgeIter = EdgeIter<'a>;
    fn edges(&'a self) -> Self::EdgeIter {
        EdgeIter {
            base_iter: self.adj_list.iter(),
            sink_iter: None,
        }
    }
}

impl Weighted for AdjListGraph {
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        self.adj_list
            .get(&node1)
            .and_then(|edges| edges.get(&node2).copied())
    }
}

impl<'a> Graph<'a> for AdjListGraph {
    fn contains_node(&self, node: Node) -> bool {
        self.adj_list.contains_key(&node)
    }

    fn contains_edge(&self, node1: Node, node2: Node) -> bool {
        self.adj_list
            .get(&node1)
            .and_then(|edges| edges.get(&node2))
            .is_some()
    }
}

impl TotalWeight for AdjListGraph {
    fn total_weight(&self) -> Cost {
        self.adj_list
            .values()
            .map(|n| n.values().sum::<Cost>())
            .sum::<Cost>()
            / 2.0
    }
}

impl ExpRoundable for AdjListGraph {
    fn to_exp_rounded(&self) -> AdjListGraph {
        let mut r_graph = AdjListGraph::new();
        for node in self.nodes() {
            for edge in self.adjacent(node) {
                if edge.source() < edge.sink() {
                    r_graph.add_edge(edge.source(), edge.sink(), edge.cost().to_exp_rounded())
                }
            }
        }
        r_graph
    }
}

#[cfg(test)]
mod test_adj_list_graph {
    use super::*;

    #[test]
    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn test_edge_iter() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        assert_eq!(graph.edges().count(), 7);
    }

    #[test]
    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn test_split() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        assert_eq!(graph.edges().count(), 7);
        assert_eq!(graph.nodes().count(), 6);

        graph.split_edge_at(Node::new(7),1.into(), 2.into(), 2.into());

        assert_eq!(graph.edges().count(), 8);
        assert_eq!(graph.nodes().count(), 7);

        assert_eq!(graph.edge_cost(2.into(), 7.into()).unwrap(), Cost::new(3));
    }
}
