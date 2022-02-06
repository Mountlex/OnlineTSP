use std::fmt::Debug;

use crate::{
    Adjacency, Cost, Cut, CutIter, Edge, Graph, GraphSize, Metric, Neighbors, Node, NodeIndex,
    NodeSet, Nodes, Weighted,
};

#[derive(Clone, Debug)]
pub struct MetricView<'a, M> {
    nodes: Vec<Node>,
    node_index: NodeIndex,
    metric: &'a M,
}

impl<'a, M> MetricView<'a, M>
where
    M: Metric,
{
    pub fn from_metric_on_nodes(nodes: Vec<Node>, metric: &'a M) -> Self {
        let node_index = NodeIndex::init(&nodes);
        Self {
            nodes,
            node_index,
            metric,
        }
    }

    pub fn add_nodes(&mut self, mut nodes: Vec<Node>) {
        self.nodes.append(&mut nodes);
        self.nodes.sort();
        self.nodes.dedup();
        self.node_index = NodeIndex::init(&self.nodes);
    }

    pub fn remove_node(&mut self, node: Node) {
        self.nodes.retain(|n| *n != node);
        self.node_index = NodeIndex::init(&self.nodes);
    }
}

impl<M> Metric for MetricView<'_, M>
where
    M: Metric,
{
    fn distance(&self, node1: Node, node2: Node) -> Cost {
        self.metric.distance(node1, node2)
    }
}

impl<M> Weighted for MetricView<'_, M>
where
    M: Metric + Clone + Debug,
{
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        Some(self.metric.distance(node1, node2))
    }
}

impl<'a, M> Graph<'a> for MetricView<'a, M>
where
    M: 'a + Metric + Clone + Debug,
{
    fn contains_node(&self, node: Node) -> bool {
        self.node_index.get(&node).is_some()
    }

    fn contains_edge(&self, node1: Node, node2: Node) -> bool {
        self.contains_node(node1) && self.contains_node(node2)
    }
}

pub struct AdjacencyIter<'a, M> {
    metric: &'a M,
    node: Node,
    adj_iter: std::slice::Iter<'a, Node>,
}

impl<'a, M> AdjacencyIter<'a, M> {
    fn new(metric: &'a M, node: Node, adj_iter: std::slice::Iter<'a, Node>) -> Self {
        Self {
            metric,
            node,
            adj_iter,
        }
    }
}

impl<'a, M> Iterator for AdjacencyIter<'a, M>
where
    M: Metric,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        let some_next = self.adj_iter.next();
        if let Some(next) = some_next {
            if *next == self.node {
                self.adj_iter
                    .next()
                    .map(|n| Edge::new(self.node, *n, self.metric.distance(self.node, *n)))
            } else {
                Some(Edge::new(
                    self.node,
                    *next,
                    self.metric.distance(self.node, *next),
                ))
            }
        } else {
            None
        }
    }
}

impl<'a, M> Adjacency<'a> for MetricView<'a, M>
where
    M: 'a + Metric,
{
    type AdjacencyIter = AdjacencyIter<'a, M>;

    fn adjacent(&'a self, node: Node) -> Self::AdjacencyIter {
        AdjacencyIter::new(&self.metric, node, self.nodes.iter())
    }
}

impl<'a, M> Cut<'a> for MetricView<'a, M>
where
    M: 'a + Metric,
{
    type CutIter = CutIter<'a, MetricView<'a, M>>;
    fn cut(&'a self, nodes: &NodeSet) -> Self::CutIter {
        CutIter::new(self, nodes)
    }
}

impl<M> GraphSize for MetricView<'_, M> {
    fn n(&self) -> usize {
        self.nodes.len()
    }
}

pub struct NeighborIter<'a> {
    adj_iter: std::slice::Iter<'a, Node>,
    node: Node,
}

impl<'a> NeighborIter<'a> {
    fn new(node: Node, adj_iter: std::slice::Iter<'a, Node>) -> Self {
        Self { node, adj_iter }
    }
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        let some_next = self.adj_iter.next();
        if let Some(next) = some_next {
            if *next == self.node {
                self.adj_iter.next().copied()
            } else {
                Some(*next)
            }
        } else {
            None
        }
    }
}

impl<'a, M> Neighbors<'a> for MetricView<'a, M> {
    type NeighborIter = NeighborIter<'a>;

    fn neighbors(&'a self, node: Node) -> Self::NeighborIter {
        NeighborIter::new(node, self.nodes.iter())
    }
}

impl<'a, M> Nodes<'a> for MetricView<'a, M> {
    type NodeIter = std::iter::Copied<std::slice::Iter<'a, Node>>;

    fn nodes(&'a self) -> Self::NodeIter {
        self.nodes.iter().copied()
    }
}
