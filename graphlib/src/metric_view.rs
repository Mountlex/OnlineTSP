use std::fmt::Debug;

use crate::{
    dijkstra::dijkstra_path,
    sp::{DistanceCache, ShortestPathsCache},
    AdjListGraph, Adjacency, Cost, Cut, CutIter, Edge, Graph, GraphSize, Metric, Neighbors, Node,
    NodeIndex, NodeSet, Nodes, Weighted,
};

#[derive(Clone, Debug)]
pub struct MetricView<'a, M> {
    nodes: Vec<Node>,
    node_index: NodeIndex,
    metric: &'a M,
    virtual_node: Option<Node>,
    buffer: Option<DistanceCache>,
}

impl<'a, M> MetricView<'a, M>
where
    M: Metric,
{
    pub fn from_metric_on_nodes(
        mut nodes: Vec<Node>,
        metric: &'a M,
        virtual_node: Option<Node>,
        buffer: Option<DistanceCache>,
    ) -> Self {
        if let Some(v_node) = virtual_node {
            if !nodes.contains(&v_node) {
                nodes.push(v_node);
            }
        }
        let node_index = NodeIndex::init(&nodes);
        Self {
            nodes,
            node_index,
            metric,
            virtual_node,
            buffer,
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

impl MetricView<'_, ShortestPathsCache> {
    pub fn split_virtual_edge(
        &self,
        virtual_source: Node,
        virtual_sink: Node,
        at: Cost,
        base_graph: &mut AdjListGraph,
    ) -> (Node, Option<DistanceCache>) {
        let (path_cost, path) = dijkstra_path(base_graph, virtual_source, virtual_sink);
        assert_eq!(self.distance(virtual_sink, virtual_source), path_cost);
        assert!(at > Cost::new(0));
        assert!(at < self.distance(virtual_sink, virtual_source));
        assert_eq!(path.first().copied(), Some(virtual_source));
        assert_eq!(path.last().copied(), Some(virtual_sink));

        let mut walked = Cost::new(0);

        for edge in path.windows(2) {
            let edge_cost = base_graph.edge_cost(edge[0], edge[1]).unwrap();
            if walked + edge_cost == at {
                // split is at base_graph node edge[1]
                //assert!(self.metric.contains_node(edge[1]));
                return (edge[1], None);
            } else if walked + edge_cost > at {
                let new_node = base_graph.split_edge_at(edge[0], edge[1], at - walked);

                if let Some(virtual_node) = self.virtual_node {
                    base_graph.remove_virtual_node(virtual_node);
                }

                let buffer = self.metric.split_edge_to_buffer(
                    edge[0],
                    edge[1],
                    at - walked,
                    edge_cost,
                    self.virtual_node,
                    self.buffer.clone(),
                );

                return (new_node, Some(buffer));
            }
            walked += edge_cost;
        }
        return (virtual_sink, None);
        panic!("Could not split edge!")
    }
}

impl<M> Metric for MetricView<'_, M>
where
    M: Metric,
{
    fn distance(&self, node1: Node, node2: Node) -> Cost {
        if Some(node1) == self.virtual_node {
            self.buffer.as_ref().unwrap().get(node2)
        } else if Some(node2) == self.virtual_node {
            self.buffer.as_ref().unwrap().get(node1)
        } else {
            self.metric.distance(node1, node2)
        }
    }
}

impl<M> Weighted for MetricView<'_, M>
where
    M: Metric + Clone + Debug,
{
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        Some(self.distance(node1, node2))
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
    metric: &'a MetricView<'a, M>,
    node: Node,
    adj_iter: std::slice::Iter<'a, Node>,
}

impl<'a, M> AdjacencyIter<'a, M> {
    fn new(metric: &'a MetricView<M>, node: Node, adj_iter: std::slice::Iter<'a, Node>) -> Self {
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
        AdjacencyIter::new(self, node, self.nodes.iter())
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
