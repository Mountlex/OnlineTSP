use std::fmt::Debug;

use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    dijkstra::dijkstra_path, sp::ShortestPathsCache, AdjListGraph, Adjacency, Cost, Cut, CutIter,
    Edge, Graph, GraphSize, Neighbors, Node, NodeIndex, NodeSet, Nodes, ShortestPaths, Weighted,
};

pub trait Metric {
    fn distance(&self, node1: Node, node2: Node) -> Cost;
}

impl<SP> Metric for SP
where
    SP: ShortestPaths,
{
    fn distance(&self, node1: Node, node2: Node) -> Cost {
        self.shortest_path_cost(node1, node2)
    }
}

pub trait MetricCompletion<M> {
    fn compute_metric_completion(self) -> MetricGraph<M>;
}

impl<'a, G> MetricCompletion<ShortestPathsCache> for &'a G
where
    G: Graph<'a> + Sync,
{
    fn compute_metric_completion(self) -> MetricGraph<ShortestPathsCache> {
        SpMetricGraph::from_graph_on_nodes(self.nodes().collect(), self)
    }
}

pub trait SpMetricCompletion<'a> {
    fn compute_metric_completion_by(
        self,
        sp: &'a ShortestPathsCache,
    ) -> MetricGraph<&'a ShortestPathsCache>;
}

impl<'a, G> SpMetricCompletion<'a> for &'a G
where
    G: Graph<'a> + Sync,
{
    fn compute_metric_completion_by(
        self,
        sp: &'a ShortestPathsCache,
    ) -> MetricGraph<&'a ShortestPathsCache> {
        let nodes: Vec<Node> = self.nodes().collect();
        let node_index = NodeIndex::init(&nodes);
        MetricGraph {
            nodes,
            node_index,
            metric: &sp,
        }
    }
}

impl<M> Metric for MetricGraph<M>
where
    M: Metric,
{
    fn distance(&self, node1: Node, node2: Node) -> Cost {
        self.metric.distance(node1, node2)
    }
}

#[derive(Debug, Clone)]
pub struct MetricGraph<M> {
    nodes: Vec<Node>,
    node_index: NodeIndex,
    metric: M,
}

impl <M> MetricGraph<M> where M: Metric + Clone + Debug {
    pub fn from_metric_on_nodes(nodes: Vec<Node>, metric: M) -> Self {
        let node_index = NodeIndex::init(&nodes);
        Self {
            nodes,
            node_index,
            metric,
        }
    }
}

pub type SpMetricGraph = MetricGraph<ShortestPathsCache>;

impl SpMetricGraph {
    pub fn from_graph_on_nodes<'a, G>(nodes: Vec<Node>, graph: &'a G) -> Self
    where
        G: Graph<'a> + Sync,
    {
        let node_index = NodeIndex::init(&nodes);
        let sp = ShortestPathsCache::compute_all_pairs_par(nodes.clone(), graph);
        Self {
            nodes,
            node_index,
            metric: sp,
        }
    }

    

    pub fn from_graph<'a, G>(graph: &'a G) -> Self
    where
        G: Graph<'a> + Sync,
    {
        Self::from_graph_on_nodes(graph.nodes().collect(), graph)
    }

    pub fn metric_clone(&self) -> ShortestPathsCache {
        self.metric.clone()
    }

    pub fn metric_ref(&self) -> &ShortestPathsCache {
        &self.metric
    }

    pub fn into_metric(self) -> ShortestPathsCache {
        self.metric
    }

    pub fn split_virtual_edge(
        &mut self,
        virtual_source: Node,
        virtual_sink: Node,
        at: Cost,
        base_graph: &mut AdjListGraph,
    ) -> Node {
        let (path_cost, path) = dijkstra_path(base_graph, virtual_source, virtual_sink);
        assert!(at > Cost::new(0));
        assert!(at < path_cost);
        assert_eq!(path.first().copied(), Some(virtual_source));
        assert_eq!(path.last().copied(), Some(virtual_sink));

        let mut at_node: Option<Node> = None;
        let mut walked = Cost::new(0);

        for edge in path.windows(2) {
            let edge_cost = base_graph.edge_cost(edge[0], edge[1]).unwrap();
            if walked + edge_cost == at {
                // split is at base_graph node edge[1]
                at_node = Some(edge[1]);
                assert!(self.metric.contains_node(edge[1]));
                break;
            } else if walked + edge_cost > at {
                let new_node = base_graph.split_edge_at(edge[0], edge[1], at - walked);
                self.metric.split_edge_to_buffer(new_node, edge[0], edge[1], at - walked, edge_cost, base_graph);
                at_node = Some(new_node);
                break;
            }
            walked += edge_cost;
        }

        let new_node = at_node.unwrap();
        if !self.nodes.contains(&new_node) {
            self.node_index.add_node(new_node);
            
        }

        return new_node;
    }
}

impl<M> Weighted for MetricGraph<M>
where
    M: Metric + Clone + Debug,
{
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        Some(self.metric.distance(node1, node2))
    }
}

impl<'a, M> Graph<'a> for MetricGraph<M>
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

impl<'a, M> Adjacency<'a> for MetricGraph<M>
where
    M: 'a + Metric,
{
    type AdjacencyIter = AdjacencyIter<'a, M>;

    fn adjacent(&'a self, node: Node) -> Self::AdjacencyIter {
        AdjacencyIter::new(&self.metric, node, self.nodes.iter())
    }
}

impl<'a, M> Cut<'a> for MetricGraph<M>
where
    M: 'a + Metric,
{
    type CutIter = CutIter<'a, MetricGraph<M>>;
    fn cut(&'a self, nodes: &NodeSet) -> Self::CutIter {
        CutIter::new(self, nodes)
    }
}

impl<M> GraphSize for MetricGraph<M> {
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

impl<'a, M> Neighbors<'a> for MetricGraph<M> {
    type NeighborIter = NeighborIter<'a>;

    fn neighbors(&'a self, node: Node) -> Self::NeighborIter {
        NeighborIter::new(node, self.nodes.iter())
    }
}

impl<'a, M> Nodes<'a> for MetricGraph<M> {
    type NodeIter = std::iter::Copied<std::slice::Iter<'a, Node>>;

    fn nodes(&'a self) -> Self::NodeIter {
        self.nodes.iter().copied()
    }
}

#[cfg(test)]
mod test_metric_graph {
    use crate::AdjListGraph;

    use super::*;

    #[test]
    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn test_from_graph_on_nodes() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        let nodes: Vec<Node> = vec![1.into(), 3.into(), 4.into(), 6.into()];

        let metric_graph = SpMetricGraph::from_graph_on_nodes(nodes, &graph);

        assert_eq!(4, metric_graph.n());
        assert!(metric_graph.contains_node(1.into()));
        assert!(metric_graph.contains_node(3.into()));
        assert!(metric_graph.contains_node(4.into()));
        assert!(metric_graph.contains_node(6.into()));

        assert_eq!(Cost::new(6), metric_graph.distance(1.into(), 3.into()));
        assert_eq!(Cost::new(3), metric_graph.distance(1.into(), 4.into()));
        assert_eq!(Cost::new(9), metric_graph.distance(1.into(), 6.into()));

        assert_eq!(Cost::new(3), metric_graph.distance(3.into(), 4.into()));
        assert_eq!(Cost::new(3), metric_graph.distance(3.into(), 6.into()));

        assert_eq!(Cost::new(6), metric_graph.distance(4.into(), 6.into()));
    }

    #[test]
    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn test_1() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        let sp = ShortestPathsCache::compute_all_graph_pairs(&graph);

        let nodes: Vec<Node> = vec![1.into(), 3.into(), 4.into(), 6.into()];
        let mut metric_graph = SpMetricGraph::from_metric_on_nodes(nodes, sp);

        metric_graph.split_virtual_edge(4.into(), 3.into(), Cost::new(1), &mut graph);
        assert!(metric_graph.contains_node(5.into()));
        assert_eq!(Cost::new(4), metric_graph.distance(5.into(), 1.into()));
        assert_eq!(Cost::new(2), metric_graph.distance(5.into(), 3.into()));
        assert_eq!(Cost::new(1), metric_graph.distance(5.into(), 4.into()));
        assert_eq!(Cost::new(5), metric_graph.distance(5.into(), 6.into()));

        metric_graph.split_virtual_edge(1.into(), 4.into(), Cost::new(1), &mut graph);
        assert!(metric_graph.contains_node(7.into()));
        assert_eq!(Cost::new(1), metric_graph.distance(7.into(), 1.into()));
        assert_eq!(Cost::new(5), metric_graph.distance(7.into(), 3.into()));
        assert_eq!(Cost::new(2), metric_graph.distance(7.into(), 4.into()));
        assert_eq!(Cost::new(3), metric_graph.distance(7.into(), 5.into()));
        assert_eq!(Cost::new(8), metric_graph.distance(7.into(), 6.into()));
    }
}
