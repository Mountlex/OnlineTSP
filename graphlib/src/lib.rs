mod christofides;
mod cost;
pub mod dijkstra;
mod edge;
mod graph_view;
mod index;
mod metric;
mod metric_view;
pub mod mst;
mod multi_graph;
mod node;
mod single_graph;
pub mod sp;
mod tour;
mod tree;
pub mod tsp;
pub mod two_opt;

pub mod graphml;

use std::fmt::Debug;

pub use edge::*;
pub use graph_view::GraphView;
pub use index::*;
pub use multi_graph::MultiGraph;
pub use node::*;
pub use single_graph::AdjListGraph;
pub use tour::{NeighborTour, Tour};
pub use tree::Tree;

pub use metric::*;
pub use metric_view::*;

pub use cost::Cost;

pub trait Graph<'a>:
    Adjacency<'a> + Neighbors<'a> + GraphSize + Nodes<'a> + Cut<'a> + Debug + Clone
{
    fn contains_node(&self, node: Node) -> bool;

    fn contains_edge(&self, node1: Node, node2: Node) -> bool;
}

pub trait Connected<'a>: Component<'a> + DetStartNode<'a> + GraphSize {
    fn connected(&'a self) -> bool {
        let component = self.component_of(self.start_node());
        component.len() == self.n()
    }
}

impl<'a, G> Connected<'a> for G where G: Component<'a> + DetStartNode<'a> + GraphSize {}

pub trait Weighted {
    /// Returns the cost of an edge between two nodes if such exists.
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost>;
}

pub struct CutIter<'a, G>
where
    G: Adjacency<'a>,
{
    graph: &'a G,
    fixed_nodes: fixedbitset::FixedBitSet,
    nodes: Vec<Node>,
    node_idx: usize,
    adj_iter: Option<<G as Adjacency<'a>>::AdjacencyIter>,
}

impl<'a, G> CutIter<'a, G>
where
    G: Adjacency<'a>,
{
    fn new(graph: &'a G, nodes: &NodeSet) -> Self {
        let max = nodes.into_iter().map(|n| n.id()).max().unwrap();
        let mut fixed_nodes = fixedbitset::FixedBitSet::with_capacity(max + 1);
        for node in nodes {
            fixed_nodes.insert(node.id());
        }
        Self {
            graph,
            fixed_nodes,
            nodes: nodes.to_vec(),
            node_idx: 0,
            adj_iter: None,
        }
    }
}

impl<'a, G> Iterator for CutIter<'a, G>
where
    G: Adjacency<'a>,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(adj_iter) = &mut self.adj_iter {
                if let Some(next) = adj_iter.next() {
                    if !self.fixed_nodes.contains(next.sink().id()) {
                        return Some(next);
                    } else {
                        continue;
                    }
                }
            }
            if self.node_idx < self.nodes.len() {
                self.adj_iter = Some(self.graph.adjacent(self.nodes[self.node_idx]));
                self.node_idx += 1;
            } else {
                return None;
            }
        }
    }
}

pub trait ShortestPaths {
    fn shortest_path_cost(&self, n1: Node, n2: Node) -> Cost;
}

pub trait Cut<'a> {
    type CutIter: Iterator<Item = Edge>;
    fn cut(&'a self, nodes: &NodeSet) -> Self::CutIter;
}

pub trait Neighbors<'a> {
    type NeighborIter: Iterator<Item = Node>;

    fn neighbors(&'a self, node: Node) -> Self::NeighborIter;
}

pub trait Nodes<'a> {
    type NodeIter: Iterator<Item = Node>;

    fn nodes(&'a self) -> Self::NodeIter;
}

pub struct DefaultNodeIter<'a>(std::slice::Iter<'a, Node>);

impl<'a> Iterator for DefaultNodeIter<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().copied()
    }
}

impl<'a> Nodes<'a> for Vec<Node> {
    type NodeIter = DefaultNodeIter<'a>;

    fn nodes(&'a self) -> Self::NodeIter {
        DefaultNodeIter(self.iter())
    }
}

pub trait Edges<'a> {
    type EdgeIter: Iterator<Item = Edge>;

    fn edges(&'a self) -> Self::EdgeIter;
}

pub trait Adjacency<'a> {
    type AdjacencyIter: Iterator<Item = Edge>;

    fn adjacent(&'a self, node: Node) -> Self::AdjacencyIter;
}

pub trait DetStartNode<'a> {
    fn start_node(&'a self) -> Node;
}

impl<'a, G> DetStartNode<'a> for G
where
    G: Graph<'a> + Nodes<'a>,
{
    fn start_node(&'a self) -> Node {
        self.nodes().min().unwrap()
    }
}

pub trait Component<'a> {
    fn component_of(&'a self, node: Node) -> NodeSet;
}

pub trait GraphSize {
    fn n(&self) -> usize;
}

impl<'a, G> Component<'a> for G
where
    G: Neighbors<'a> + GraphSize,
{
    fn component_of(&'a self, component: Node) -> NodeSet {
        let mut nodes: NodeSet = NodeSet::empty();
        let mut fixed_nodes = fixedbitset::FixedBitSet::with_capacity(self.n() + 1);
        let mut stack: Vec<Node> = vec![component];
        while let Some(node) = stack.pop() {
            nodes.insert(node);
            fixed_nodes.insert(node.id());
            for n in self.neighbors(node) {
                if !fixed_nodes.contains(n.id()) {
                    stack.push(n)
                }
            }
        }
        nodes
    }
}

pub trait ExpRoundable {
    fn to_exp_rounded(&self) -> Self;
}

pub trait TotalWeight {
    fn total_weight(&self) -> Cost;
}

#[cfg(test)]
mod test_graph {
    use super::*;

    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    #[test]
    fn test_graph_construction() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        assert_eq!(6, graph.n());

        assert!(graph.contains_node(1.into()));
        assert!(graph.contains_node(2.into()));
        assert!(graph.contains_node(3.into()));
        assert!(graph.contains_node(4.into()));
        assert!(graph.contains_node(5.into()));
        assert!(graph.contains_node(6.into()));

        assert!(graph.contains_edge(1.into(), 2.into()));
        assert!(graph.contains_edge(1.into(), 4.into()));
        assert!(graph.contains_edge(2.into(), 5.into()));
        assert!(graph.contains_edge(2.into(), 3.into()));
        assert!(graph.contains_edge(3.into(), 6.into()));
        assert!(graph.contains_edge(4.into(), 5.into()));
        assert!(graph.contains_edge(5.into(), 6.into()));

        assert!(graph.contains_edge(2.into(), 1.into()));
        assert!(graph.contains_edge(4.into(), 1.into()));
        assert!(graph.contains_edge(5.into(), 2.into()));
        assert!(graph.contains_edge(3.into(), 2.into()));
        assert!(graph.contains_edge(6.into(), 3.into()));
        assert!(graph.contains_edge(5.into(), 4.into()));
        assert!(graph.contains_edge(6.into(), 5.into()));
    }

    #[test]
    fn test_neighbors() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 3.into(), 3.into());
        graph.add_edge(2.into(), 3.into(), 1.into());

        assert!(graph
            .neighbors(1.into())
            .collect::<NodeSet>()
            .contains(&2.into()));
        assert!(graph
            .neighbors(1.into())
            .collect::<NodeSet>()
            .contains(&3.into()));

        let adjacent: Vec<Edge> = graph.adjacent(1.into()).collect();

        assert!(adjacent
            .clone()
            .into_iter()
            .collect::<EdgeSet>()
            .contains_sink(&2.into()));
        assert!(adjacent
            .into_iter()
            .collect::<EdgeSet>()
            .contains_sink(&3.into()));
    }

    #[test]
    fn test_cut() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 3.into(), 3.into());
        graph.add_edge(2.into(), 3.into(), 1.into());

        let s: NodeSet = vec![1.into()].into_iter().collect();

        assert!(graph.cut(&s).collect::<EdgeSet>().contains_sink(&2.into()));
        assert!(graph.cut(&s).collect::<EdgeSet>().contains_sink(&3.into()));
    }
}
