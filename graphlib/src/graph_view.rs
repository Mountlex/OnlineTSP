use crate::{Cost, Weighted};

use super::{Adjacency, Cut, CutIter, Edge, Graph, GraphSize, Neighbors, Node, NodeSet, Nodes};

#[derive(Clone, Debug)]
pub struct GraphView<'a, G> {
    base: &'a G,
    component: Node,
    edge_bound: EdgeBound,
    nodes: NodeSet,
}
#[derive(Clone, Debug)]
enum EdgeBound {
    Upper(Cost),
    UpperStrict(Cost),
}

impl EdgeBound {
    fn check(&self, edge: &Edge) -> bool {
        match self {
            EdgeBound::Upper(bound) => edge.cost() <= *bound,
            EdgeBound::UpperStrict(bound) => edge.cost() < *bound,
        }
    }
}

impl<'a, G> GraphView<'a, G>
where
    G: GraphSize + Adjacency<'a>,
{
    fn new(base: &'a G, component: Node, edge_bound: EdgeBound) -> Self {
        let mut nodes: NodeSet = NodeSet::empty();
        let mut fixed_nodes = fixedbitset::FixedBitSet::with_capacity(base.n() + 1);
        let mut stack: Vec<Node> = vec![component];
        while let Some(node) = stack.pop() {
            nodes.insert(node);
            fixed_nodes.insert(node.id());
            for edge in base.adjacent(node) {
                if edge_bound.check(&edge) && !fixed_nodes.contains(edge.sink().id()) {
                    stack.push(edge.sink())
                }
            }
        }

        GraphView {
            base,
            component,
            edge_bound,
            nodes,
        }
    }

    pub fn component(&self) -> Node {
        self.component.clone()
    }

    pub fn with_upper_bound(base: &'a G, component: Node, upper_bound: Cost) -> Self {
        GraphView::new(base, component, EdgeBound::Upper(upper_bound))
    }

    pub fn with_upper_bound_strict(base: &'a G, component: Node, upper_bound: Cost) -> Self {
        GraphView::new(base, component, EdgeBound::UpperStrict(upper_bound))
    }
}

impl<'a, G> GraphSize for GraphView<'a, G>
where
    G: GraphSize,
{
    fn n(&self) -> usize {
        self.nodes.len()
    }
}

impl<'a, G> Cut<'a> for GraphView<'a, G>
where
    G: Adjacency<'a>,
{
    type CutIter = CutIter<'a, GraphView<'a, G>>;
    fn cut(&'a self, nodes: &NodeSet) -> Self::CutIter {
        CutIter::new(self, nodes)
    }
}

impl<'a, G> Graph<'a> for GraphView<'a, G>
where
    G: Graph<'a> + Weighted,
{
    fn contains_node(&self, node: Node) -> bool {
        self.nodes.contains(&node)
    }

    fn contains_edge(&self, node1: Node, node2: Node) -> bool {
        if let Some(cost) = self.base.edge_cost(node1, node2) {
            self.edge_bound.check(&Edge::new(node1, node2, cost))
        } else {
            false
        }
    }
}

impl<'a, G> Weighted for GraphView<'a, G>
where
    G: Weighted,
{
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        if let Some(cost) = self.base.edge_cost(node1, node2) {
            if self.edge_bound.check(&Edge::new(node1, node2, cost)) {
                Some(cost)
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub struct AdjacencyIter<'a, G>
where
    G: Adjacency<'a>,
{
    iter: <G as Adjacency<'a>>::AdjacencyIter,
    edge_bound: &'a EdgeBound,
}

impl<'a, G> Iterator for AdjacencyIter<'a, G>
where
    G: Adjacency<'a>,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(edge) = self.iter.next() {
                if self.edge_bound.check(&edge) {
                    return Some(edge);
                }
            } else {
                return None;
            }
        }
    }
}

impl<'a, G> Adjacency<'a> for GraphView<'a, G>
where
    G: Adjacency<'a>,
{
    type AdjacencyIter = AdjacencyIter<'a, G>;

    fn adjacent(&'a self, node: Node) -> Self::AdjacencyIter {
        AdjacencyIter {
            iter: self.base.adjacent(node),
            edge_bound: &self.edge_bound,
        }
    }
}

pub struct NeighborIter<'a, G>
where
    G: Neighbors<'a>,
{
    iter: <G as Neighbors<'a>>::NeighborIter,
    nodes: &'a NodeSet,
}

impl<'a, G> Iterator for NeighborIter<'a, G>
where
    G: Neighbors<'a>,
{
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(node) = self.iter.next() {
                if self.nodes.contains(&node) {
                    return Some(node);
                }
            } else {
                return None;
            }
        }
    }
}

impl<'a, G> Neighbors<'a> for GraphView<'a, G>
where
    G: Neighbors<'a>,
{
    type NeighborIter = NeighborIter<'a, G>;

    fn neighbors(&'a self, node: Node) -> Self::NeighborIter {
        NeighborIter {
            iter: self.base.neighbors(node),
            nodes: &self.nodes,
        }
    }
}

impl<'a, G> Nodes<'a> for GraphView<'a, G>
where
    G: Nodes<'a>,
{
    type NodeIter = <&'a NodeSet as IntoIterator>::IntoIter;

    fn nodes(&'a self) -> Self::NodeIter {
        (&self.nodes).into_iter()
    }
}
