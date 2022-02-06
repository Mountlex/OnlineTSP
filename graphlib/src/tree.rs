use rand::seq::IteratorRandom;

use crate::{dijkstra::dijkstra, Cost, Weighted};

use super::{
    Adjacency, Cut, CutIter, Edge, Graph, GraphSize, Neighbors, Node, NodeSet, Nodes,
    ShortestPaths, TotalWeight, Tour,
};

#[derive(Clone, Debug)]
pub struct Tree {
    root: Option<TreeNode>,
}

impl Tree {
    pub fn add_edge_raw(&mut self, edge: Edge) -> bool {
        if self.contains_node(edge.sink()) && self.contains_node(edge.source()) {
            false // no circles
        } else if let Some(root) = &mut self.root {
            root.add_edge(edge)
        } else {
            self.root = Some(TreeNode::new(edge.source(), 0.into()));
            self.root.as_mut().unwrap().add_edge(edge)
        }
    }

    pub fn add_edge(&mut self, source: Node, sink: Node, cost: Cost) -> bool {
        self.add_edge_raw(Edge::new(source, sink, cost))
    }

    pub fn add_node(&mut self, node: Node) -> bool {
        if self.root.is_none() {
            self.root = Some(TreeNode::new(node, 0.into()));
            true
        } else {
            false
        }
    }

    pub fn empty() -> Self {
        Tree { root: None }
    }

    /// Returns a random subgraph of this three with `num_nodes` many nodes.
    pub fn random_subtree(&self, num_nodes: usize) -> Tree {
        if let Some(root) = &self.root {
            assert!(num_nodes <= root.nodes_in_subtree);
            if num_nodes > 1 {
                let mut tree = Tree::empty();
                root.random_subtree(None, num_nodes, &mut tree);
                tree
            } else if num_nodes == 1 {
                Tree {
                    root: Some(TreeNode::new(root.node, 0.into())),
                }
            } else {
                Tree::empty()
            }
        } else {
            Tree::empty()
        }
    }

    pub fn to_tour(&self) -> Tour {
        let mut nodes = vec![];
        if let Some(root) = &self.root {
            root.construct_tour(&mut nodes);
            nodes.push(root.node);
        }
        Tour::with_cost_from(nodes, self)
    }
}

impl<'a> Cut<'a> for Tree {
    type CutIter = CutIter<'a, Tree>;
    fn cut(&'a self, nodes: &NodeSet) -> Self::CutIter {
        CutIter::new(self, nodes)
    }
}

pub struct NeighborIterator<'a> {
    tree: &'a Tree,
    node: Node,
    tree_node: Option<&'a TreeNode>,
    child_idx: usize,
}

impl<'a> NeighborIterator<'a> {
    fn new(tree: &'a Tree, node: Node) -> Self {
        Self {
            tree,
            node,
            tree_node: None,
            child_idx: 0,
        }
    }
}

impl<'a> Iterator for NeighborIterator<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tree_node.is_none() {
            if let Some((parent, tree_node)) = self
                .tree
                .root
                .as_ref()
                .and_then(|r| r.find_edge_to(self.node))
            {
                self.tree_node = Some(tree_node);
                return Some(parent.node);
            } else if self.tree.root.as_ref().map(|r| r.node) == Some(self.node) {
                self.tree_node = self.tree.root.as_ref();
            } else {
                return None;
            }
        }

        if let Some(tree_node) = self.tree_node {
            if self.child_idx < tree_node.children.len() {
                let c = tree_node.children.get(self.child_idx).unwrap();
                self.child_idx += 1;
                return Some(c.node);
            } else {
                return None;
            }
        } else {
            panic!("Should not happen!")
        }
    }
}

impl<'a> Neighbors<'a> for Tree {
    type NeighborIter = NeighborIterator<'a>;

    fn neighbors(&'a self, node: Node) -> Self::NeighborIter {
        NeighborIterator::new(self, node)
    }
}

impl ShortestPaths for Tree {
    fn shortest_path_cost(&self, n1: Node, n2: Node) -> Cost {
        dijkstra(self, n1, n2)
    }
}

pub struct AdjacencyIterator<'a> {
    tree: &'a Tree,
    node: Node,
    tree_node: Option<&'a TreeNode>,
    child_idx: usize,
}

impl<'a> AdjacencyIterator<'a> {
    fn new(tree: &'a Tree, node: Node) -> Self {
        Self {
            tree,
            node,
            tree_node: None,
            child_idx: 0,
        }
    }
}

impl<'a> Iterator for AdjacencyIterator<'a> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tree_node.is_none() {
            if let Some((parent, tree_node)) = self
                .tree
                .root
                .as_ref()
                .and_then(|r| r.find_edge_to(self.node))
            {
                self.tree_node = Some(tree_node);
                return Some(Edge::new(tree_node.node, parent.node, tree_node.cost));
            } else if self.tree.root.as_ref().map(|r| r.node) == Some(self.node) {
                self.tree_node = self.tree.root.as_ref();
            } else {
                return None;
            }
        }

        if let Some(tree_node) = self.tree_node {
            if self.child_idx < tree_node.children.len() {
                let c = tree_node.children.get(self.child_idx).unwrap();
                self.child_idx += 1;
                return Some(Edge::new(tree_node.node, c.node, c.cost));
            } else {
                return None;
            }
        } else {
            panic!("Should not happen!")
        }
    }
}

impl<'a> Adjacency<'a> for Tree {
    type AdjacencyIter = AdjacencyIterator<'a>;

    fn adjacent(&'a self, node: Node) -> Self::AdjacencyIter {
        AdjacencyIterator::new(self, node)
    }
}

pub struct NodeIter<'a> {
    stack: Vec<(&'a TreeNode, usize)>,
}

impl<'a> NodeIter<'a> {
    fn new(root: Option<&'a TreeNode>) -> Self {
        if let Some(root) = root {
            Self {
                stack: vec![(root, 0)],
            }
        } else {
            Self { stack: vec![] }
        }
    }
}

impl<'a> Iterator for NodeIter<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((node, idx)) = self.stack.last_mut() {
                if *idx < node.children.len() {
                    let i = *idx;
                    *idx += 1;
                    let n = node.children.get(i).unwrap();
                    self.stack.push((n, 0))
                } else {
                    let n = node.node;
                    self.stack.pop();
                    return Some(n);
                }
            } else {
                return None;
            }
        }
    }
}

impl<'a> Nodes<'a> for Tree {
    type NodeIter = NodeIter<'a>;

    fn nodes(&'a self) -> Self::NodeIter {
        NodeIter::new(self.root.as_ref())
    }
}

impl GraphSize for Tree {
    fn n(&self) -> usize {
        if let Some(root) = &self.root {
            root.nodes_in_subtree
        } else {
            0
        }
    }
}

impl<'a> Graph<'a> for Tree {
    fn contains_edge(&self, node1: Node, node2: Node) -> bool {
        if let Some(ref root) = self.root {
            root.contains_edge(None, node1, node2)
        } else {
            false
        }
    }

    fn contains_node(&self, node: Node) -> bool {
        if let Some(ref root) = self.root {
            root.contains_node(node)
        } else {
            false
        }
    }
}

impl Weighted for Tree {
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        self.root.as_ref().and_then(|r| r.edge_cost(node1, node2))
    }
}

impl TotalWeight for Tree {
    fn total_weight(&self) -> Cost {
        self.root
            .as_ref()
            .map(|r| r.total_weight())
            .unwrap_or(0.into())
    }
}

#[derive(Clone, Debug)]
struct TreeNode {
    node: Node,
    cost: Cost,
    children: Vec<TreeNode>,
    nodes_in_subtree: usize,
}

impl TotalWeight for TreeNode {
    fn total_weight(&self) -> Cost {
        self.cost + self.children.iter().map(|c| c.total_weight()).sum()
    }
}

impl TreeNode {
    fn new(node: Node, cost: Cost) -> Self {
        TreeNode {
            node,
            cost,
            children: vec![],
            nodes_in_subtree: 1,
        }
    }

    fn add_edge(&mut self, edge: Edge) -> bool {
        if self.node == edge.source() {
            if !self.children.iter().any(|c| c.node == edge.sink()) {
                self.children.push(TreeNode::new(edge.sink(), edge.cost()));
                self.nodes_in_subtree += 1;
                true
            } else {
                false
            }
        } else if self.node == edge.sink() {
            if !self.children.iter().any(|c| c.node == edge.source()) {
                self.children
                    .push(TreeNode::new(edge.source(), edge.cost()));
                self.nodes_in_subtree += 1;
                true
            } else {
                false
            }
        } else {
            for child in self.children.iter_mut() {
                if child.add_edge(edge) {
                    self.nodes_in_subtree += 1;
                    return true;
                }
            }
            false
        }
    }

    fn random_subtree(&self, prev: Option<Node>, nodes_to_add: usize, tree: &mut Tree) {
        assert!(nodes_to_add <= self.nodes_in_subtree);
        if nodes_to_add > 0 {
            if let Some(prev) = prev {
                tree.add_edge(prev, self.node, self.cost);
            }

            let mut rng = rand::thread_rng();

            let mut remaining = nodes_to_add - 1;
            let mut distr: Vec<usize> = self.children.iter().map(|c| c.nodes_in_subtree).collect();
            while remaining > 0 {
                let rem = distr
                    .iter_mut()
                    .filter(|nodes| **nodes > 0)
                    .choose(&mut rng)
                    .unwrap();
                *rem -= 1;
                remaining -= 1;
            }
            for (child, rem) in self.children.iter().zip(distr) {
                child.random_subtree(Some(self.node), child.nodes_in_subtree - rem, tree);
            }
        }
    }

    fn construct_tour(&self, nodes: &mut Vec<Node>) {
        nodes.push(self.node);

        for child in self.children.iter() {
            child.construct_tour(nodes);
        }
    }

    fn find_edge_to<'a>(&'a self, node: Node) -> Option<(&'a TreeNode, &'a TreeNode)> {
        for child in self.children.iter() {
            if child.node == node {
                return Some((self, child));
            }
            if let Some(res) = child.find_edge_to(node) {
                return Some(res);
            }
        }
        None
    }

    fn contains_edge(&self, prev: Option<Node>, node1: Node, node2: Node) -> bool {
        if prev.is_some()
            && ((prev.unwrap() == node1 && self.node == node2)
                || (prev.unwrap() == node2 && self.node == node1))
        {
            true
        } else {
            for child in self.children.iter() {
                if child.contains_edge(Some(self.node), node1, node2) {
                    return true;
                }
            }
            false
        }
    }

    fn contains_node(&self, node: Node) -> bool {
        if self.node == node {
            true
        } else {
            for child in self.children.iter() {
                if child.contains_node(node) {
                    return true;
                }
            }
            false
        }
    }

    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        for child in self.children.iter() {
            if (self.node == node1 && child.node == node2)
                || (self.node == node2 && child.node == node1)
            {
                return Some(child.cost);
            }

            if let Some(cost) = child.edge_cost(node1, node2) {
                return Some(cost);
            }
        }
        None
    }
}

#[cfg(test)]
mod test_tree {
    use super::*;

    ///   7 --2 -- 1 --5-- 2 --1-- 3
    ///           |2|        
    ///            4 --1-- 5 --6-- 6
    #[test]
    fn test_tree() {
        let mut tree = Tree::empty();
        assert!(tree.add_edge(1.into(), 2.into(), 5.into()));
        assert!(tree.add_edge(1.into(), 7.into(), 2.into()));
        assert!(tree.add_edge(1.into(), 4.into(), 2.into()));
        assert!(tree.add_edge(4.into(), 5.into(), 1.into()));
        assert!(!tree.add_edge(2.into(), 5.into(), 1.into()));
        assert!(tree.add_edge(2.into(), 3.into(), 1.into()));
        assert!(tree.add_edge(5.into(), 6.into(), 6.into()));
        assert!(!tree.add_edge(3.into(), 6.into(), 1.into()));

        assert_eq!(tree.neighbors(1.into()).count(), 3);
        assert_eq!(tree.neighbors(3.into()).count(), 1);
        assert_eq!(tree.neighbors(5.into()).count(), 2);
        assert_eq!(tree.neighbors(7.into()).count(), 1);
        assert_eq!(tree.neighbors(8.into()).count(), 0);

        assert_eq!(tree.adjacent(1.into()).count(), 3);
        assert_eq!(tree.adjacent(3.into()).count(), 1);
        assert_eq!(tree.adjacent(5.into()).count(), 2);

        assert_eq!(tree.edge_cost(5.into(), 6.into()).unwrap(), Cost::new(6));
        assert_eq!(tree.edge_cost(3.into(), 2.into()).unwrap(), Cost::new(1));
        assert_eq!(tree.edge_cost(5.into(), 1.into()), None);

        assert_eq!(tree.nodes().count(), 7);
        assert_eq!(tree.n(), 7);
        assert!(tree.contains_edge(7.into(), 1.into()));
        assert!(tree.contains_edge(4.into(), 1.into()));
        assert!(tree.contains_edge(2.into(), 3.into()));
        assert!(!tree.contains_edge(1.into(), 6.into()));
        assert!(!tree.contains_edge(6.into(), 4.into()));
    }

    #[test]
    fn test_random_subtree() {
        let mut tree = Tree::empty();
        tree.add_edge(1.into(), 2.into(), 5.into());
        tree.add_edge(1.into(), 7.into(), 2.into());
        tree.add_edge(1.into(), 4.into(), 2.into());
        tree.add_edge(4.into(), 5.into(), 1.into());
        tree.add_edge(2.into(), 3.into(), 1.into());
        tree.add_edge(5.into(), 6.into(), 6.into());

        assert_eq!(tree.random_subtree(7).n(), 7);
        assert_eq!(tree.random_subtree(6).n(), 6);
        assert_eq!(tree.random_subtree(5).n(), 5);
        assert_eq!(tree.random_subtree(2).n(), 2);
        assert_eq!(tree.random_subtree(1).n(), 1);
        assert_eq!(tree.random_subtree(0).n(), 0);
    }

    ///   7 --2 -- 1 --5-- 2 --1-- 3
    ///           |2|        
    ///            4 --1-- 5 --6-- 6
    #[test]
    fn test_to_tour() {
        let mut tree = Tree::empty();
        tree.add_edge(1.into(), 2.into(), 5.into());
        tree.add_edge(1.into(), 4.into(), 2.into());
        tree.add_edge(1.into(), 7.into(), 2.into());
        tree.add_edge(4.into(), 5.into(), 1.into());
        tree.add_edge(2.into(), 3.into(), 1.into());
        tree.add_edge(5.into(), 6.into(), 6.into());

        let tour = tree.to_tour().to_neighbor_tour(&tree);

        assert_eq!(tour.get(0), Some(1.into()));
        assert_eq!(tour.get(1), Some(2.into()));
        assert_eq!(tour.get(2), Some(3.into()));
        assert_eq!(tour.get(3), Some(2.into()));
        assert_eq!(tour.get(4), Some(1.into()));
        assert_eq!(tour.get(5), Some(4.into()));
        assert_eq!(tour.get(6), Some(5.into()));
        assert_eq!(tour.get(7), Some(6.into()));
        assert_eq!(tour.get(8), Some(5.into()));
        assert_eq!(tour.get(9), Some(4.into()));
        assert_eq!(tour.get(10), Some(1.into()));
        assert_eq!(tour.get(11), Some(7.into()));
        assert_eq!(tour.get(12), Some(1.into()));

        assert_eq!(tour.len(), 13);
    }

    ///   5 --2 -- 1 --5-- 2 --1-- 3
    ///           |2|        
    ///            4
    #[test]
    fn test_tour_tree_tour() {
        let mut graph = Tree::empty();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 5.into(), 2.into());
        graph.add_edge(1.into(), 4.into(), 2.into());
        graph.add_edge(2.into(), 3.into(), 1.into());

        let tour = graph.clone().to_tour();
        let n_tour = tour.to_neighbor_tour(&graph);
        let tree = n_tour.to_tree(&graph);

        assert!(tree.contains_edge(1.into(), 2.into()));
        assert!(tree.contains_edge(2.into(), 3.into()));
        assert!(tree.contains_edge(1.into(), 4.into()));
        assert!(tree.contains_edge(1.into(), 5.into()));
    }
}
