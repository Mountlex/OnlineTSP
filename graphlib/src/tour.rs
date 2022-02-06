use crate::{dijkstra::dijkstra_path, Cost};

use super::{Edge, Graph, Node, NodeSet, ShortestPaths, Tree};

/// A tour in a graph. Note that we assume that `nodes.first() == nodes.last()`, and that no node except the start node appears more than once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tour {
    nodes: Vec<Node>,
    cost: Cost,
}

impl From<(Vec<usize>, Cost)> for Tour {
    fn from(input: (Vec<usize>, Cost)) -> Self {
        Tour::new(input.0.into_iter().map(|n| n.into()).collect(), input.1)
    }
}

impl IntoIterator for Tour {
    type Item = Node;
    type IntoIter = std::vec::IntoIter<Node>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

impl<'a> IntoIterator for &'a Tour {
    type Item = &'a Node;
    type IntoIter = std::slice::Iter<'a, Node>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.iter()
    }
}

impl Tour {
    pub fn empty() -> Self {
        Self {
            nodes: vec![],
            cost: Cost::new(0),
        }
    }

    pub fn new(nodes: Vec<Node>, cost: Cost) -> Self {
        assert_eq!(nodes.first(), nodes.last());
        debug_assert_eq!(
            nodes.iter().copied().collect::<NodeSet>().len(),
            nodes.len() - 1
        );
        Self { nodes, cost }
    }

    pub fn with_cost_from<G>(nodes: Vec<Node>, sp: &G) -> Self
    where
        G: ShortestPaths,
    {
        let mut cost = Cost::new(0);
        for e in nodes.windows(2) {
            cost += sp.shortest_path_cost(e[0], e[1]);
        }
        Self::new(nodes, cost)
    }

    pub fn cost(&self) -> Cost {
        self.cost
    }

    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_slice()
    }

    pub fn to_nodes(self) -> Vec<Node> {
        self.nodes
    }

    pub fn contains_edge(&self, edge: &Edge) -> bool {
        self.nodes
            .windows(2)
            .any(|n| edge.sink() == n[1] && edge.source() == n[0])
    }

    pub fn first(&self) -> Node {
        *self.nodes.first().unwrap()
    }

    pub fn last(&self) -> Node {
        *self.nodes.last().unwrap()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn get(&self, idx: usize) -> Option<Node> {
        self.nodes.get(idx).copied()
    }

    pub fn to_neighbor_tour<'a, G>(&self, graph: &'a G) -> NeighborTour
    where
        G: Graph<'a>,
    {
        let mut tour: Vec<Node> = vec![self.first()];
        for node in self.nodes.windows(2) {
            assert!(tour.last() == Some(&node[0]));

            let (_, path) = dijkstra_path(graph, node[0], node[1]);
            for n in path.into_iter().skip(1) {
                tour.push(n);
            }
            assert!(tour.last() == Some(&node[1]));
        }
        NeighborTour::new(tour, self.cost)
    }
}

/// An augmented tour in a graph. Note that we assume that `nodes.first() == nodes.last()`, and that two consecutive nodes in `nodes` are neighbors in the corresponding graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborTour {
    nodes: Vec<Node>,
    cost: Cost,
}

impl NeighborTour {
    pub fn new(nodes: Vec<Node>, cost: Cost) -> Self {
        assert_eq!(nodes.first(), nodes.last());
        Self { nodes, cost }
    }

    pub fn to_tree<'a, G>(&self, graph: &G) -> Tree
    where
        G: ShortestPaths,
    {
        let mut tree = Tree::empty();

        let mut visited = NodeSet::empty();

        for e in self.nodes.windows(2) {
            let n1 = e[0];
            let n2 = e[1];
            if !visited.contains(&n2) {
                visited.insert(n1);
                visited.insert(n2);
                let cost = graph.shortest_path_cost(n1, n2);

                tree.add_edge(n1, n2, cost);
            }
        }

        tree
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn cost(&self) -> Cost {
        self.cost
    }

    pub fn get(&self, idx: usize) -> Option<Node> {
        self.nodes.get(idx).copied()
    }
}

#[cfg(test)]
mod test_tour {
    use super::*;
    use crate::Graph;

    ///   5 --2 -- 1 --5-- 2 --1-- 3
    ///           |2|        
    ///            4
    #[test]
    fn test_to_neighbor_tour() {
        let mut graph = Tree::empty();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 5.into(), 2.into());
        graph.add_edge(1.into(), 4.into(), 2.into());
        graph.add_edge(2.into(), 3.into(), 1.into());

        let tour = Tour::with_cost_from(
            vec![1, 2, 3, 4, 5, 1]
                .into_iter()
                .map(|n| n.into())
                .collect(),
            &graph,
        );
        assert_eq!(tour.cost, 20.into());

        let n_tour = tour.to_neighbor_tour(&graph);
        assert_eq!(n_tour.cost, 20.into());
        assert_eq!(
            n_tour.nodes,
            vec![1, 2, 3, 2, 1, 4, 1, 5, 1]
                .into_iter()
                .map(|n| Node::new(n))
                .collect::<Vec<Node>>()
        );
    }

    ///   5 --2 -- 1 --5-- 2 --1-- 3
    ///           |2|        
    ///            4
    #[test]
    fn test_to_tree() {
        let mut graph = Tree::empty();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 5.into(), 2.into());
        graph.add_edge(1.into(), 4.into(), 2.into());
        graph.add_edge(2.into(), 3.into(), 1.into());

        let tour = Tour::with_cost_from(
            vec![1, 2, 3, 4, 5, 1]
                .into_iter()
                .map(|n| n.into())
                .collect(),
            &graph,
        );
        let n_tour = tour.to_neighbor_tour(&graph);

        let tree = n_tour.to_tree(&graph);

        assert!(tree.contains_edge(1.into(), 2.into()));
        assert!(tree.contains_edge(2.into(), 3.into()));
        assert!(tree.contains_edge(1.into(), 4.into()));
        assert!(tree.contains_edge(1.into(), 5.into()));
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

        let tour = Tour::with_cost_from(
            vec![1, 2, 3, 4, 5, 1]
                .into_iter()
                .map(|n| n.into())
                .collect(),
            &graph,
        );
        let n_tour = tour.to_neighbor_tour(&graph);
        let tree = n_tour.to_tree(&graph);

        let new_tour = tree.to_tour();
        assert_eq!(tour, new_tour);
    }
}
