use crate::{cost::Cost, AdjListGraph, Adjacency, Node, Nodes, Tree};
use binary_heap_plus::BinaryHeap;
use compare::Compare;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

struct PrioComp(HashMap<Node, Cost>);

impl Compare<Node> for PrioComp {
    fn compare(&self, l: &Node, r: &Node) -> Ordering {
        self.0
            .get(r)
            .unwrap()
            .cmp(self.0.get(l).unwrap())
            .then(r.cmp(l))
    }
}

fn prims<'a, G>(graph: &'a G) -> (Option<Node>, HashMap<Node, Vec<(Node, Cost)>>)
where
    G: Nodes<'a> + Adjacency<'a>,
{
    let n = graph.nodes().count();
    if n == 0 {
        return (None, HashMap::new());
    }
    let mut start_node: Option<Node> = None;

    let mut costs = HashMap::<Node, Cost>::with_capacity(n);
    let mut parent = HashMap::<Node, (Node, Cost)>::with_capacity(n);

    let nodes_vec: Vec<Node> = graph.nodes().into_iter().collect();
    for n in nodes_vec.iter() {
        costs.insert(*n, Cost::max());
    }

    let mut heap = BinaryHeap::from_vec_cmp(nodes_vec, PrioComp(costs.clone()));
    let mut visited_nodes = HashSet::<Node>::new();
    while let Some(u) = heap.pop() {
        if visited_nodes.is_empty() {
            start_node = Some(u);
        }

        visited_nodes.insert(u);
        for e in graph.adjacent(u) {
            let v = e.sink();
            if !visited_nodes.contains(&v) && e.cost() < costs[&v] {
                costs.insert(v, e.cost());
                parent.insert(v, (u, e.cost()));
            }
        }
        heap.replace_cmp(PrioComp(costs.clone()));
    }

    let mut childs: HashMap<Node, Vec<(Node, Cost)>> = HashMap::new();
    for (n1, (n2, cost)) in parent {
        childs.entry(n2).or_default().push((n1, cost));
    }
    (start_node, childs)
}

pub fn prims_tree<'a, G>(graph: &'a G) -> (Tree, Cost)
where
    G: Nodes<'a> + Adjacency<'a>,
{
    let mut total_cost = 0.into();
    let mut mst = Tree::empty();

    let (start_node, childs) = prims(graph);
    if let Some(root) = start_node {
        mst.add_node(root);

        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if let Some(c) = childs.get(&node) {
                for (child, cost) in c {
                    assert!(mst.add_edge(node, *child, *cost));
                    stack.push(*child);
                    total_cost += *cost;
                }
            }
        }
    }
    (mst, total_cost)
}

pub fn prims_adj_graph<'a, G>(graph: &'a G) -> (AdjListGraph, Cost)
where
    G: Nodes<'a> + Adjacency<'a>,
{
    let mut total_cost = 0.into();
    let mut mst = AdjListGraph::new();

    let childs = prims(graph).1;
    for (node, childs) in childs {
        for (child, cost) in childs {
            total_cost += cost;
            mst.add_edge(node, child, cost);
            mst.add_edge(child, node, cost);
        }
    }

    (mst, total_cost)
}

pub fn prims_cost<'a, G>(graph: &'a G) -> Cost
where
    G: Nodes<'a> + Adjacency<'a>,
{
    let mut total_cost = 0.into();

    let childs = prims(graph).1;
    for (_node, childs) in childs {
        for (_child, cost) in childs {
            total_cost += cost;
        }
    }

    total_cost
}

#[cfg(test)]
mod test_prims {
    use super::*;
    use crate::{GraphSize, TotalWeight, Graph};

    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    #[test]
    fn test_min_spanning_tree_tree() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        let (mst, total_cost) = prims_tree(&graph);

        assert_eq!(6, mst.n());
        assert_eq!(Cost::new(9), total_cost);
        assert_eq!(mst.total_weight(), total_cost);

        assert!(mst.contains_edge(1.into(), 4.into()));
        assert!(mst.contains_edge(4.into(), 5.into()));
        assert!(mst.contains_edge(5.into(), 2.into()));
        assert!(mst.contains_edge(2.into(), 3.into()));
        assert!(mst.contains_edge(3.into(), 6.into()));
    }

    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    #[test]
    fn test_min_spanning_tree_graph() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        let (mst, total_cost) = prims_adj_graph(&graph);

        assert_eq!(6, mst.n());
        assert_eq!(Cost::new(9), total_cost);
        assert_eq!(mst.total_weight(), total_cost);

        assert!(mst.contains_edge(1.into(), 4.into()));
        assert!(mst.contains_edge(4.into(), 5.into()));
        assert!(mst.contains_edge(5.into(), 2.into()));
        assert!(mst.contains_edge(2.into(), 3.into()));
        assert!(mst.contains_edge(3.into(), 6.into()));
    }

    #[test]
    fn test_singleton() {
        let mut graph = AdjListGraph::new();
        graph.add_node(1.into());

        let (mst, total_cost) = prims_tree(&graph);

        assert_eq!(1, mst.n());
        assert_eq!(Cost::new(0), total_cost);
    }

    #[test]
    fn test_empty() {
        let graph = AdjListGraph::new();

        let (mst, total_cost) = prims_tree(&graph);

        assert_eq!(0, mst.n());
        assert_eq!(Cost::new(0), total_cost);
    }

    ///   7 --2 -- 1 --5-- 2 --1-- 3
    ///           |2|        
    ///            4 --1-- 5 --6-- 6
    #[test]
    fn test_test() {
        let mut tree = Tree::empty();
        tree.add_edge(1.into(), 2.into(), 5.into());
        tree.add_edge(1.into(), 7.into(), 2.into());
        tree.add_edge(1.into(), 4.into(), 2.into());
        tree.add_edge(4.into(), 5.into(), 1.into());
        tree.add_edge(2.into(), 3.into(), 1.into());
        tree.add_edge(5.into(), 6.into(), 6.into());

        let (mst, total_cost) = prims_tree(&tree);

        assert_eq!(total_cost, tree.total_weight());
        assert_eq!(mst.total_weight(), tree.total_weight());
        assert_eq!(mst.n(), tree.n());
    }
}
