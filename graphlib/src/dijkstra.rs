use std::cmp::Ordering;

use compare::Compare;
use fibheap::FibHeap;

use crate::{cost::Cost, Graph, Node, NodeIndex, NodeSet};
use binary_heap_plus::BinaryHeap;

struct PrioComp<'a> {
    index: &'a NodeIndex,
    costs: Vec<Cost>,
}

impl<'a> PrioComp<'a> {
    fn new(index: &'a NodeIndex, costs: Vec<Cost>) -> Self {
        Self { index, costs }
    }
}

impl<'a> Compare<Node> for PrioComp<'a> {
    fn compare(&self, l: &Node, r: &Node) -> Ordering {
        let li = self.index[l];
        let ri = self.index[r];
        self.costs[ri].cmp(&self.costs[li]).then(r.cmp(l))
    }
}

pub fn dijkstra<'a, G>(graph: &'a G, n1: Node, n2: Node) -> Cost
where
    G: Graph<'a>,
{
    shortest_paths_to(graph, n1, &vec![n2]).cost_to(n2).unwrap()
}

pub fn dijkstra_path<'a, G>(graph: &'a G, n1: Node, n2: Node) -> (Cost, Vec<Node>)
where
    G: Graph<'a>,
{
    let paths = shortest_paths_to(graph, n1, &vec![n2]);
    (paths.cost_to(n2).unwrap(), paths.path_to(n2).unwrap())
}

// pub fn path_to(parent_map: &HashMap<Node, Node>, n1: Node, n2: Node) -> Vec<Node> {
//     let mut path = vec![n2];
//     let mut n = n2;
//     while n != n1 {
//         n = parent_map[&n];
//         path.push(n);
//     }
//     path.reverse();
//     path
// }

pub fn shortest_paths_to<'a, G>(graph: &'a G, node: Node, goals: &[Node]) -> Paths
where
    G: Graph<'a>,
{
    let n = graph.n();
    if n == 0 {
        return Paths::empty(node);
    }

    let mut to_visit = None;
    if goals.len() != n {
        to_visit = Some(goals.into_iter().copied().collect::<NodeSet>())
    }
    let mut costs = vec![Cost::max(); n];
    let mut prev: Vec<Option<Node>> = vec![None; n];

    let nodes_vec: Vec<Node> = graph.nodes().collect::<Vec<Node>>();
    let index = NodeIndex::init(&nodes_vec);

    costs[index[&node]] = Cost::new(0);

    let mut heap = BinaryHeap::from_vec_cmp(nodes_vec, PrioComp::new(&index, costs.clone()));
    while let Some(u) = heap.pop() {
        if let Some(to_visit) = to_visit.as_mut() {
            to_visit.remove(u);
            if to_visit.len() == 0 {
                break;
            }
        }

        for edge in graph.adjacent(u) {
            let update = costs[index[&u]] + edge.cost();
            let dist_v = costs.get_mut(index[&edge.sink()]).unwrap();
            if update < *dist_v {
                *dist_v = update;
                prev[index[&edge.sink()]] = Some(u);
            }
        }
        heap.replace_cmp(PrioComp::new(&index, costs.clone()));
    }

    Paths {
        node,
        index,
        costs,
        prev,
    }
}

pub fn shortest_paths_to_fib<'a, G>(graph: &'a G, node: Node, goals: &[Node]) -> Paths
where
    G: Graph<'a>,
{
    let n = graph.n();
    if n == 0 {
        return Paths::empty(node);
    }

    let mut to_visit = None;
    if goals.len() != n {
        to_visit = Some(goals.into_iter().copied().collect::<NodeSet>())
    }
    let mut costs = vec![Cost::max(); n];
    let mut prev: Vec<Option<Node>> = vec![None; n];

    let nodes_vec: Vec<Node> = graph.nodes().collect::<Vec<Node>>();
    let index = NodeIndex::init(&nodes_vec);

    costs[index[&node]] = Cost::new(0);

    let mut heap = FibHeap::from_vec(nodes_vec.into_iter().map(|n| (n, Cost::max())).collect());
    heap.decrease_key(&node, 0.into());

    while let Some(u) = heap.pop_min() {
        if let Some(to_visit) = to_visit.as_mut() {
            to_visit.remove(u);
            if to_visit.len() == 0 {
                break;
            }
        }

        for edge in graph.adjacent(u) {
            let update = costs[index[&u]] + edge.cost();
            let dist_v = costs.get_mut(index[&edge.sink()]).unwrap();
            if update < *dist_v {
                *dist_v = update;
                heap.decrease_key(&edge.sink(), update);
                prev[index[&edge.sink()]] = Some(u);
            }
        }
    }

    Paths {
        node,
        index,
        costs,
        prev,
    }
}

pub struct Paths {
    node: Node,
    index: NodeIndex,
    costs: Vec<Cost>,
    prev: Vec<Option<Node>>,
}

impl Paths {
    fn empty(node: Node) -> Self {
        Self {
            node,
            index: NodeIndex::empty(),
            costs: vec![],
            prev: vec![],
        }
    }

    pub fn cost_to(&self, n2: Node) -> Option<Cost> {
        self.index.get(&n2).map(|idx| self.costs[idx])
    }

    pub fn path_to(&self, n2: Node) -> Option<Vec<Node>> {
        self.index.get(&n2).and_then(|_| {
            let mut path = vec![n2];
            let mut n = n2;
            while n != self.node {
                if let Some(next) = self.prev[self.index[&n]] {
                    n = next;
                    path.push(n);
                } else {
                    return None;
                }
            }
            path.reverse();
            Some(path)
        })
    }
}

#[cfg(test)]
mod test_dijkstra {
    use super::*;
    use crate::AdjListGraph;

    ///   1 --6-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    #[test]
    fn test_dijkstra() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 6.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        let paths = shortest_paths_to(&graph, 1.into(), &vec![2.into(), 3.into()]);

        assert_eq!(paths.cost_to(2.into()), Some(5.into()));
        assert_eq!(paths.cost_to(3.into()), Some(6.into()));
    }

    ///  https://de.wikipedia.org/wiki/Algorithmus_von_Christofides
    ///
    ///  1 ------- 2
    ///  |  \   /  |
    ///  |    3    |
    ///  |  /   \  |
    ///  4 ------- 5
    #[test]
    fn test_dijkstra2() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(1.into(), 3.into(), 1.into());
        graph.add_edge(1.into(), 4.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 1.into());
        graph.add_edge(3.into(), 5.into(), 1.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 4.into(), 2.into());
        graph.add_edge(1.into(), 5.into(), 2.into());

        let paths = shortest_paths_to(&graph, 4.into(), &vec![1.into(), 5.into()]);

        assert_eq!(paths.cost_to(5.into()), Some(1.into()));
        assert_eq!(paths.cost_to(1.into()), Some(1.into()));
    }

    #[test]
    fn test_dijkstra_fib() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 6.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        let paths = shortest_paths_to_fib(&graph, 1.into(), &vec![2.into(), 3.into()]);

        assert_eq!(paths.cost_to(2.into()), Some(5.into()));
        assert_eq!(paths.cost_to(3.into()), Some(6.into()));
    }

    ///  https://de.wikipedia.org/wiki/Algorithmus_von_Christofides
    ///
    ///  1 ------- 2
    ///  |  \   /  |
    ///  |    3    |
    ///  |  /   \  |
    ///  4 ------- 5
    #[test]
    fn test_dijkstra_fib2() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(1.into(), 3.into(), 1.into());
        graph.add_edge(1.into(), 4.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 1.into());
        graph.add_edge(3.into(), 5.into(), 1.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 4.into(), 2.into());
        graph.add_edge(1.into(), 5.into(), 2.into());

        let paths = shortest_paths_to_fib(&graph, 4.into(), &vec![1.into(), 5.into()]);

        assert_eq!(paths.cost_to(5.into()), Some(1.into()));
        assert_eq!(paths.cost_to(1.into()), Some(1.into()));
    }
}
