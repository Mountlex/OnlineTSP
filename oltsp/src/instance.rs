use std::{error::Error, path::PathBuf};

use graphlib::{
    mst,
    tsp::{self, SolutionType, TimedTour},
    Adjacency, Cost, Edge, GraphSize, Metric, MetricGraph, MetricView, Node, Nodes, Weighted,
};
use rustc_hash::FxHashMap;
use std::fmt::Debug;

#[derive(Default, Clone, Debug)]
pub struct Instance {
    requests: Vec<(Node, usize)>,
}

impl From<Vec<(usize, usize)>> for Instance {
    fn from(raw: Vec<(usize, usize)>) -> Self {
        raw.into_iter()
            .map(|(x, t)| (x.into(), t.into()))
            .collect::<Vec<(Node, usize)>>()
            .into()
    }
}

impl From<Vec<(Node, usize)>> for Instance {
    fn from(reqs: Vec<(Node, usize)>) -> Self {
        let mut time_sorted = reqs;
        time_sorted.sort_by_key(|(_, t)| *t);
        Self {
            requests: time_sorted,
        }
    }
}

impl Instance {
    pub fn scale_rd(&mut self, scale: usize) {
        self.requests.iter_mut().for_each(|(_, r)| *r = *r * scale)
    }

    pub fn reqs(&self) -> &Vec<(Node, usize)> {
        &self.requests
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn nodes(&self) -> Vec<Node> {
        self.requests.iter().map(|(r, _)| *r).collect()
    }

    pub fn distinct_nodes(&self) -> Vec<Node> {
        let mut nodes = self.nodes();
        nodes.sort();
        nodes.dedup();
        nodes
    }

    pub fn release_dates(&self) -> FxHashMap<Node, usize> {
        self.requests.iter().map(|(r, t)| (*r, *t)).collect()
    }

    /// Returns the set of requests that are released at times t as well as the next release date if there is such
    pub fn released_at(&mut self, t: usize) -> (Vec<Node>, Option<usize>) {
        let mut released: Vec<Node> = vec![];
        let mut i = 0;
        while i < self.requests.len() {
            if self.requests[i].1 == t {
                released.push(self.requests[i].0.clone())
            }
            if self.requests[i].1 > t {
                return (released, Some(self.requests[i].1));
            }
            i += 1;
        }

        return (released, None);
    }

    /// return all request released at time t with l < t <= r
    pub fn released_between(&mut self, l: usize, r: usize) -> (Vec<Node>, Option<usize>) {
        assert!(l <= r);
        let mut released: Vec<Node> = vec![];
        let mut i = 0;
        while i < self.requests.len() {
            if l < self.requests[i].1 && self.requests[i].1 <= r {
                released.push(self.requests[i].0.clone())
            }
            if self.requests[i].1 > r {
                return (released, Some(self.requests[i].1));
            }
            i += 1;
        }

        return (released, None);
    }

    /// metric graph of instance nodes
    pub fn optimal_solution<M>(
        &self,
        start_node: Node,
        metric: &M,
        sol_type: SolutionType,
    ) -> (Cost, TimedTour)
    where
        M: Metric + Clone + Debug,
    {
        let mut distinct_release_dates = FxHashMap::<Node, usize>::default();
        for (n, r) in &self.requests {
            let entry = distinct_release_dates.entry(*n).or_insert(0);
            *entry = (*entry).max(*r);
        }
        let mut nodes = self.nodes();
        nodes.push(start_node);
        nodes.sort();
        nodes.dedup();

        let tour_graph = MetricView::from_metric_on_nodes(nodes, metric, None, None);

        let max_rd: usize = distinct_release_dates.values().copied().max().unwrap_or_else(|| 0);
        let max_t = mst::prims_cost(&tour_graph).get_usize() * 2 + max_rd;
        tsp::tsp_rd_tour(
            &tour_graph,
            &distinct_release_dates,
            start_node,
            max_t,
            sol_type,
        )
    }

    pub fn lower_bound<M>(&self, start_node: Node, metric: &M) -> usize where M: Metric + Clone + Debug {
        if self.requests.is_empty() {
            return 0
        }

        let (approx, _) = self.optimal_solution(start_node, metric, SolutionType::Approx);
        let lb1 = (approx.as_float() / 2.5).floor() as usize;
        let lb2 = self.requests.iter().map(|(node, r)| *r + metric.distance(start_node, *node).get_usize()).max().unwrap();

        lb1.max(lb2)
    }
}

struct InstanceGraph<'a, M> {
    /// Nodes representing requests
    req_nodes: Vec<Node>,

    /// Maps requests to graph nodes. graph_nodes[0] == start_node, length == num_requests + 1
    req_to_graph_nodes: Vec<Node>,

    metric_graph: &'a MetricGraph<M>,
}

impl<'a, M> Nodes<'a> for InstanceGraph<'_, M> {
    type NodeIter = <Vec<Node> as Nodes<'a>>::NodeIter;

    fn nodes(&'a self) -> Self::NodeIter {
        self.req_nodes.nodes()
    }
}

impl<M> Weighted for InstanceGraph<'_, M>
where
    M: Metric,
{
    fn edge_cost(&self, node1: Node, node2: Node) -> Option<Cost> {
        Some(self.metric_graph.distance(
            self.req_to_graph_nodes[node1.id()],
            self.req_to_graph_nodes[node2.id()],
        ))
    }
}

impl<M> Metric for InstanceGraph<'_, M>
where
    M: Metric,
{
    fn distance(&self, node1: Node, node2: Node) -> Cost {
        self.metric_graph.distance(
            self.req_to_graph_nodes[node1.id()],
            self.req_to_graph_nodes[node2.id()],
        )
    }
}

impl<M> GraphSize for InstanceGraph<'_, M> {
    fn n(&self) -> usize {
        self.req_nodes.len()
    }
}

pub struct AdjacencyIter<'a, M> {
    node: Node,
    graph: &'a InstanceGraph<'a, M>,
    node_iter: Option<<Vec<Node> as Nodes<'a>>::NodeIter>,
}

impl<'a, M> AdjacencyIter<'a, M> {
    fn new(
        node: Node,
        graph: &'a InstanceGraph<M>,
        node_iter: Option<<Vec<Node> as Nodes<'a>>::NodeIter>,
    ) -> Self {
        Self {
            node,
            graph,
            node_iter,
        }
    }
}

impl<'a, M> Iterator for AdjacencyIter<'a, M>
where
    M: Metric,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.node;
        self.node_iter.as_mut().and_then(|values| {
            values
                .next()
                .map(|n| Edge::new(node, n, self.graph.edge_cost(node, n).unwrap()))
        })
    }
}

impl<'a, M> Adjacency<'a> for InstanceGraph<'a, M>
where
    M: 'a + Metric,
{
    type AdjacencyIter = AdjacencyIter<'a, M>;

    fn adjacent(&'a self, node: Node) -> Self::AdjacencyIter {
        AdjacencyIter::new(node, self, Some(self.nodes()))
    }
}

pub fn instance_from_file(filename: &PathBuf) -> Result<Instance, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(filename)?;
    let mut requests: Vec<(usize, usize)> = vec![];
    for record in reader.records() {
        let record = record?;
        let x: usize = record[0].parse()?;
        let t: usize = record[1].parse()?;
        requests.push((x, t))
    }
    let instance: Instance = requests.into();
    Ok(instance)
}

#[cfg(test)]
mod test_instance {
    use graphlib::{sp::ShortestPathsCache, AdjListGraph};

    use super::*;

    #[test]
    ///   1 --1-- 2 --1-- 3 --1-- 4
    fn test_optimal_solution() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 1.into());
        let sp = ShortestPathsCache::compute_all_graph_pairs(&graph);

        let instance: Instance = vec![(4, 1), (2, 7)].into();

        let (obj, tour) = instance.optimal_solution(1.into(), &sp, SolutionType::Optimal);

        assert_eq!(tour.nodes(), vec![1.into(), 4.into(), 2.into(), 1.into()]);
        assert_eq!(
            tour.waiting_until(),
            vec![0.into(), 3.into(), 7.into(), 8.into()]
        );
        assert_eq!(obj.get_usize(), 8);
    }
}
