use std::{error::Error, path::PathBuf};

use graphlib::{
    mst,
    tsp::{tsp_rd_tour, SolutionType, TimedTour},
    Adjacency, Cost, Edge, GraphSize, Metric, MetricGraph, Node, Nodes, SpMetricGraph, Weighted,
};
use rustc_hash::FxHashMap;

pub trait Request {
    fn node(&self) -> Node;
}

#[derive(Clone, Debug, Default)]
pub struct NodeRequest(Node);

impl From<usize> for NodeRequest {
    fn from(id: usize) -> Self {
        NodeRequest(id.into())
    }
}

impl Request for NodeRequest {
    fn node(&self) -> Node {
        self.0
    }
}

#[derive(Default, Clone, Debug)]
pub struct Instance<R> {
    requests: Vec<(R, usize)>,
}

impl From<Vec<(usize, usize)>> for Instance<NodeRequest> {
    fn from(raw: Vec<(usize, usize)>) -> Self {
        raw.into_iter()
            .map(|(x, t)| (x.into(), t.into()))
            .collect::<Vec<(NodeRequest, usize)>>()
            .into()
    }
}

impl<R> From<Vec<(R, usize)>> for Instance<R> {
    fn from(reqs: Vec<(R, usize)>) -> Self {
        let mut time_sorted = reqs;
        time_sorted.sort_by_key(|(_, t)| *t);
        Self {
            requests: time_sorted,
        }
    }
}

impl<R> Instance<R>
where
    R: Clone + Request,
{
    pub fn nodes(&self) -> Vec<Node> {
        self.requests.iter().map(|(r, _)| r.node()).collect()
    }

    pub fn distinct_nodes(&self) -> Vec<Node> {
        let mut nodes = self.nodes();
        nodes.sort();
        nodes.dedup();
        nodes
    }

    pub fn release_dates(&self) -> FxHashMap<Node, usize> {
        self.requests.iter().map(|(r, t)| (r.node(), *t)).collect()
    }

    /// Returns the set of requests that are released at times t as well as the next release date if there is such
    pub fn released_at(&mut self, t: usize) -> (Vec<Node>, Option<usize>) {
        let mut released: Vec<Node> = vec![];
        let mut i = 0;
        while i < self.requests.len() {
            if self.requests[i].1 == t {
                released.push(self.requests[i].0.clone().node())
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
                released.push(self.requests[i].0.clone().node())
            }
            if self.requests[i].1 > r {
                return (released, Some(self.requests[i].1));
            }
            i += 1;
        }

        return (released, None);
    }

    pub fn optimal_solution(
        &self,
        start_node: Node,
        metric_graph: &SpMetricGraph,
        sol_type: SolutionType,
    ) -> (Cost, TimedTour) {
        let req_nodes = (0..(self.requests.len() + 1))
            .map(|id| Node::new(id))
            .collect::<Vec<Node>>();
        let mut req_to_graph_nodes = vec![start_node];

        let (nodes_raw, mut release_dates_raw): (Vec<R>, Vec<usize>) =
            self.requests.iter().cloned().unzip();
        let mut release_dates = vec![0];
        release_dates.append(&mut release_dates_raw);

        let mut nodes = nodes_raw.into_iter().map(|r| r.node()).collect();
        req_to_graph_nodes.append(&mut nodes);

        let mut graph_nodes = req_to_graph_nodes.clone();
        graph_nodes.sort_by_key(|n| n.id());
        graph_nodes.dedup();

        let max_rd: usize = *release_dates.iter().max().unwrap();
        let max_t = mst::prims_cost(metric_graph).get_usize() * 2 + max_rd;

        let instance_graph = InstanceGraph {
            req_nodes,
            req_to_graph_nodes,
            metric_graph,
        };

        let release_map: FxHashMap<Node, usize> = release_dates
            .into_iter()
            .enumerate()
            .map(|(n, r)| (Node::new(n), r))
            .collect();
        let (obj, mut tour) =
            tsp_rd_tour(&instance_graph, &release_map, Node::new(0), max_t, sol_type);
        *tour.mut_nodes() = tour
            .nodes()
            .into_iter()
            .map(|n| instance_graph.req_to_graph_nodes[n.id()])
            .collect::<Vec<Node>>();
        (obj, tour)
    }

    pub fn lower_bound(&self, start_node: Node, metric_graph: &SpMetricGraph) -> Cost {
        let (approx, _) = self.optimal_solution(start_node, metric_graph, SolutionType::Approx);
        approx / 2.5
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

pub fn instance_from_file(filename: &PathBuf) -> Result<Instance<NodeRequest>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(filename)?;
    let mut requests: Vec<(usize, usize)> = vec![];
    for record in reader.records() {
        let record = record?;
        let x: usize = record[0].parse()?;
        let t: usize = record[1].parse()?;
        requests.push((x, t))
    }
    let instance: Instance<NodeRequest> = requests.into();
    Ok(instance)
}

#[cfg(test)]
mod test_instance {
    use graphlib::AdjListGraph;

    use super::*;

    #[test]
    ///   1 --1-- 2 --1-- 3 --1-- 4
    fn test_optimal_solution() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 1.into());
        let metric_graph = SpMetricGraph::from_graph(&graph);

        let instance: Instance<NodeRequest> = vec![(4, 1), (2, 7)].into();

        let (obj, tour) = instance.optimal_solution(1.into(), &metric_graph, SolutionType::Optimal);

        assert_eq!(tour.nodes(), vec![1.into(), 4.into(), 2.into(), 1.into()]);
        assert_eq!(
            tour.waiting_until(),
            vec![0.into(), 3.into(), 7.into(), 8.into()]
        );
        assert_eq!(obj.get_usize(), 8);
    }
}
