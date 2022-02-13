use std::{path::PathBuf, io::{BufReader, BufWriter}, fs::File, ops::{Div, Mul}};
use anyhow::Result;
use ndarray::{Array2, ArrayView, NdProducer, Axis, Array1};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Serialize, Deserialize};

use crate::{
    dijkstra::shortest_paths_to, Cost, Graph, Node, NodeIndex, Nodes, ShortestPaths, Weighted,
};

impl ShortestPaths for ShortestPathsCache {
    fn shortest_path_cost(&self, n1: Node, n2: Node) -> Cost {
        self.get(n1, n2)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum PathCost {
    Unreachable,
    Path(Cost),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortestPathsCache {
    matrix: Array2<PathCost>,
    index: NodeIndex,
    node_in_buffer: Option<Node>,
    buffer: Option<Array1<PathCost>>
}

impl ShortestPathsCache {
    pub fn scale(&mut self, scale: usize) {
        self.matrix.map_inplace(|entry| *entry = match entry {
            PathCost::Unreachable => PathCost::Unreachable,
            PathCost::Path(cost) => PathCost::Path(*cost / scale)
        })
    }

    pub fn empty<'a, G>(graph: &'a G) -> Self
    where
        G: Nodes<'a>,
    {
        let mut nodes = graph.nodes().collect::<Vec<Node>>();
        nodes.sort();
        let n = nodes.len();
        let mut d = Array2::from_elem((n, n), PathCost::Unreachable);
        for i in 0..n {
            d[[i, i]] = PathCost::Path(0.into());
        }

        ShortestPathsCache {
            matrix: d,
            index: NodeIndex::init(&nodes),
            node_in_buffer: None,
            buffer: None,
        }
    }

    pub fn add_node<'a, G>(&mut self, new_node: Node, graph: &'a G)
    where
        G: Graph<'a>,
    {
        let goals = graph
            .nodes()
            .filter(|node| self.index.get(node).is_some())
            .collect::<Vec<Node>>();
        let idx = self.index.add_node(new_node);
        self.matrix
            .push_row(ArrayView::from(&vec![PathCost::Unreachable; idx]))
            .unwrap();
        self.matrix
            .push_column(ArrayView::from(&vec![PathCost::Unreachable; idx + 1]))
            .unwrap();
        assert!(self.matrix.is_square());

        self.set(new_node, new_node, Cost::new(0));
        let paths = shortest_paths_to(graph, new_node, &goals);
        for node in goals {
            self.set(new_node, node, paths.cost_to(node).unwrap());
        }
    }

    pub fn split_edge_to_buffer<'a, G>(&mut self, new_node: Node, source: Node, sink: Node, at: Cost, edge_cost: Cost, graph: &'a G)
    where
        G: Graph<'a>,
    {
        let goals = graph
            .nodes()
            .filter(|node| self.index.get(node).is_some())
            .collect::<Vec<Node>>();
        let idx = self.index.add_node(new_node);
        self.matrix
            .push_row(ArrayView::from(&vec![PathCost::Unreachable; idx]))
            .unwrap();
        self.matrix
            .push_column(ArrayView::from(&vec![PathCost::Unreachable; idx + 1]))
            .unwrap();
        assert!(self.matrix.is_square());

        self.set(new_node, new_node, Cost::new(0));       
        for node in goals {
            let p1 = self.get(node, source) + at;
            let p2 = self.get(node, sink) + edge_cost - at;
            self.set(new_node, node, Cost::new(p1.get_usize().min(p2.get_usize())));
        }
    }

    pub fn compute_all_graph_pairs<'a, G>(graph: &'a G) -> Self
    where
        G: Graph<'a>,
    {
        Self::compute_all_pairs(graph.nodes().collect(), graph)
    }

    pub fn compute_all_pairs<'a, G>(nodes: Vec<Node>, graph: &'a G) -> Self
    where
        G: Graph<'a>,
    {
        log::info!("Starting to compute all pair shortest paths.");

        let mut sorted_nodes = nodes;
        sorted_nodes.sort();
        let n = sorted_nodes.len();
        let mut d = Array2::from_elem((n, n), PathCost::Unreachable);
        for i in 0..n {
            d[[i, i]] = PathCost::Path(0.into());
        }

        let mut sp = ShortestPathsCache {
            matrix: d,
            index: NodeIndex::init(&sorted_nodes),
            buffer: None,
            node_in_buffer: None,
        };

        for (i, &n1) in sorted_nodes.iter().enumerate() {
            log::trace!("Start node {}/{}", i, n);

            let goals: &[Node] = sorted_nodes.split_at(i).1;
            let paths = shortest_paths_to(graph, n1, &goals);
            for &n in goals {
                sp.set(n1, n, paths.cost_to(n).unwrap());
            }
        }

        log::info!("Finished computing all pair shortest paths!");

        sp
    }



    pub fn compute_all_graph_pairs_par<'a, G>(graph: &'a G) -> Self
    where
        G: Graph<'a> + Sync,
    {
        Self::compute_all_pairs_par(graph.nodes().collect(), graph)
    }

    pub fn compute_all_pairs_par<'a, G>(nodes: Vec<Node>, graph: &'a G) -> Self
    where
        G: Graph<'a> + Sync,
    {
        log::info!(
            "Starting to compute all pair shortest paths for {} nodes (parallel).",
            nodes.len()
        );

        let mut sorted_nodes: Vec<Node> = nodes;
        sorted_nodes.sort();
        let n = sorted_nodes.len();
        let mut d = Array2::from_elem((n, n), PathCost::Unreachable);
        for i in 0..n {
            d[[i, i]] = PathCost::Path(0.into());
        }

        let mut sp = ShortestPathsCache {
            matrix: d,
            index: NodeIndex::init(&sorted_nodes),
            node_in_buffer : None,
            buffer: None
        };

        let indexed_nodes: Vec<(usize, Node)> = sorted_nodes.iter().copied().enumerate().collect();
        let costs = indexed_nodes
            .into_par_iter()
            .map(|(i, n)| {
                log::trace!("Start node {}/{}", i, n);
                let goals: &[Node] = sorted_nodes.split_at(i).1;
                let paths = shortest_paths_to(graph, n, &goals);
                (
                    i,
                    n,
                    goals
                        .into_iter()
                        .map(|g| paths.cost_to(*g).unwrap())
                        .collect::<Vec<Cost>>(),
                )
            })
            .collect::<Vec<(usize, Node, Vec<Cost>)>>();

        for (i, n1, c) in costs {
            let goals: &[Node] = sorted_nodes.split_at(i).1;
            for (n, cost) in goals.into_iter().zip(c) {
                sp.set(n1, *n, cost);
            }
        }
        log::info!("Finished computing all pair shortest paths!");

        sp
    }

    pub fn get_or_compute<'a, G>(&mut self, n1: Node, n2: Node, graph: &'a G) -> Cost
    where
        G: Graph<'a> + Weighted,
    {
        let i1 = self.index[&n1];
        let i2 = self.index[&n2];

        let x = i1.min(i2);
        let y = i1.max(i2);
        if let PathCost::Path(cost) = self.matrix[[x, y]] {
            cost
        } else if let Some(cost) = graph.edge_cost(n1, n2) {
            self.matrix[[x, y]] = PathCost::Path(cost);
            cost
        } else {
            let goals: Vec<Node> = graph.nodes().collect();
            let paths = shortest_paths_to(graph, n1, &goals);
            for n in goals {
                self.set(n1, n, paths.cost_to(n).unwrap());
            }
            if let PathCost::Path(cost) = self.matrix[[x, y]] {
                cost
            } else {
                panic!("Should not happen")
            }
        }
    }

    pub fn contains_node(&self, node: Node) -> bool {
        self.index.get(&node).is_some()
    }

    pub fn get(&self, n1: Node, n2: Node) -> Cost {
        let i1 = self.index[&n1];
        let i2 = self.index[&n2];

        self.get_by_index(i1, i2)
    }

    fn get_by_index(&self, i1: usize, i2: usize) -> Cost {
        let x = i1.min(i2);
        let y = i1.max(i2);
        if let PathCost::Path(cost) = self.matrix[[x, y]] {
            cost
        } else {
            panic!("No path known!")
        }
    }

    pub fn set(&mut self, n1: Node, n2: Node, cost: Cost) {
        let i1 = self.index[&n1];
        let i2 = self.index[&n2];

        self.set_by_index(i1, i2, cost);
    }

    fn set_by_index(&mut self, i1: usize, i2: usize, cost: Cost) {
        let x = i1.min(i2);
        let y = i1.max(i2);

        self.matrix[[x, y]] = PathCost::Path(cost);
    }
}



pub fn load_or_compute<'a, G>(path: &PathBuf, graph: &'a G, scale: usize,) -> Result<ShortestPathsCache>
where
    G: Graph<'a> + Sync, {
        if path.is_file() {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let mut sp: ShortestPathsCache = bincode::deserialize_from(reader)?;
            sp.scale(scale);
            Ok(sp)
        } else {
            let sp = ShortestPathsCache::compute_all_graph_pairs_par(graph);
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            bincode::serialize_into(writer, &sp)?;
            Ok(sp)
        }
    }

