mod algorithms;
mod instance;
mod prediction;

use algorithms::Environment;
use graphlib::{tsp::SolutionType, AdjListGraph, Node, SpMetricGraph, sp::ShortestPathsCache};
pub use instance::{instance_from_file, Instance, NodeRequest};
pub use prediction::gaussian_prediction;

pub fn replan(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance<NodeRequest>,
    start_node: Node,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::replan(&mut env, sol_type)
}

pub fn ignore(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance<NodeRequest>,
    start_node: Node,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::ignore(&mut env, None, sol_type)
}

pub fn smartstart(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance<NodeRequest>,
    start_node: Node,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::smartstart(&mut env, None, sol_type)
}

pub fn learning_augmented(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance<NodeRequest>,
    start_node: Node,
    alpha: f64,
    prediction: Instance<NodeRequest>,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::learning_augmented(&mut env, alpha, prediction, sol_type)
}
