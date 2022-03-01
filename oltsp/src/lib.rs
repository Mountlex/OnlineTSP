mod algorithms;
mod instance;
mod prediction;

use algorithms::Environment;
use graphlib::{sp::ShortestPathsCache, tsp::SolutionType, AdjListGraph, Node};
pub use instance::{instance_from_file, Instance};
pub use prediction::gaussian_prediction;

pub fn replan(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance,
    start_node: Node,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::replan(&mut env, None, sol_type)
}

pub fn ignore(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance,
    start_node: Node,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::ignore(&mut env, None, sol_type)
}

pub fn smartstart(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance,
    start_node: Node,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::smartstart(&mut env, None, sol_type)
}

pub fn smart_trust(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance,
    start_node: Node,
    alpha: f64,
    prediction: Instance,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::smart_trust(&mut env, alpha, prediction, sol_type)
}

pub fn delayed_trust(
    graph: &AdjListGraph,
    metric: &ShortestPathsCache,
    instance: Instance,
    start_node: Node,
    alpha: f64,
    prediction: Instance,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric, instance, start_node);
    algorithms::delayed_trust(&mut env, alpha, prediction, sol_type)
}
