mod algorithms;
mod instance;
mod prediction;

use algorithms::Environment;
use graphlib::{tsp::SolutionType, AdjListGraph, Node, SpMetricGraph};
pub use instance::{instance_from_file, Instance, NodeRequest};

pub fn replan(
    graph: &AdjListGraph,
    metric_graph: &SpMetricGraph,
    instance: Instance<NodeRequest>,
    start_node: Node,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric_graph, instance, start_node);
    algorithms::replan(&mut env, None, sol_type)
}

pub fn ignore(
    graph: &AdjListGraph,
    metric_graph: &SpMetricGraph,
    instance: Instance<NodeRequest>,
    start_node: Node,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric_graph, instance, start_node);
    algorithms::ignore(&mut env, None, sol_type)
}

pub fn learning_augmented(
    graph: &AdjListGraph,
    metric_graph: &SpMetricGraph,
    instance: Instance<NodeRequest>,
    start_node: Node,
    alpha: f64,
    prediction: Instance<NodeRequest>,
    sol_type: SolutionType,
) -> usize {
    let mut env = Environment::init(graph, metric_graph, instance, start_node);
    algorithms::learning_augmented(&mut env, alpha, prediction, sol_type)
}
