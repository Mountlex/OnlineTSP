use std::fmt::Display;

use good_lp::{
    default_solver, variable, variables, Constraint, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};
use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{
    christofides::{christofides, MatchingAlgorithm},
    Adjacency, Cost, GraphSize, Metric, Node, Nodes, Weighted,
};

#[derive(Clone, Copy, Debug)]
pub enum SolutionType {
    Optimal,
    Approx,
}

pub fn tsp_tour<'a, G>(graph: &'a G, start_node: Node, sol_type: SolutionType) -> (Cost, Vec<Node>)
where
    G: Nodes<'a> + Adjacency<'a> + Weighted + GraphSize + Metric,
{
    match sol_type {
        SolutionType::Optimal => tsp_path_optimal(graph, start_node, start_node),
        SolutionType::Approx => {
            christofides(graph, start_node, start_node, MatchingAlgorithm::Blossom)
        }
    }
}

pub fn tsp_path<'a, G>(
    graph: &'a G,
    start_node: Node,
    end_node: Node,
    sol_type: SolutionType,
) -> (Cost, Vec<Node>)
where
    G: Nodes<'a> + Adjacency<'a> + Weighted + GraphSize + Metric,
{
    match sol_type {
        SolutionType::Optimal => tsp_path_optimal(graph, start_node, end_node),
        SolutionType::Approx => {
            christofides(graph, start_node, end_node, MatchingAlgorithm::Blossom)
        }
    }
}

pub fn tsp_path_optimal<'a, G>(graph: &'a G, start_node: Node, end_node: Node) -> (Cost, Vec<Node>)
where
    G: Nodes<'a> + Weighted + GraphSize,
{
    let n = graph.n();
    if n == 1 {
        if start_node == end_node {
            return (Cost::new(0), vec![start_node, end_node]);
        } else {
            panic!("Cannot compute path between different vertices if n=1!")
        }
    }
    if n == 2 {
        if start_node != end_node {
            return (Cost::new(0), vec![start_node, end_node]);
        } else {
            let other = graph
                .nodes()
                .find(|&n| n != start_node && n != end_node)
                .unwrap();
            return (
                graph.edge_cost(start_node, other).unwrap()
                    + graph.edge_cost(end_node, other).unwrap(),
                vec![start_node, other, end_node],
            );
        }
    }

    let mut objective: Expression = 0.into();
    let mut constraints = Vec::<Constraint>::new();
    let mut vars = variables!();
    let var_by_edge = add_tour_variables_and_constraints(
        &mut constraints,
        &mut vars,
        graph,
        start_node,
        end_node,
    );

    for i in graph.nodes() {
        for j in graph.nodes() {
            if let Some(&var) = var_by_edge.get(&(i, j)) {
                if i == end_node && j == start_node {
                    objective += var * 0;
                } else if let Some(distance) = graph.edge_cost(i, j) {
                    objective += var * distance.as_float();
                }
            }
        }
    }

    let mut model = vars.minimise(objective.clone()).using(default_solver);
    for constr in constraints {
        model.add_constraint(constr);
    }
    let solution = model.solve().unwrap();
    let cost = solution.eval(&objective);

    let mut tour = Vec::with_capacity(n + 1);
    tour.push(start_node);
    let mut current = start_node;
    loop {
        for j in graph.nodes() {
            if current != j {
                if let Some(&var) = var_by_edge.get(&(current, j)) {
                    if solution.value(var).eq(&1.0) {
                        current = j;
                        break;
                    }
                }
            }
        }
        tour.push(current);
        if current == end_node {
            break;
        }
    }

    (Cost::new(cost.round() as usize), tour)
}

fn add_tour_variables_and_constraints<'a, G>(
    constraints: &mut Vec<Constraint>,
    vars: &mut ProblemVariables,
    graph: &'a G,
    start_node: Node,
    end_node: Node,
) -> FxHashMap<(Node, Node), Variable>
where
    G: Nodes<'a> + Weighted + GraphSize,
{
    let mut var_by_edge: FxHashMap<(Node, Node), Variable> = FxHashMap::default();

    for i in graph.nodes() {
        let mut one_succ: Expression = 0.into();
        for j in graph.nodes() {
            if i != j {
                if graph.edge_cost(i, j).is_some() || (i == end_node && j == start_node) {
                    let var = vars.add(variable().binary().name(format!("x_{}{}", i.id(), j.id())));
                    one_succ += var;
                    var_by_edge.insert((i, j), var);
                }
            }
        }

        constraints.push(one_succ.eq(1 as i32));
    }

    for j in graph.nodes() {
        let mut one_prec: Expression = 0.into();
        for i in graph.nodes() {
            if i != j {
                if let Some(var) = var_by_edge.get(&(i, j)) {
                    one_prec += var
                }
            }
        }
        constraints.push(one_prec.eq(1 as i32));
    }

    log::trace!("Adding subtour constraints");
    for subset in graph
        .nodes()
        .powerset()
        .filter(|set| set.len() > 1 && set.len() < graph.n())
    {
        let mut exp: Expression = 0.into();
        for &i in &subset {
            for &j in &subset {
                if i != j {
                    if let Some(var) = var_by_edge.get(&(i, j)) {
                        exp += var
                    }
                }
            }
        }
        constraints.push(exp.leq((subset.len() - 1) as i32));
    }

    var_by_edge
}

#[derive(Clone)]
pub struct TimedTour {
    nodes: Vec<Node>,
    waiting_until: Vec<Option<usize>>,
}

impl Display for TimedTour {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nodes = self.nodes.iter().zip(&self.waiting_until).fold(
            String::new(),
            |acc, (&x, &t)| match t {
                Some(r) => acc + &format!("({},{})", x, r) + ", ",
                None => acc + &format!("({},-)", x) + ", ",
            },
        );
        write!(f, "[{}]", nodes)
    }
}

impl TimedTour {
    pub fn from_tour(tour: Vec<Node>) -> Self {
        Self {
            waiting_until: vec![None; tour.len()],
            nodes: tour,
        }
    }

    pub fn from_tour_and_release_dates<M>(
        tour: Vec<Node>,
        release_dates: &FxHashMap<Node, usize>,
        metric: &M,
    ) -> (Cost, Self)
    where
        M: Metric,
    {
        let mut time = 0;
        let mut waiting_until: Vec<Option<usize>> = vec![];
        for edge in tour.windows(2) {
            if let Some(r) = release_dates.get(&edge[0]) {
                time = time.max(*r);
            }
            waiting_until.push(Some(time));
            time += metric.distance(edge[0], edge[1]).get_usize();
        }
        if let Some(r) = release_dates.get(tour.last().unwrap()) {
            time = time.max(*r);
        }
        waiting_until.push(Some(time));
        (
            Cost::new(time),
            Self {
                nodes: tour,
                waiting_until,
            },
        )
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn mut_nodes(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    pub fn waiting_until(&self) -> &[Option<usize>] {
        &self.waiting_until
    }

    pub fn start_node(&self) -> Node {
        self.nodes.first().unwrap().clone()
    }

    pub fn end_node(&self) -> Node {
        self.nodes.last().unwrap().clone()
    }

    pub fn finish_time<M>(&self, start_time: usize, metric: M) -> usize
    where
        M: Metric,
    {
        let mut time = start_time;
        for (edge, source_wait) in self.nodes.windows(2).zip(self.waiting_until.iter()) {
            if let Some(source_wait) = source_wait {
                // wait until source_wait before we traverse edge, if this time has not passed yet.
                time = time.max(*source_wait);
            }
            let length = metric.distance(edge[0], edge[1]).get_usize();
            time += length;
        }
        time
    }
}

/// start_node shall not have a release date
pub fn tsp_rd_tour<'a, G>(
    graph: &'a G,
    release_dates: &FxHashMap<Node, usize>,
    start_node: Node,
    max_t: usize,
    sol_type: SolutionType,
) -> (Cost, TimedTour)
where
    G: Nodes<'a> + Adjacency<'a> + Weighted + GraphSize + Metric,
{
    match sol_type {
        SolutionType::Optimal => {
            tsp_rd_optimal(graph, release_dates, start_node, start_node, max_t)
        }
        SolutionType::Approx => {
            let nodes = christofides(graph, start_node, start_node, MatchingAlgorithm::Blossom).1;
            TimedTour::from_tour_and_release_dates(nodes, release_dates, graph)
        }
    }
}

pub fn tsp_rd_path<'a, G>(
    graph: &'a G,
    release_dates: &FxHashMap<Node, usize>,
    start_node: Node,
    end_node: Node,
    max_t: usize,
    sol_type: SolutionType,
) -> (Cost, TimedTour)
where
    G: Nodes<'a> + Adjacency<'a> + Weighted + GraphSize + Metric,
{
    match sol_type {
        SolutionType::Optimal => tsp_rd_optimal(graph, release_dates, start_node, end_node, max_t),
        SolutionType::Approx => {
            let nodes = christofides(graph, start_node, end_node, MatchingAlgorithm::Blossom).1;
            TimedTour::from_tour_and_release_dates(nodes, release_dates, graph)
        }
    }
}

pub fn tsp_rd_optimal<'a, G>(
    graph: &'a G,
    release_dates: &FxHashMap<Node, usize>,
    start_node: Node,
    end_node: Node,
    max_t: usize,
) -> (Cost, TimedTour)
where
    G: Nodes<'a> + Adjacency<'a> + Weighted + GraphSize,
{
    let n = graph.n();

    log::info!(
        "Computing optimal TSP-rd tour on {} nodes (max_t = {})",
        n,
        max_t
    );

    println!(
        "{:?} {} {}",
        graph.nodes().collect::<Vec<Node>>(),
        start_node,
        max_t
    );

    let mut vars = variables!();
    let obj_var = vars.add(variable().name("T"));
    let mut y_by_edge: FxHashMap<(Node, usize), Variable> = FxHashMap::default();
    let mut constraints = Vec::<Constraint>::new();

    log::trace!("Adding tour constraints to ILP");
    let x_by_edge: FxHashMap<(Node, Node), Variable> = add_tour_variables_and_constraints(
        &mut constraints,
        &mut vars,
        graph,
        start_node,
        end_node,
    );

    log::trace!("Adding time variables to ILP");
    for i in graph.nodes() {
        for t in 0..max_t {
            if i == start_node || t >= release_dates.get(&i).copied().unwrap_or(0) {
                let var = vars.add(variable().binary().name(format!("y_{}{}", i.id(), t)));
                constraints.push((t as i32 * var).leq(obj_var));
                y_by_edge.insert((i, t), var);
            }
        }
    }

    constraints.push((1 * y_by_edge[&(start_node, 0)]).eq(1 as i32));

    log::trace!("Adding time constraints to ILP");
    for i in graph.nodes() {
        for j in graph.nodes() {
            if i != j {
                if let Some(distance) = graph.edge_cost(i, j) {
                    for t in 0..max_t {
                        if let Some(yit) = y_by_edge.get(&(i, t)) {
                            let x = x_by_edge.get(&(i, j)).unwrap();

                            let tj = release_dates
                                .get(&j)
                                .copied()
                                .unwrap_or(0)
                                .max(t + distance.get_usize());
                            if tj < max_t && !(i == start_node && t > 0) {
                                if let Some(yj) = y_by_edge.get(&(j, tj)) {
                                    constraints.push((1 * *x + 1 * *yit).leq(1 * *yj + 1 as i32));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut model = vars.minimise(obj_var).using(default_solver);
    for constr in constraints {
        model.add_constraint(constr);
    }

    log::trace!("Starting solver");
    let solution = model.solve().unwrap();

    log::trace!("Computing tour from ILP solution");
    let mut tour = Vec::with_capacity(n + 1);
    let mut waiting_until: Vec<Option<usize>> = Vec::with_capacity(n + 1);
    tour.push(start_node);
    waiting_until.push(Some(0));

    let mut current = start_node;
    let mut current_time: usize = 0;
    loop {
        'node_loop: for j in graph.nodes() {
            if current != j {
                if let Some(&var) = x_by_edge.get(&(current, j)) {
                    if solution.value(var) > 0.9 {
                        current = j;
                        'time_loop: for t in current_time..max_t {
                            if let Some(&y_var) = y_by_edge.get(&(j, t)) {
                                if solution.value(y_var) > 0.9 {
                                    current_time = t;
                                    waiting_until.push(Some(t));
                                    break 'time_loop;
                                }
                            }
                        }
                        break 'node_loop;
                    }
                }
            }
        }
        tour.push(current);
        if current == end_node {
            break;
        }
    }

    let timed_tour = TimedTour {
        nodes: tour,
        waiting_until,
    };

    (
        Cost::new(solution.eval(1 * obj_var).round() as usize),
        timed_tour,
    )
}

#[cfg(test)]
mod test_tsp {
    use crate::AdjListGraph;

    use super::*;

    fn get_graph1() -> AdjListGraph {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 4.into());
        graph.add_edge(1.into(), 3.into(), 1.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 3.into(), 2.into());
        graph.add_edge(2.into(), 4.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 5.into());

        graph
    }

    fn get_graph2() -> AdjListGraph {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 4.into());
        graph.add_edge(1.into(), 3.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(2.into(), 4.into(), 2.into());
        graph.add_edge(3.into(), 4.into(), 2.into());
        graph
    }

    #[test]
    fn test_tour_solution() {
        let graph = get_graph1();

        let tour = tsp_tour(&graph, 1.into(), SolutionType::Optimal).1;

        assert_eq!(tour, vec![1.into(), 3.into(), 2.into(), 4.into(), 1.into()])
    }

    #[test]
    fn test_path_solution() {
        let graph = get_graph2();
        let tour = tsp_path(&graph, 1.into(), 4.into(), SolutionType::Optimal).1;
        assert_eq!(tour, vec![1.into(), 3.into(), 2.into(), 4.into()])
    }

    #[test]
    fn test_rd_tour_solution1() {
        let graph = get_graph1();

        let mut release_map = FxHashMap::<Node, usize>::default();
        release_map.insert(1.into(), 3);
        release_map.insert(2.into(), 2);
        release_map.insert(3.into(), 7);
        release_map.insert(4.into(), 1);
        let (obj, tour) = tsp_rd_tour(&graph, &release_map, 1.into(), 15, SolutionType::Optimal);

        assert_eq!(
            tour.nodes(),
            vec![1.into(), 4.into(), 2.into(), 3.into(), 1.into()]
        );
        assert_eq!(
            tour.waiting_until(),
            vec![0.into(), 3.into(), 4.into(), 7.into(), 8.into()]
        );
        assert_eq!(obj, Cost::new(8));
    }

    #[test]
    fn test_rd_path_solution() {
        let mut graph = get_graph2();
        graph.add_edge(0.into(), 1.into(), Cost::new(0));
        graph.add_edge(0.into(), 2.into(), Cost::new(4));
        graph.add_edge(0.into(), 3.into(), Cost::new(1));

        let mut release_map = FxHashMap::<Node, usize>::default();
        release_map.insert(1.into(), 1);
        release_map.insert(2.into(), 0);
        release_map.insert(3.into(), 6);
        release_map.insert(4.into(), 0);
        let (obj, tour) = tsp_rd_path(
            &graph,
            &release_map,
            0.into(),
            4.into(),
            15,
            SolutionType::Optimal,
        );

        assert_eq!(
            tour.nodes(),
            vec![0.into(), 1.into(), 2.into(), 3.into(), 4.into()]
        );
        assert_eq!(
            tour.waiting_until(),
            vec![0.into(), 1.into(), 5.into(), 6.into(), 8.into()]
        );
        assert_eq!(obj, Cost::new(8));
    }
}
