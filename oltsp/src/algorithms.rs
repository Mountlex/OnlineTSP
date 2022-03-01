use graphlib::{
    mst,
    sp::{DistanceCache, ShortestPathsCache},
    tsp::{self, SolutionType, TimedTour},
    AdjListGraph, Cost, Metric, MetricView, Node,
};
use rustc_hash::FxHashMap;

use crate::instance::Instance;

pub struct Environment<'a, G> {
    base_graph: G,
    metric: &'a ShortestPathsCache,
    open_requests: Vec<Node>,
    time: usize,
    origin: Node,
    pos: Node,
    instance: Instance,
    next_release: Option<usize>,
    virtual_node: Option<Node>,
    buffer: Option<DistanceCache>,
}

impl<'a> Environment<'a, AdjListGraph> {
    pub fn init(
        base_graph: &AdjListGraph,
        metric: &'a ShortestPathsCache,
        mut instance: Instance,
        origin: Node,
    ) -> Self {
        let (known_requests, next_release) = instance.released_at(0);

        Self {
            base_graph: base_graph.clone(),
            metric,
            open_requests: known_requests,
            time: 0,
            origin,
            pos: origin,
            instance,
            next_release,
            virtual_node: None,
            buffer: None,
        }
    }

    fn current_nodes(&self) -> Vec<Node> {
        let mut req_nodes: Vec<Node> = self.open_requests.clone();
        req_nodes.push(self.origin);
        req_nodes.push(self.pos);
        req_nodes.sort();
        req_nodes.dedup();
        req_nodes
    }

    fn follow_tour(&mut self, tour: TimedTour) {
        assert_eq!(Some(&self.pos), tour.nodes().first());

        for (edge, source_wait) in tour.nodes().windows(2).zip(tour.waiting_until()) {
            if let Some(source_wait) = source_wait {
                // wait until source_wait before we traverse edge, if this time has not passed yet.
                self.time = self.time.max(*source_wait);
            }
            let length = self.metric.distance(edge[0], edge[1]);

            self.pos = edge[1];
            self.time += length.get_usize();
        }
    }

    fn follow_tour_and_be_back_by(
        &mut self,
        tour_graph: &MetricView<ShortestPathsCache>,
        tour: TimedTour,
        time_back: usize,
    ) -> (bool, Vec<Node>) {
        assert_eq!(Some(&self.pos), tour.nodes().first());
        assert!(tour_graph.distance(self.origin, self.pos).get_usize() + self.time <= time_back);

        // nodes that we visited until its release date in tour
        let mut served_nodes: Vec<Node> = vec![];

        for (edge, source_wait) in tour.nodes().windows(2).zip(tour.waiting_until()) {
            let distance_back = tour_graph.distance(self.origin, edge[1]).get_usize();
            if let Some(source_wait) = source_wait {
                if *source_wait + distance_back > time_back {
                    self.time = time_back;
                    self.pos = self.origin;
                    return (true, served_nodes);
                }

                // wait until source_wait before we traverse edge, if this time has not passed yet.
                self.time = self.time.max(*source_wait);
            }
            let length = tour_graph.distance(edge[0], edge[1]).get_usize();
            // we leave edge[0]
            served_nodes.push(edge[0]);

            // we cannot reach edge[1]
            if self.time + length + distance_back > time_back {
                self.pos = self.origin;
                served_nodes.push(self.pos);
                self.time += distance_back;
                return (true, served_nodes);
            } else {
                self.time += length;
                self.pos = edge[1];
            }
        }

        served_nodes.push(self.pos);
        (false, served_nodes)
    }

    fn follow_tour_until_time(
        &mut self,
        tour_graph: &MetricView<ShortestPathsCache>,
        tour: TimedTour,
        until_time: Option<usize>,
    ) -> Vec<Node> {
        assert_eq!(Some(&self.pos), tour.nodes().first());
        assert!(until_time.is_none() || self.time <= until_time.unwrap());

        // nodes that we visited until its release date in tour
        let mut served_nodes: Vec<Node> = vec![];

        for (edge, source_wait) in tour.nodes().windows(2).zip(tour.waiting_until()) {
            if let Some(source_wait) = source_wait {
                if let Some(until_time) = until_time {
                    if *source_wait > until_time {
                        self.time = until_time;
                        return served_nodes;
                    }
                }

                // wait until source_wait before we traverse edge, if this time has not passed yet.
                self.time = self.time.max(*source_wait);
            }
            let length = tour_graph.distance(edge[0], edge[1]).get_usize();
            // we leave edge[0]
            served_nodes.push(edge[0]);

            if let Some(until_time) = until_time {
                // we cannot reach edge[1]
                if self.time + length > until_time {
                    if self.time == until_time {
                        return served_nodes;
                    }

                    assert!(self.time <= until_time);
                    assert!(tour_graph.distance(self.origin, edge[0]).get_usize() <= self.time);

                    let (pos, buffer) = tour_graph.split_virtual_edge(
                        edge[0],
                        edge[1],
                        Cost::new(until_time - self.time),
                        &mut self.base_graph,
                    );
                    if buffer.is_some() {
                        assert!(
                            buffer.as_ref().unwrap().get(self.origin).get_usize() <= until_time
                        );
                        self.virtual_node = Some(pos);
                        self.buffer = buffer;
                    }

                    self.pos = pos;
                    self.time = until_time;
                    return served_nodes;
                }
            }
            self.pos = edge[1];
            self.time += length;
        }

        served_nodes.push(self.pos);
        served_nodes
    }

    fn add_requests(&mut self, mut nodes: Vec<Node>) {
        self.open_requests.append(&mut nodes);
        self.open_requests.sort();
        self.open_requests.dedup();
    }

    fn remove_served_requests(&mut self, nodes: &[Node]) {
        self.open_requests.retain(|n| !nodes.contains(n))
    }
}

pub fn ignore(
    env: &mut Environment<AdjListGraph>,
    back_until: Option<usize>,
    sol_type: SolutionType,
) -> usize {
    log::info!("======== Starting IGNORE");
    loop {
        assert_eq!(env.origin, env.pos);
        let tour_graph = MetricView::from_metric_on_nodes(
            env.current_nodes(),
            env.metric,
            env.virtual_node,
            env.buffer.clone(),
        );
        let (_, tour) = tsp::tsp_tour(&tour_graph, env.origin, sol_type);
        let start_time = env.time;
        if let Some(back) = back_until {
            let (abort, served) =
                env.follow_tour_and_be_back_by(&tour_graph, TimedTour::from_tour(tour), back);
            env.remove_served_requests(&served);
            if abort {
                return env.time;
            }
        } else {
            env.follow_tour(TimedTour::from_tour(tour.clone()));
            env.remove_served_requests(&tour);
        }

        if env.next_release.is_none() && env.open_requests.is_empty() {
            return env.time;
        }

        // wait until next release
        let wait_until = env.time.max(env.next_release.unwrap());
        if let Some(back) = back_until {
            if wait_until > back {
                env.time = back.max(env.time);
                return back;
            }
        }
        env.time = wait_until;

        // get released requests
        let (new_requests, next_release) = env.instance.released_between(start_time, env.time);
        env.add_requests(new_requests);
        env.next_release = next_release;
    }
}

pub fn smartstart(
    env: &mut Environment<AdjListGraph>,
    back_until: Option<usize>,
    sol_type: SolutionType,
) -> usize {
    log::info!("======== Starting SMARTSTART");
    loop {
        assert_eq!(env.origin, env.pos);
        let tour_graph = MetricView::from_metric_on_nodes(
            env.current_nodes(),
            env.metric,
            env.virtual_node,
            env.buffer.clone(),
        );
        let (cost, tour) = tsp::tsp_tour(&tour_graph, env.origin, sol_type);

        let start_time = env.time;
        if cost.get_usize() <= env.time {
            // work

            if let Some(back) = back_until {
                if env.time + cost.get_usize() > back {
                    return env.time;
                }
            }
            env.follow_tour(TimedTour::from_tour(tour.clone()));
            assert_eq!(env.time, start_time + cost.get_usize());
            assert_eq!(env.pos, env.origin);
            env.remove_served_requests(&tour);
        } else {
            // sleep

            let sleep_until = cost.get_usize();

            if let Some(back) = back_until {
                if back <= sleep_until {
                    env.time = back;
                    return back;
                }
            }
            env.time = sleep_until;
        }

        if env.next_release.is_none() && env.open_requests.is_empty() {
            return env.time;
        }

        if env.next_release.is_some() {
            // wait until next release
            let wait_until = env.time.max(env.next_release.unwrap());
            if let Some(back) = back_until {
                if wait_until > back {
                    env.time = back;
                    return env.time;
                }
            }
            env.time = wait_until;

            // get released requests
            let (new_requests, next_release) = env.instance.released_between(start_time, env.time);
            env.add_requests(new_requests);
            env.next_release = next_release;
        }
    }
}

pub fn replan(
    env: &mut Environment<AdjListGraph>,
    back_until: Option<usize>,
    sol_type: SolutionType,
) -> usize {
    log::info!("======== Starting REPLAN");

    loop {
        let start_time = env.time;

        log::info!("Replan: compute tsp path from {} to origin", env.pos);
        let tour_graph = MetricView::from_metric_on_nodes(
            env.current_nodes(),
            env.metric,
            env.virtual_node,
            env.buffer.clone(),
        );
        let (_, tour) = tsp::tsp_path(&tour_graph, env.pos, env.origin, sol_type);
        log::info!("Replan: current tour = {:?}", tour);

        if let Some(back_until) = back_until {
            assert!(env.time + tour_graph.distance(env.pos, env.origin).get_usize() <= back_until);
        }

        // nodes that we visited until its release date in tour
        let mut served_nodes: Vec<Node> = vec![];

        'tour_loop: for edge in tour.windows(2) {
            let length = tour_graph.distance(edge[0], edge[1]).get_usize();
            // we leave edge[0]
            served_nodes.push(edge[0]);

            if let Some(back_until) = back_until {
                assert!(
                    env.time + tour_graph.distance(env.origin, edge[0]).get_usize() <= back_until
                );
                if tour_graph.distance(env.origin, edge[1]).get_usize() + length + env.time
                    > back_until
                {
                    env.pos = env.origin;
                    env.time += tour_graph.distance(env.origin, edge[0]).get_usize();
                    served_nodes.push(env.pos);
                    env.remove_served_requests(&served_nodes);
                    return env.time;
                }
            }

            if let Some(next_release) = env.next_release {
                // we cannot reach edge[1]
                if env.time + length > next_release {
                    if env.time == next_release {
                        break 'tour_loop;
                    }

                    assert!(env.time <= next_release);
                    assert!(tour_graph.distance(env.origin, edge[0]).get_usize() <= env.time);

                    let (pos, buffer) = tour_graph.split_virtual_edge(
                        edge[0],
                        edge[1],
                        Cost::new(next_release - env.time),
                        &mut env.base_graph,
                    );
                    if buffer.is_some() {
                        assert!(
                            buffer.as_ref().unwrap().get(env.origin).get_usize() <= next_release
                        );
                        env.virtual_node = Some(pos);
                        env.buffer = buffer;
                    }

                    env.pos = pos;
                    env.time = next_release;
                    break 'tour_loop;
                }
            }
            env.pos = edge[1];
            env.time += length;
        }

        served_nodes.push(env.pos);
        env.remove_served_requests(&served_nodes);

        if env.next_release.is_none() {
            assert_eq!(env.pos, env.origin);
            assert!(env.open_requests.is_empty());
            return env.time;
        }

        // wait until next release
        if let Some(next_release) = env.next_release {
            if let Some(back_until) = back_until {
                if back_until < next_release {
                    return back_until;
                }
            }
            assert!(env.time == next_release || env.pos == env.origin);
            env.time = env.time.max(next_release)
        }

        let (new_requests, next_release) = env.instance.released_between(start_time, env.time);
        log::info!(
            "Replan: released nodes at time {} are {:?}",
            env.time,
            new_requests
        );
        env.add_requests(new_requests);
        env.next_release = next_release;
    }
}

fn predict_replan(
    env: &mut Environment<AdjListGraph>,
    prediction: Instance,
    sol_type: SolutionType,
) -> usize {
    let release_dates: FxHashMap<Node, usize> = prediction.release_dates();

    // only consider predicted requests that are released after now
    let mut open_preds: Vec<Node> = prediction
        .distinct_nodes()
        .into_iter()
        .filter(|n| release_dates[n] > env.time)
        .collect();

    // If the next release date is already due, get new requests
    if let Some(next_release) = env.next_release {
        if next_release <= env.time {
            let (new_requests, next_release) =
                env.instance.released_between(next_release, env.time);
            env.add_requests(new_requests);
            env.next_release = next_release;
        }
    }

    log::info!(
        "Predict-Replan: Start phase (iii) at time {}; next release {:?}",
        env.time,
        env.next_release
    );

    // Compute interesting time points
    let mut time_points: Vec<usize> = open_preds.iter().map(|n| release_dates[n]).collect();
    time_points.append(
        &mut env
            .instance
            .release_dates()
            .values()
            .filter(|&r| *r > env.time)
            .copied()
            .collect::<Vec<usize>>(),
    );
    time_points.sort();
    time_points.dedup();

    if env.time == 0 {
        for &pred in &open_preds {
            for (req, r) in env.instance.reqs() {
                if pred == *req && *r == release_dates[&pred] {
                    time_points.retain(|t| t != r);
                }
            }
        }
    }

    //assert!((time_points.is_empty() && env.next_release.is_none()) || (time_points[0] > env.time));

    let mut i = 0;
    let start_phase_three = env.time;

    // Phase (iii)
    loop {
        let start_time = env.time;

        // current metric graph
        let mut tour_nodes = [env.current_nodes(), open_preds.clone()].concat();
        log::info!(
            "Predict-Replan: replanning at time {}. pos = {}, tour nodes = {:?}",
            env.time,
            env.pos,
            tour_nodes
        );
        tour_nodes.sort();
        tour_nodes.dedup();
        let tour_graph = MetricView::from_metric_on_nodes(
            tour_nodes,
            env.metric,
            env.virtual_node,
            env.buffer.clone(),
        );
        assert!(
            tour_graph.distance(env.origin, env.pos).get_usize() <= env.time - start_phase_three
        );

        // compute tour
        // update release date w.r.t. current time
        let updated_release_dates: FxHashMap<Node, usize> = release_dates
            .iter()
            .map(|(n, r)| (*n, (*r as i64 - start_time as i64).max(0) as usize))
            .collect();
        let max_rd: usize = updated_release_dates
            .values()
            .copied()
            .max()
            .unwrap_or_else(|| 0);
        let max_t = mst::prims_cost(&tour_graph).get_usize() * 2 + max_rd;
        let (_, tour) = tsp::tsp_rd_path(
            // in this tour, the algorithm only waits until a requests arrives at the current point
            &tour_graph,
            &updated_release_dates,
            env.pos,
            env.origin,
            max_t,
            sol_type,
        );
        log::info!("Predict-Replan: current tour = {}", tour);

        // follow tour
        if i < time_points.len() {
            log::info!("Predict-Replan: follow tour until time {}", time_points[i]);
            let served = env.follow_tour_until_time(&tour_graph, tour, Some(time_points[i]));
            i += 1;
            env.remove_served_requests(&served);
        } else {
            log::info!("Predict-Replan: follow tour until origin");
            env.follow_tour_until_time(&tour_graph, tour, None);
            return env.time;
        }

        // Checks
        assert!(env.time <= time_points[i - 1]);
        let mut tour_nodes = [env.current_nodes(), open_preds.clone()].concat();
        tour_nodes.sort();
        tour_nodes.dedup();
        let tour_graph = MetricView::from_metric_on_nodes(
            tour_nodes,
            env.metric,
            env.virtual_node,
            env.buffer.clone(),
        );
        assert!(
            tour_graph.distance(env.origin, env.pos).get_usize() <= env.time - start_phase_three
        );

        if env.next_release.is_none() && env.open_requests.is_empty() {
            if env.pos == env.origin {
                return env.time;
            } else {
                return env.time + tour_graph.distance(env.pos, env.origin).get_usize();
            }
        }

        // Only consider future predictions
        open_preds.retain(|n| release_dates[n] > env.time);

        if open_preds.is_empty() && env.next_release.is_some() {
            // wait until next release
            env.time = env.time.max(env.next_release.unwrap());
        }

        // Add released requests
        let (new_requests, next_release) = env.instance.released_between(start_time, env.time);
        log::info!("New requests: {:?}", new_requests);
        env.add_requests(new_requests);
        env.next_release = next_release;
    }
}

pub fn smart_trust(
    env: &mut Environment<AdjListGraph>,
    alpha: f64,
    prediction: Instance,
    sol_type: SolutionType,
) -> usize {
    log::info!("======== Starting Predict-Replan with alpha = {}.", &alpha);

    // Phase (i)
    let opt_pred = prediction
        .optimal_solution(env.origin, env.metric, sol_type)
        .0
        .as_float();
    let back_until = (opt_pred * alpha).floor() as usize;

    log::info!(
        "Predict-Replan: execute SMARTSTART until time {}.",
        back_until
    );
    smartstart(env, Some(back_until), sol_type);

    assert_eq!(env.pos, env.origin);
    assert!(env.time <= back_until);

    // If all actual requests are served
    if env.next_release.is_none() && env.open_requests.is_empty() {
        return env.time;
    }

    // Phase (ii)
    env.time = env.time.max((opt_pred * alpha / 2.0).floor() as usize);

    // Phase (iii)
    predict_replan(env, prediction, sol_type)
}

pub fn delayed_trust(
    env: &mut Environment<AdjListGraph>,
    alpha: f64,
    prediction: Instance,
    sol_type: SolutionType,
) -> usize {
    log::info!("======== Starting Predict-Replan with alpha = {}.", &alpha);

    // Phase (i)
    let opt_pred = prediction
        .optimal_solution(env.origin, env.metric, sol_type)
        .0
        .as_float();
    let back_until = (opt_pred * alpha).floor() as usize;

    log::info!(
        "Predict-Replan: execute SMARTSTART until time {}.",
        back_until
    );
    replan(env, Some(back_until), sol_type);

    assert_eq!(env.pos, env.origin);
    assert!(env.time <= back_until);

    // If all actual requests are served
    if env.next_release.is_none() && env.open_requests.is_empty() {
        return env.time;
    }

    // Phase (iii)
    predict_replan(env, prediction, sol_type)
}

#[cfg(test)]
mod test_algorithms {
    use graphlib::SpMetricGraph;

    use super::*;

    #[test]
    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn test_ignore() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());
        let metric_graph = SpMetricGraph::from_graph(&graph);

        let instance = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(&graph, metric_graph.metric_ref(), instance, 1.into());

        let cost = ignore(&mut env, None, SolutionType::Optimal);
        assert_eq!(cost, 18 + 10)
    }

    #[test]
    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn test_ignore_1() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());
        let metric_graph = SpMetricGraph::from_graph(&graph);

        let instance = Instance::default();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, None, SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, Some(0), SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance = vec![(2, 0), (5, 0), (6, 1), (3, 1)].into();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 10);

        let instance: Instance = vec![(2, 11)].into();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 10);
    }

    #[test]
    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn test_replan() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());
        let metric_graph = SpMetricGraph::from_graph(&graph);

        let instance = vec![(3, 0), (6, 0), (2, 7), (5, 7)].into();
        let mut env = Environment::init(&graph, metric_graph.metric_ref(), instance, 1.into());

        let cost = replan(&mut env, None, SolutionType::Optimal);
        assert_eq!(cost, 7 + 11)
    }

    #[test]
    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn test_replan_1() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());
        let metric_graph = SpMetricGraph::from_graph(&graph);

        let instance = Instance::default();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = replan(&mut env, None, SolutionType::Optimal);
        assert_eq!(cost, 0);
    }
}
