use graphlib::{
    mst,
    sp::{DistanceCache, ShortestPathsCache},
    tsp::{self, SolutionType, TimedTour},
    AdjListGraph, Cost, Metric, MetricView, Node,
};
use rustc_hash::FxHashMap;

use crate::instance::{Instance, NodeRequest, Request};

pub struct Environment<'a, G, R> {
    base_graph: G,
    metric: &'a ShortestPathsCache,
    current_nodes: Vec<Node>,
    time: usize,
    origin: Node,
    pos: Node,
    instance: Instance<R>,
    known_requests: Vec<Node>,
    next_release: Option<usize>,
    virtual_node: Option<Node>,
    buffer: Option<DistanceCache>,
}

impl<'a, R> Environment<'a, AdjListGraph, R>
where
    R: Clone + Request,
{
    pub fn init(
        base_graph: &AdjListGraph,
        metric: &'a ShortestPathsCache,
        mut instance: Instance<R>,
        origin: Node,
    ) -> Self {
        let (known_requests, next_release) = instance.released_at(0);

        let mut nodes = known_requests.clone();
        nodes.push(origin);

        Self {
            base_graph: base_graph.clone(),
            metric,
            current_nodes: nodes,
            time: 0,
            origin,
            pos: origin,
            instance,
            known_requests,
            next_release,
            virtual_node: None,
            buffer: None,
        }
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
        tour: TimedTour,
        time_back: usize,
    ) -> (bool, Vec<Node>) {
        assert_eq!(Some(&self.pos), tour.nodes().first());
        assert!(self.metric.distance(self.origin, self.pos).get_usize() + self.time <= time_back);

        // nodes that we visited until its release date in tour
        let mut served_nodes: Vec<Node> = vec![];

        for (edge, source_wait) in tour.nodes().windows(2).zip(tour.waiting_until()) {
            let start_time = if let Some(source_wait) = source_wait {
                // wait until source_wait before we traverse edge, if this time has not passed yet.
                self.time.max(*source_wait)
            } else {
                self.time
            };

            let length = self.metric.distance(edge[0], edge[1]);
            if self.metric.distance(self.origin, edge[1]).get_usize()
                + start_time
                + length.get_usize()
                <= time_back
            {
                served_nodes.push(edge[0]);
                self.pos = edge[1];
                self.time = start_time + length.get_usize();
            } else {
                // move back to the origin
                let back_distance = self.metric.distance(self.origin, self.pos).get_usize();

                if start_time + back_distance <= time_back {
                    // serve current request
                    served_nodes.push(self.pos);
                    self.time = start_time + back_distance
                } else {
                    self.time += back_distance;
                }
                self.pos = self.origin;
                return (true, served_nodes);
            }
        }
        (false, served_nodes)
    }

    fn follow_tour_until_next_release(&mut self, tour: TimedTour) -> Vec<Node> {
            self.follow_tour_until_time(tour, self.next_release)
        
    }

    fn follow_tour_until_time(&mut self, tour: TimedTour, until_time: Option<usize>) -> Vec<Node> {
        assert_eq!(Some(&self.pos), tour.nodes().first());

        // nodes that we visited until its release date in tour
        let mut served_nodes: Vec<Node> = vec![];

        let metric_graph = MetricView::from_metric_on_nodes(
            self.current_nodes.clone(),
            self.metric,
            self.virtual_node,
            self.buffer.clone(),
        );

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
            let length = metric_graph.distance(edge[0], edge[1]).get_usize();
            // we leave edge[0]
            served_nodes.push(edge[0]);

            if let Some(until_time) = until_time {
            // we cannot reach edge[1]
            if self.time + length > until_time {
                if length == 0 {
                    self.time = until_time;
                    return served_nodes;
                }
                if self.time == until_time {
                    return served_nodes;
                }
                // we don't reach edge[1]
                //log::trace!("Split edge {}-{} at {}, until_time={}, time={}", edge[0], edge[1], until_time - self.time, until_time, self.time);

                let metric_graph = MetricView::from_metric_on_nodes(
                    self.current_nodes.clone(),
                    self.metric,
                    self.virtual_node,
                    self.buffer.clone(),
                );

                let (pos, buffer) = metric_graph.split_virtual_edge(
                    edge[0],
                    edge[1],
                    Cost::new(until_time - self.time),
                    &mut self.base_graph,
                );
                if buffer.is_some() {
                    self.virtual_node = Some(pos);
                    self.buffer = buffer;
                }

                if !self.current_nodes.contains(&pos) {
                    self.current_nodes.push(pos);
                }
            
                self.pos = pos;
                self.time = until_time;
                return served_nodes;
            }
        }
            self.pos = edge[1];
            self.time += length;
        }

        served_nodes
    }

    fn add_requests(&mut self, mut nodes: Vec<Node>) {
        self.current_nodes.append(&mut nodes);
        self.current_nodes.sort();
        self.current_nodes.dedup();
        self.known_requests.append(&mut nodes);
    }

    fn remove_served_requests(&mut self, nodes: &[Node]) {
        self.current_nodes
            .retain(|n| *n == self.pos || *n == self.origin || !nodes.contains(n))
    }
}

pub fn ignore(
    env: &mut Environment<AdjListGraph, NodeRequest>,
    back_until: Option<usize>,
    sol_type: SolutionType,
) -> usize {
    log::info!("======== Starting IGNORE");
    loop {
        assert_eq!(env.origin, env.pos);
        let tour_graph = MetricView::from_metric_on_nodes(
            env.current_nodes.clone(),
            env.metric,
            env.virtual_node,
            env.buffer.clone(),
        );
        let (_, tour) = tsp::tsp_tour(&tour_graph, env.origin, sol_type);
        let start_time = env.time;
        if let Some(back) = back_until {
            let (abort, served) = env.follow_tour_and_be_back_by(TimedTour::from_tour(tour), back);
            env.remove_served_requests(&served);
            if abort {
                return env.time;
            }
        } else {
            env.follow_tour(TimedTour::from_tour(tour.clone()));
            env.remove_served_requests(&tour);
        }

        if env.next_release.is_none() {
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
    env: &mut Environment<AdjListGraph, NodeRequest>,
    back_until: Option<usize>,
    sol_type: SolutionType,
) -> usize {
    log::info!("======== Starting SMARTSTART");
    loop {
        assert_eq!(env.origin, env.pos);
        let tour_graph = MetricView::from_metric_on_nodes(
            env.current_nodes.clone(),
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

        if env.next_release.is_none()
            && env.current_nodes.len() == 1
            && env.current_nodes.first() == Some(&env.origin)
        {
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

pub fn replan(env: &mut Environment<AdjListGraph, NodeRequest>, sol_type: SolutionType) -> usize {
    log::info!("======== Starting REPLAN");
    loop {
        let start_time = env.time;

        log::info!("Replan: compute tsp path from {} to origin", env.pos);
        let tour_graph = MetricView::from_metric_on_nodes(
            env.current_nodes.clone(),
            env.metric,
            env.virtual_node,
            env.buffer.clone(),
        );
        let (_, tour) = tsp::tsp_path(&tour_graph, env.pos, env.origin, sol_type);
        log::info!("Replan: current tour = {:?}", tour);

        log::info!("Replan: follow tour until next release date");
        let served = env.follow_tour_until_next_release(TimedTour::from_tour(tour));
        env.remove_served_requests(&served);

        if env.next_release.is_none() {
            assert_eq!(vec![env.origin], env.current_nodes);
            return env.time;
        }

        // wait until next release
        let wait_until = env.time.max(env.next_release.unwrap());
        if wait_until > env.time {
            log::info!("Replan: wait until {}", wait_until);
        }
        env.time = wait_until;

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

pub fn learning_augmented(
    env: &mut Environment<AdjListGraph, NodeRequest>,
    alpha: f64,
    prediction: Instance<NodeRequest>,
    sol_type: SolutionType,
) -> usize {
    log::info!("======== Starting Predict-Replan with alpha = {}.", &alpha);

    let mut pred_nodes = prediction.nodes();
    let mut nodes: Vec<Node> = env.instance.nodes();
    nodes.append(&mut pred_nodes);
    nodes.sort();
    nodes.dedup();

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

    // Phase (ii)
    env.time = env.time.max((opt_pred * alpha / 2.0).floor() as usize);

    // Phase (iii)
    let release_dates: FxHashMap<Node, usize> = prediction.release_dates();

    // only consider predicted requests that are released after now
    let preds: Vec<Node> = prediction
        .distinct_nodes()
        .into_iter()
        .filter(|n| release_dates[n] > env.time)
        .collect();
    env.add_requests(preds.clone());

    // If the next release date is already due, get new requests
    if let Some(next_release) = env.next_release {
        if next_release < env.time {
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
    let mut time_points: Vec<usize> = preds.iter().map(|n| release_dates[n]).collect();
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

    let mut i = 0;
    let start_phase_three = env.time;

    // Phase (iii)
    loop {

        let start_time = env.time;

        let tour_graph = MetricView::from_metric_on_nodes(
            env.current_nodes.clone(),
            env.metric,
            env.virtual_node,
            env.buffer.clone(),
        );

        // update release date w.r.t. current time
        let updated_release_dates: FxHashMap<Node, usize> = release_dates
            .iter()
            .map(|(n, r)| (*n, (*r as i64 - start_time as i64).max(0) as usize))
            .collect();

        // compute tour
        let max_rd: usize = updated_release_dates.values().copied().max().unwrap();
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

        log::info!("Predict-Replan: follow tour until next import time");
        if i < time_points.len() {
            let served = env.follow_tour_until_time(tour, Some(time_points[i]));
            i += 1;
            env.remove_served_requests(&served);
        } else {
            env.follow_tour_until_time(tour, None);
            return env.time;
        }
        assert!(env.time <= time_points[i-1]);
        assert!(env.metric.distance(env.origin, env.pos).get_usize() <= env.time - start_phase_three);

        if env.pos == env.origin && env.next_release.is_none() {
            return env.time;
        }

        env.current_nodes.retain(|n| {
            *n == env.pos
                || *n == env.origin
                || !release_dates.contains_key(n)
                || release_dates[n] > env.time
        });

        // wait in origin until next release
        if env.current_nodes.len() == 1 {
            env.time = env.time.max(env.next_release.unwrap());
        }

        let (new_requests, next_release) = env.instance.released_between(start_time, env.time);
        env.add_requests(new_requests);
        env.next_release = next_release;
    }
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

        let instance = Instance::<NodeRequest>::default();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, None, SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, Some(0), SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(2, 0), (5, 0), (6, 1), (3, 1)].into();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = ignore(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 10);

        let instance: Instance<NodeRequest> = vec![(2, 11)].into();
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

        let cost = replan(&mut env, SolutionType::Optimal);
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

        let instance = Instance::<NodeRequest>::default();
        let mut env = Environment::init(
            &graph,
            metric_graph.metric_ref(),
            instance.clone(),
            1.into(),
        );
        let cost = replan(&mut env, SolutionType::Optimal);
        assert_eq!(cost, 0);
    }
}
