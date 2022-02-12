use graphlib::{
    tsp::{self, SolutionType, TimedTour},
    AdjListGraph, Cost, Metric, Node, Nodes,
    SpMetricGraph, mst,
};
use rustc_hash::FxHashMap;


use crate::instance::{Instance, NodeRequest, Request};

pub struct Environment<G, R> {
    base_graph: G,
    metric_graph: SpMetricGraph,
    current_nodes: Vec<Node>,
    time: usize,
    origin: Node,
    pos: Node,
    instance: Instance<R>,
    known_requests: Vec<Node>,
    next_release: Option<usize>,
}

impl<R> Environment<AdjListGraph, R>
where
    R: Clone + Request,
{
    pub fn init(
        base_graph: &AdjListGraph,
        metric_graph: &SpMetricGraph,
        mut instance: Instance<R>,
        origin: Node,
    ) -> Self {
        let (known_requests, next_release) = instance.released_at(0);

        let mut nodes = known_requests.clone();
        nodes.push(origin);

        Self {
            base_graph: base_graph.clone(),
            metric_graph: metric_graph.clone(),
            current_nodes: nodes,
            time: 0,
            origin,
            pos: origin,
            instance,
            known_requests,
            next_release,
        }
    }

    fn follow_tour(&mut self, tour: TimedTour) {
        assert_eq!(Some(&self.pos), tour.nodes().first());

        for (edge, source_wait) in tour.nodes().windows(2).zip(tour.waiting_until()) {
            if let Some(source_wait) = source_wait {
                // wait until source_wait before we traverse edge, if this time has not passed yet.
                self.time = self.time.max(*source_wait);
            }
            let length = self.metric_graph.distance(edge[0], edge[1]);

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
        assert!(
            self.metric_graph
                .distance(self.origin, self.pos)
                .get_usize()
                + self.time
                <= time_back
        );

        // nodes that we visited until its release date in tour
        let mut served_nodes: Vec<Node> = vec![];

        for (edge, source_wait) in tour.nodes().windows(2).zip(tour.waiting_until()) {
            let start_time = if let Some(source_wait) = source_wait {
                // wait until source_wait before we traverse edge, if this time has not passed yet.
                self.time.max(*source_wait)
            } else {
                self.time
            };

            let length = self.metric_graph.distance(edge[0], edge[1]);
            if self.metric_graph.distance(self.origin, edge[1]).get_usize()
                + start_time
                + length.get_usize()
                <= time_back
            {
                served_nodes.push(edge[0]);
                self.pos = edge[1];
                self.time = start_time + length.get_usize();
            } else {
                // move back to the origin
                let back_distance = self
                    .metric_graph
                    .distance(self.origin, self.pos)
                    .get_usize();

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
        assert_eq!(Some(&self.pos), tour.nodes().first());

        // nodes that we visited until its release date in tour
        let mut served_nodes: Vec<Node> = vec![];

        for (edge, source_wait) in tour.nodes().windows(2).zip(tour.waiting_until()) {
            if let Some(source_wait) = source_wait {
                if let Some(next_release) = self.next_release {
                    if *source_wait > next_release {
                        self.time = self.time.max(*source_wait);
                        return served_nodes;
                    }
                }
                // wait until source_wait before we traverse edge, if this time has not passed yet.
                self.time = self.time.max(*source_wait);
            }
            let length = self.metric_graph.distance(edge[0], edge[1]).get_usize();
            served_nodes.push(edge[0]);
            if let Some(next_release) = self.next_release {
                if self.time + length > next_release {
                    if length == 0 {
                        self.time = next_release;
                        return served_nodes;
                    }
                    if self.time == next_release {
                        return served_nodes;
                    }
                    // we don't reach edge[1]
                    //log::trace!("Split edge {}-{} at {}, next_release={}, time={}", edge[0], edge[1], next_release - self.time, next_release, self.time);
                    self.pos = self.metric_graph.split_virtual_edge(
                        edge[0],
                        edge[1],
                        Cost::new(next_release - self.time),
                        &mut self.base_graph,
                    );
                    assert!(!self.current_nodes.contains(&self.pos));
                    self.current_nodes.push(self.pos);

                    self.time = next_release;
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
    loop {
        assert_eq!(env.origin, env.pos);
        let tour_graph = SpMetricGraph::from_metric_on_nodes(
            env.current_nodes.clone(),
            env.metric_graph.metric_clone(),
        );
        let tour = tsp::tsp_tour(&tour_graph, env.origin, sol_type);
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
                env.time = back;
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

pub fn replan(
    env: &mut Environment<AdjListGraph, NodeRequest>,
    back_until: Option<usize>,
    sol_type: SolutionType,
) -> usize {
    loop {
        let start_time = env.time;

        log::info!("Replan: compute tsp path from {} to origin", env.pos);
        let tour_graph = SpMetricGraph::from_metric_on_nodes(
            env.current_nodes.clone(),
            env.metric_graph.metric_clone(),
        );
        let tour = tsp::tsp_path(&tour_graph, env.pos, env.origin, sol_type);
        log::info!("Replan: current tour = {:?}", tour);

        if let Some(back) = back_until {
            // here is a bug
            if env.next_release.is_none() || env.next_release.unwrap() > back {
                log::info!("Replan: follow tour and be back by {}", back);
                let (abort, served) =
                    env.follow_tour_and_be_back_by(TimedTour::from_tour(tour), back);
                env.remove_served_requests(&served);
                if abort {
                    // if we cancel the tour
                    return env.time;
                } else {
                    // we reached origin without cancel -> wait until back_until
                    env.time = back;
                    return back;
                }
            }
        }
        log::info!("Replan: follow tour until next release date");
        let served = env.follow_tour_until_next_release(TimedTour::from_tour(tour));
        env.remove_served_requests(&served);

        if env.next_release.is_none() {
            return env.time;
        }

        // wait until next release
        let wait_until = env.time.max(env.next_release.unwrap());
        if let Some(back) = back_until {
            if wait_until > back {
                env.time = back;
                return back;
            }
        }
        if wait_until > env.time {
            log::info!("Replan: wait until {}", wait_until);
        }
        env.time = wait_until;

        let (new_requests, next_release) = env.instance.released_between(start_time, env.time);
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
    let mut instances_nodes: Vec<Node> = env.metric_graph.nodes().collect();
    instances_nodes.append(&mut pred_nodes);
    instances_nodes.sort();
    instances_nodes.dedup();
    let metric_graph = SpMetricGraph::from_metric_on_nodes(instances_nodes, env.metric_graph.metric_clone());
    env.metric_graph = metric_graph;

    let opt_pred = prediction
        .optimal_solution(env.origin, env.metric_graph.metric_ref(), sol_type)
        .0
        .as_float();
    let back_until = (opt_pred * alpha).floor() as usize;

    log::info!("Predict-Replan: execute replan until time {}.", back_until);
    ignore(env, Some(back_until), sol_type);
    assert_eq!(env.pos, env.origin);
    assert!(env.time <= back_until);

    let mut release_dates: FxHashMap<Node, usize> = prediction.release_dates();
    for node in env.metric_graph.nodes() {
        if !release_dates.contains_key(&node) && node != env.origin {
            release_dates.insert(node, 0);
        }
    }

    // some predicted requests may already be served, but we dont care?
    env.add_requests(prediction.nodes());

    if let Some(next_release) = env.next_release {
        if next_release < env.time {
            let (new_requests, next_release) = env.instance.released_between(next_release, env.time);
            env.add_requests(new_requests);
            env.next_release = next_release;
        }
    }


    log::info!("Predict-Replan: Start phase (iii) at time {}; next release {:?}", env.time, env.next_release);

    // Phase (iii)
    loop {
        let start_time = env.time;

        let tour_graph = SpMetricGraph::from_metric_on_nodes(
            env.current_nodes.clone(),
            env.metric_graph.metric_clone(),
        );

        // update release date w.r.t. current time
        let updated_release_dates: FxHashMap<Node, usize> = release_dates.iter().map(|(n, r)| (*n, (*r as i64 - start_time as i64).max(0)as usize ) ).collect();

        let max_rd: usize = updated_release_dates.values().copied().max().unwrap();
        let max_t = mst::prims_cost(&tour_graph).get_usize() * 2 + max_rd;
        let (_, tour) = tsp::tsp_rd_path(
            &tour_graph,
            &updated_release_dates,
            env.pos,
            env.origin,
            max_t,
            sol_type,
        );
        log::info!("Predict-Replan: current tour = {}", tour);

        let mut r = env.next_release;
        'req_search: while let Some(next_release) = r {
            let (reqs, r_new) = env.instance.released_at(next_release);
            for req in reqs {
                let mut on_tour = false;
                for (tour_node, tour_wait) in tour.nodes().iter().zip(tour.waiting_until()) {
                    if *tour_node == req {
                        on_tour = true;
                    }
                    // is this right?
                    if *tour_node == req && tour_wait.is_some() && tour_wait.unwrap() < next_release {
                        log::info!("Predict-Replan: req {} on current tour, but release later {} > {}", req, next_release, tour_wait.unwrap());
                        break 'req_search;
                    }
                }
                if !on_tour {
                    log::info!("Predict-Replan: req {} not found on current tour!", req);
                    break 'req_search;
                }
            }
            r = r_new;
        }
        env.next_release = r;

        log::info!("Predict-Replan: follow tour until next release date");
        let served = env.follow_tour_until_next_release(tour);
        env.remove_served_requests(&served);

        if env.next_release.is_none() {
            return env.time;
        }

        let (new_requests, next_release) = env.instance.released_between(start_time, env.time);
        env.add_requests(new_requests);
        env.next_release = next_release;
    }
    
}

#[cfg(test)]
mod test_algorithms {
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
        let mut env = Environment::init(&graph, &metric_graph, instance, 1.into());

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
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = ignore(&mut env, None, SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = ignore(&mut env, Some(0), SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = ignore(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(2, 0), (5, 0), (6, 1), (3, 1)].into();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = ignore(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 10);

        let instance: Instance<NodeRequest> = vec![(2, 11)].into();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
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
        let mut env = Environment::init(&graph, &metric_graph, instance, 1.into());

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

        let instance = Instance::<NodeRequest>::default();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = replan(&mut env, None, SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = replan(&mut env, Some(0), SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(3, 0), (6, 0), (2, 1), (5, 1)].into();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = replan(&mut env, Some(0), SolutionType::Optimal);
        assert_eq!(cost, 0);

        let instance: Instance<NodeRequest> = vec![(2, 0), (5, 0), (6, 1), (3, 1)].into();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = replan(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 10);

        let instance: Instance<NodeRequest> = vec![(2, 11)].into();
        let mut env = Environment::init(&graph, &metric_graph, instance.clone(), 1.into());
        let cost = replan(&mut env, Some(10), SolutionType::Optimal);
        assert_eq!(cost, 10);
    }
}
