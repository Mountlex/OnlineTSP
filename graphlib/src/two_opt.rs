use crate::{metric::Metric, Cost, Node, Tour};

use rand::seq::SliceRandom;

#[derive(Debug, Clone, PartialEq)]
pub enum TwoOptType {
    Increase,
    Decrease,
}

pub struct TwoOpt<'b, M> {
    current_tour: Vec<Node>,
    current_cost: Cost,
    start_cost: Cost,
    iteration: usize,
    metric: &'b M,
    n: usize,
    alg_type: TwoOptType,
}

impl<'b, M> TwoOpt<'b, M>
where
    M: Metric,
{
    pub fn with_sp_cache(n: usize, start_tour: Tour, metric: &'b M, alg_type: TwoOptType) -> Self {
        TwoOpt {
            current_tour: start_tour.nodes().to_vec(),
            current_cost: start_tour.cost(),
            start_cost: start_tour.cost(),
            iteration: 0,
            metric,
            alg_type,
            n,
        }
    }

    pub fn current_tour(&self) -> Tour {
        Tour::new(self.current_tour.clone(), self.current_cost)
    }

    pub fn run_for(&mut self, max_iterations: usize) -> Tour {
        log::info!("Starting 2-OPT. Initial tour cost: {}", self.current_cost);
        self.iteration = 0;

        let mut rng = rand::thread_rng();

        'iter: while self.iteration < max_iterations {
            self.iteration += 1;
            let log_msg = format!(
                "Iteration {}/{}: {}",
                self.iteration, max_iterations, self.current_cost
            );
            match self.alg_type {
                TwoOptType::Decrease => log::info!("{}", log_msg),
                TwoOptType::Increase => log::trace!("{}", log_msg),
            }

            let mut indices: Vec<usize> = (1..self.n - 1).collect();
            if TwoOptType::Decrease == self.alg_type {
                indices.shuffle(&mut rng)
            }
            for i in indices {
                for k in i + 2..self.n {
                    two_opt_swap(&mut self.current_tour, i, k);
                    let cost = self.tour_cost(&self.current_tour);
                    match self.alg_type {
                        TwoOptType::Increase => {
                            if cost > self.current_cost {
                                self.current_cost = cost;
                                continue 'iter;
                            } else {
                                two_opt_swap(&mut self.current_tour, i, k);
                            }
                        }
                        TwoOptType::Decrease => {
                            if cost < self.current_cost {
                                self.current_cost = cost;
                                continue 'iter;
                            } else {
                                two_opt_swap(&mut self.current_tour, i, k);
                            }
                        }
                    }
                }
            }
            log::info!("Stopping 2-OPT early: No improvement could be made.");
            break 'iter;
        }
        log::info!(
            "2-OPT results after {} iterations: {} -> {}",
            self.iteration,
            self.start_cost,
            self.current_cost
        );
        self.current_tour()
    }

    fn tour_cost(&self, nodes: &[Node]) -> Cost {
        let mut cost = Cost::new(0);
        for e in nodes.windows(2) {
            cost += self.metric.distance(e[0], e[1]);
        }
        cost
    }
}

fn two_opt_swap(nodes: &mut [Node], i: usize, k: usize) {
    let (_, second) = nodes.split_at_mut(i);
    let (middle, _) = second.split_at_mut(k - i);
    middle.reverse();
}

#[cfg(test)]
mod test_twp_opt {
    use super::*;

    #[test]
    fn test_swap() {
        let mut tour: Vec<Node> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]
            .into_iter()
            .map(|n| Node::new(n))
            .collect();
        two_opt_swap(&mut tour, 2, 6);
        assert_eq!(
            tour,
            vec![1, 2, 6, 5, 4, 3, 7, 8, 9]
                .into_iter()
                .map(|n| Node::new(n))
                .collect::<Vec<Node>>()
        )
    }
}
