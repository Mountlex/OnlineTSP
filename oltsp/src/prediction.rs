use graphlib::{sp::ShortestPathsCache, Cost, Metric, Node};

use rand::seq::{IteratorRandom, SliceRandom};
use rand_distr::{Distribution, Normal};

use crate::Instance;

pub fn gaussian_prediction(
    instance: &Instance,
    metric: &ShortestPathsCache,
    nodes: &[Node],
    sigma: f64,
    rel_num_pred: f64,
    correct_rd: bool,
    correct_loc: bool,
) -> Instance {
    let mut rng = rand::thread_rng();

    let n_pred = (instance.len() as f64 * rel_num_pred).floor() as usize;

    if sigma == 0.0 {
        let reqs: Vec<(Node, usize)> = instance
            .reqs()
            .choose_multiple(&mut rng, n_pred)
            .copied()
            .collect();
        return reqs.into();
    }

    let dist = Normal::new(0.0, sigma).unwrap();
    let preds: Vec<(Node, usize)> = instance
        .reqs()
        .iter()
        .map(|(x, t)| {
            let pred_t = if correct_rd {
                *t
            } else {
                (*t as f64 + dist.sample(&mut rng)).max(0.0).round() as usize
            };

            // TODO
            let pred_dist = dist.sample(&mut rng).abs();
            let mut distances: Vec<(Node, Cost)> =
                nodes.iter().map(|&n| (n, metric.distance(n, *x))).collect();
            distances.sort_by_key(|(_, d)| *d);

            let mut pred_n = distances.last().unwrap().0;
            for (n, d) in distances {
                if d.as_float() + 0.01 >= pred_dist {
                    pred_n = n;
                    break;
                }
            }

            if correct_loc {
                pred_n = *x;
            }

            (pred_n, pred_t)
        })
        .choose_multiple(&mut rand::thread_rng(), n_pred);
    preds.into()
}
