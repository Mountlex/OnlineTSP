use graphlib::{sp::ShortestPathsCache, Cost, Metric, Node};

use rand_distr::{Distribution, Normal};

use crate::Instance;

pub fn gaussian_prediction(
    instance: &Instance,
    metric: &ShortestPathsCache,
    nodes: &[Node],
    sigma: f64,
    length_sigma: Option<f64>,
) -> Instance {
    let mut rng = rand::thread_rng();

    let _n_pred = if let Some(l_sigma) = length_sigma {
        let dist = Normal::new(instance.len() as f64, l_sigma).unwrap();
        let n = dist.sample(&mut rng).round();
        (n.round()).max(1.0) as usize
    } else {
        instance.len()
    };

    if sigma == 0.0 {
        return instance.clone();
    }

    // TODO different sizes
    let dist = Normal::new(0.0, sigma).unwrap();
    let preds: Vec<(Node, usize)> = instance
        .reqs()
        .iter()
        .map(|(x, t)| {
            let pred_t = (*t as f64 + dist.sample(&mut rng)).max(0.0).round() as usize;

            // TODO
            let pred_dist = dist.sample(&mut rng).abs();
            let mut distances: Vec<(Node, Cost)> = nodes
                .iter()
                .map(|&n| (n, metric.distance(n, *x)))
                .collect();
            distances.sort_by_key(|(_, d)| *d);

            let mut pred_n = distances.last().unwrap().0;
            for (n, d) in distances {
                if d.as_float() + 0.01 >= pred_dist {
                    pred_n = n;
                    break;
                }
            }
            (pred_n, pred_t)
        })
        .collect();
    preds.into()
}
