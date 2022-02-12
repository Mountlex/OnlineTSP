use graphlib::{Graph, Metric, Node, sp::ShortestPathsCache, Cost};
use rand::prelude::SliceRandom;
use rand_distr::{Normal, Distribution};

use crate::{Instance, NodeRequest, instance::Request};

pub fn gaussian_prediction(
    instance: &Instance<NodeRequest>,
    metric: &ShortestPathsCache,
    nodes: &[Node],
    sigma: f64,
    length_sigma: Option<f64>,
) -> Instance<NodeRequest>
{
    let mut rng = rand::thread_rng();

    let n_pred = if let Some(l_sigma) = length_sigma {
        let dist = Normal::new(instance.len() as f64, l_sigma).unwrap();
        let n = dist.sample(&mut rng).round();
        (n.round()).max(1.0) as usize
    } else {
        instance.len()
    };

    // TODO different sizes
    let dist = Normal::new(0.0, sigma).unwrap();
    let preds: Vec<(NodeRequest, usize)> = instance.requests.iter().map(|(x,t)| {
        let pred_t = (*t as f64 + dist.sample(&mut rng)).max(0.0).round() as usize;
        
        // TODO
        let pred_dist = dist.sample(&mut rng).abs();
        let mut distances: Vec<(Node, Cost)> = nodes.iter().map(|&n| (n, metric.distance(n, x.node()))).collect();
        distances.sort_by_key(|(_,d)| *d);
        
        let mut pred_n: Option<NodeRequest> = None;
        for (n,d) in distances {
            if d.as_float() + 0.01 >= pred_dist {
                pred_n = Some(NodeRequest(n));
                break
            }
        }
        (pred_n.unwrap(), pred_t)
    }).collect();
    Instance { requests: preds }
}
