use anyhow::Result;
use clap::{Args, Parser};
use csv::Writer;
use graphlib::{
    graphml::graphml_import,
    sp::{self, ShortestPathsCache},
    tsp, Metric, MetricView, Node, Nodes, SpMetricGraph,
};
use oltsp::{
    gaussian_prediction, ignore, instance_from_file, learning_augmented, replan, smartstart,
    Instance,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser)]
enum Cli {
    Exp1(Exp1),
    Exp2(Exp2),
}

#[derive(Args)]
struct Exp1 {
    #[clap(parse(from_os_str), value_name = "INSTANCE_DIR")]
    instance_set: PathBuf,

    #[clap(long, parse(from_os_str), default_value = "data/osm/Manhattan.graphml")]
    graph: PathBuf,

    #[clap(long = "wc-rd")]
    wc_rd: bool,

    #[clap(long = "num-sigma", default_value = "10")]
    num_sigmas: i32,

    #[clap(long = "base-sigma", default_value = "2")]
    base_sigma: f64,

    #[clap(short, long, default_value = "1")]
    scale: usize,

    #[clap(
        short,
        long,
        global = true,
        default_value = "results.csv",
        parse(from_os_str)
    )]
    output: PathBuf,
}

#[derive(Args)]
struct Exp2 {
    #[clap(parse(from_os_str), value_name = "INSTANCE_DIR")]
    instance_set: PathBuf,

    #[clap(long, parse(from_os_str), default_value = "data/osm/Manhattan.graphml")]
    graph: PathBuf,

    #[clap(long = "wc-rd")]
    wc_rd: bool,

    #[clap(long = "pred-frac", default_value = "11")]
    num_predictions: usize,

    #[clap(short, long, default_value = "1")]
    scale: usize,

    #[clap(
        short,
        long,
        global = true,
        default_value = "results.csv",
        parse(from_os_str)
    )]
    output: PathBuf,
}

#[derive(Serialize)]
struct Exp1Result {
    name: String,
    param: f64,
    sigma: f64,
    opt: u64,
    alg: u64,
}

#[derive(Serialize)]
struct Exp2Result {
    name: String,
    param: f64,
    frac: f64,
    opt: u64,
    alg: u64,
}

fn compute_wc_instance(instance: Instance, start_node: Node, sp: &ShortestPathsCache) -> Instance {
    let mut nodes = instance.distinct_nodes();
    if !nodes.contains(&start_node) {
        nodes.push(start_node)
    }
    let metric_graph = MetricView::from_metric_on_nodes(nodes, sp, None, None);
    let (_, tour) = tsp::tsp_tour(
        &metric_graph,
        start_node,
        graphlib::tsp::SolutionType::Approx,
    );
    let mut reqs: Vec<(Node, usize)> = vec![];
    let mut t = 0;
    for edge in tour.windows(2) {
        t += metric_graph.distance(edge[0], edge[1]).get_usize();
        reqs.push((edge[1], t));
    }
    reqs.into()
}

fn main() -> Result<()> {
    set_up_logging()?;
    let args = Cli::parse();

    match args {
        Cli::Exp1(exp) => {
            let graph = graphml_import(exp.graph, None);
            let sp = sp::load_or_compute(&"data/osm/manhattan.dat".into(), &graph)?;

            let paths: Vec<std::fs::DirEntry> = std::fs::read_dir(exp.instance_set)?
                .filter_map(|e| e.ok())
                .collect();

            let results: Vec<Exp1Result> = paths
                .into_par_iter()
                .flat_map(|file| {
                    let start_node = 1.into();
                    log::info!("Loading instance from {:?}", file.path());
                    let mut instance = instance_from_file(&file.path()).unwrap();
                    instance.scale_rd(exp.scale);

                    if exp.wc_rd {
                        instance = compute_wc_instance(instance, start_node, &sp);
                    }

                    log::info!(
                        "Computing lower bound on optimal solution for {:?}...",
                        file.file_name()
                    );
                    let lb = instance
                        .optimal_solution(start_node, &sp, tsp::SolutionType::Approx)
                        .0
                        .get_usize();
                    log::info!("    ...success. {}", lb);

                    let base_nodes: Vec<Node> = graph.nodes().collect();

                    let t_ignore = ignore(
                        &graph,
                        &sp,
                        instance.clone(),
                        start_node,
                        graphlib::tsp::SolutionType::Approx,
                    ) as u64;

                    let t_replan = replan(
                        &graph,
                        &sp,
                        instance.clone(),
                        start_node,
                        graphlib::tsp::SolutionType::Approx,
                    ) as u64;

                    let t_smart = smartstart(
                        &graph,
                        &sp,
                        instance.clone(),
                        start_node,
                        graphlib::tsp::SolutionType::Approx,
                    ) as u64;

                    let results: Vec<Exp1Result> = (0..exp.num_sigmas)
                        .into_iter()
                        .flat_map(|sigma_num| {
                            let sigma = exp.base_sigma.powi(sigma_num) - 1.0;
                            let pred = gaussian_prediction(&instance, &sp, &base_nodes, sigma, 1.0);
                            let mut results: Vec<Exp1Result> = vec![];

                            [0.0, 0.1, 0.5, 1.0].iter().for_each(|alpha| {
                                results.push(Exp1Result {
                                    name: "pred".into(),
                                    param: *alpha,
                                    opt: lb as u64,
                                    alg: learning_augmented(
                                        &graph,
                                        &sp,
                                        instance.clone(),
                                        start_node,
                                        *alpha,
                                        pred.clone(),
                                        graphlib::tsp::SolutionType::Approx,
                                    ) as u64,
                                    sigma,
                                });
                            });

                            results.push(Exp1Result {
                                name: "ignore".into(),
                                param: 0.0,
                                opt: lb as u64,
                                alg: t_ignore,
                                sigma,
                            });
                            results.push(Exp1Result {
                                name: "replan".into(),
                                param: 0.0,
                                opt: lb as u64,
                                alg: t_replan,
                                sigma,
                            });
                            results.push(Exp1Result {
                                name: "smart".into(),
                                param: 0.0,
                                opt: lb as u64,
                                alg: t_smart,
                                sigma,
                            });
                            results
                        })
                        .collect();

                    results
                })
                .collect();
            export(&exp.output, results)?
        }
        Cli::Exp2(exp) => {
            let graph = graphml_import(exp.graph, None);
            let sp = sp::load_or_compute(&"data/osm/manhattan.dat".into(), &graph)?;

            let paths: Vec<std::fs::DirEntry> = std::fs::read_dir(exp.instance_set)?
                .filter_map(|e| e.ok())
                .collect();

            let results: Vec<Exp2Result> = paths
                .into_par_iter()
                .flat_map(|file| {
                    let start_node = 1.into();
                    log::info!("Loading instance from {:?}", file.path());
                    let mut instance = instance_from_file(&file.path()).unwrap();
                    instance.scale_rd(exp.scale);
                    if exp.wc_rd {
                        instance = compute_wc_instance(instance, start_node, &sp);
                    }

                    let lb = instance
                        .optimal_solution(start_node, &sp, tsp::SolutionType::Approx)
                        .0
                        .get_usize();

                    let base_nodes: Vec<Node> = graph.nodes().collect();

                    let t_ignore = ignore(
                        &graph,
                        &sp,
                        instance.clone(),
                        start_node,
                        graphlib::tsp::SolutionType::Approx,
                    ) as u64;

                    let t_replan = replan(
                        &graph,
                        &sp,
                        instance.clone(),
                        start_node,
                        graphlib::tsp::SolutionType::Approx,
                    ) as u64;

                    let t_smart = smartstart(
                        &graph,
                        &sp,
                        instance.clone(),
                        start_node,
                        graphlib::tsp::SolutionType::Approx,
                    ) as u64;

                    

                    let results: Vec<Exp2Result> = (0..exp.num_predictions)
                        .into_iter()
                        .flat_map(|num_p| {
                            let frac = num_p as f64 / (exp.num_predictions as f64 - 1.0);
                            let pred = gaussian_prediction(
                                &instance,
                                &sp,
                                &base_nodes,
                                0.0,
                                frac,
                            );
                            let mut results: Vec<Exp2Result> = vec![];

                            [0.0, 0.1, 0.5, 1.0].iter().for_each(|alpha| {
                                results.push(Exp2Result {
                                    name: "pred".into(),
                                    param: *alpha,
                                    opt: lb as u64,
                                    alg: learning_augmented(
                                        &graph,
                                        &sp,
                                        instance.clone(),
                                        start_node,
                                        *alpha,
                                        pred.clone(),
                                        graphlib::tsp::SolutionType::Approx,
                                    ) as u64,
                                    frac,
                                });
                            });

                            results.push(Exp2Result {
                                name: "ignore".into(),
                                param: 0.0,
                                opt: lb as u64,
                                alg: t_ignore,
                                frac,
                            });
                            results.push(Exp2Result {
                                name: "replan".into(),
                                param: 0.0,
                                opt: lb as u64,
                                alg: t_replan,
                                frac,
                            });
                            results.push(Exp2Result {
                                name: "smart".into(),
                                param: 0.0,
                                opt: lb as u64,
                                alg: t_smart,
                                frac,
                            });
                            results
                        })
                        .collect();

                    results
                })
                .collect();
            export(&exp.output, results)?
        }
    }
    Ok(())
}

fn export<E: Serialize>(output: &PathBuf, results: Vec<E>) -> Result<()> {
    let mut wtr = Writer::from_path(output)?;
    for entry in results {
        wtr.serialize(entry)?;
    }
    Ok(())
}

fn set_up_logging() -> Result<(), fern::InitError> {
    std::fs::create_dir_all("logs")?;
    fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "[{date}][{level}] {message}",
                date = chrono::Local::now().format("%H:%M:%S"),
                level = record.level(),
                message = message
            ));
        })
        .level(log::LevelFilter::Warn)
        .chain(fern::log_file(format!(
            "logs/{}.log",
            chrono::Local::now().format("%d%m%Y-%H%M")
        ))?)
        .apply()?;

    log::info!("Logger set up!");

    Ok(())
}
