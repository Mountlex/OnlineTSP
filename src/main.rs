use anyhow::Result;
use clap::{Args, Parser};
use csv::Writer;
use graphlib::{graphml::graphml_import, SpMetricGraph, sp};
use oltsp::{ignore, instance_from_file, replan, learning_augmented};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser)]
enum Cli {
    Exp1(Exp1),
}

#[derive(Args)]
struct Exp1 {
    #[clap(parse(from_os_str), value_name = "INSTANCE_DIR")]
    instance_set: PathBuf,

    #[clap(
        long,
        parse(from_os_str),
        default_value = "data/osm/Manhattan.graphml"
    )]
    graph: PathBuf,

    #[clap(short, long, default_value = "500")]
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
    opt: u64,
    alg: u64,
}

fn main() -> Result<()> {
    set_up_logging()?;
    let args = Cli::parse();

    match args {
        Cli::Exp1(exp) => {
            log::info!("Importing graph from file...");
            let graph = graphml_import(exp.graph, None, None);
            log::info!("    ...success!");

            log::info!("Importing metric from file...");
            let sp = sp::load_or_compute(&"data/osm/manhattan.dat".into(), &graph, exp.scale)?;
            log::info!("    ...success!");

            let paths: Vec<std::fs::DirEntry> = std::fs::read_dir(exp.instance_set)?
                .filter_map(|e| e.ok())
                .collect();

            let results: Vec<Exp1Result> = paths
                .into_par_iter()
                .map(|file| {
                    let start_node = 1.into();
                    let instance = instance_from_file(&file.path()).unwrap();
                    let mut nodes = instance.distinct_nodes();
                    if !nodes.contains(&start_node) {
                        nodes.push(start_node)
                    }
                    let metric_graph = SpMetricGraph::from_metric_on_nodes(nodes, sp.clone());

                    log::info!("Computing optimal solution for {:?}...", file.file_name());
                    let (opt, tour) = instance.optimal_solution(start_node, sp.clone(), graphlib::tsp::SolutionType::Approx);
                    log::info!("    ...success. Optimal tour = {}", tour);

                    let mut results: Vec<Exp1Result> = vec![];
                    
                    [0.0, 0.5, 1.0].iter().for_each(|alpha| {
                        results.push(Exp1Result {
                            name: "pred-replan".into(),
                            param: *alpha,
                            opt: opt.get_usize() as u64,
                            alg: learning_augmented(
                                &graph,
                                &metric_graph,
                                instance.clone(),
                                start_node,
                                *alpha,
                                instance.clone(),
                                graphlib::tsp::SolutionType::Approx,
                            ) as u64,
                        });
                    });

                    results.push(Exp1Result {
                        name: "ignore".into(),
                        param: 0.0,
                        opt: opt.get_usize() as u64,
                        alg: ignore(
                            &graph,
                            &metric_graph,
                            instance.clone(),
                            start_node,
                            graphlib::tsp::SolutionType::Approx,
                        ) as u64,
                    });
                    results.push(Exp1Result {
                        name: "replan".into(),
                        param: 0.0,
                        opt: opt.get_usize() as u64,
                        alg: replan(
                            &graph,
                            &metric_graph,
                            instance.clone(),
                            start_node,
                            graphlib::tsp::SolutionType::Approx,
                        ) as u64,
                    });

                    
                    

                    results
                })
                .flatten()
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
        .level(log::LevelFilter::Info)
        .chain(fern::log_file(format!(
            "logs/{}.log",
            chrono::Local::now().format("%d%m%Y-%H%M")
        ))?)
        .apply()?;

    log::info!("Logger set up!");

    Ok(())
}
