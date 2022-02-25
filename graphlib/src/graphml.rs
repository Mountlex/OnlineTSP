use std::path::PathBuf;

use crate::{cost::Cost, AdjListGraph};

pub fn graphml_import(
    filename: PathBuf,
    max_nodes: Option<usize>,
) -> AdjListGraph {
    let mut graph = AdjListGraph::new();

    let text = std::fs::read_to_string(filename).unwrap();
    match roxmltree::Document::parse(&text) {
        Ok(doc) => {
            let length_key = doc
                .descendants()
                .find(|n| {
                    n.tag_name().name() == "key" && n.attribute("attr.name") == Some("length")
                })
                .unwrap()
                .attribute("id")
                .unwrap();

            if let Some(g) = doc.descendants().find(|n| n.tag_name().name() == "graph") {
                for edge in g.descendants().filter(|n| n.tag_name().name() == "edge") {
                    let source = edge.attribute("source").unwrap().parse::<u64>().unwrap() as usize;
                    let sink = edge.attribute("target").unwrap().parse::<u64>().unwrap() as usize;

                    if source < sink && (max_nodes.is_none() || sink < max_nodes.unwrap()) {
                        if let Some(length_node) = edge.descendants().find(|c| {
                            c.tag_name().name() == "data" && c.attribute("key") == Some(length_key)
                        }) {
                            let real = length_node.text().unwrap().trim().parse::<f64>().unwrap();
                            let cost = 
                                Cost::new(real.ceil() as usize);
                            
                            graph.add_edge(source.into(), sink.into(), cost);
                        }
                    }
                }
            }
        }
        Err(e) => println!("Error: {}.", e),
    }

    graph
}
