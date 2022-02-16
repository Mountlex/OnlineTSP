use std::ops::Index;

use serde::{Serialize, Deserialize};

use super::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeIndex {
    index: Vec<Option<usize>>,
    min_node_index: usize,
    num_nodes: usize,
}

impl NodeIndex {
    pub fn init(nodes: &[Node]) -> Self {
        let min_node_index = (&nodes).into_iter().map(|node| node.id()).min().unwrap();
        let max_node_index = (&nodes).into_iter().map(|node| node.id()).max().unwrap();

        let mut index: Vec<Option<usize>> = vec![None; max_node_index - min_node_index + 1];
        for (i, node) in nodes.into_iter().enumerate() {
            index[node.id() - min_node_index] = Some(i);
        }

        Self {
            index,
            num_nodes: nodes.len(),
            min_node_index,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn add_node(&mut self, new_node: Node) -> usize {
        if self.index.len() + self.min_node_index > new_node.id()
            && new_node.id() >= self.min_node_index
        {
            self.index[new_node.id() - self.min_node_index] = Some(self.num_nodes)
        } else if new_node.id() < self.min_node_index {
            let mut front_index = vec![None; self.min_node_index - new_node.id()];
            front_index[0] = Some(self.num_nodes);
            front_index.append(&mut self.index);
            self.index = front_index;
            self.min_node_index = new_node.id();
        } else {
            // self.index.len() + self.min_node_index <= new_node.id()
            let mut back_index =
                vec![None; new_node.id() - self.index.len() - self.min_node_index + 1];
            back_index[new_node.id() - self.index.len() - self.min_node_index] =
                Some(self.num_nodes);
            self.index.append(&mut back_index);
        }

        self.num_nodes += 1;
        self.num_nodes - 1
    }

    pub fn empty() -> Self {
        Self {
            index: vec![],
            min_node_index: 0,
            num_nodes: 0,
        }
    }

    pub fn get(&self, node: &Node) -> Option<usize> {
        if let Some(entry) = self.index.get(node.id() - self.min_node_index) {
            entry.clone()
        } else {
            None
        }
    }

    pub fn remove(&mut self, node: Node) {
        if let Some(idx) = self.index.get_mut(node.id() - self.min_node_index) {
            *idx = None;
        }
    }
}

impl Index<&Node> for NodeIndex {
    type Output = usize;

    fn index(&self, node: &Node) -> &Self::Output {
        self.index[node.id() - self.min_node_index]
            .as_ref()
            .unwrap()
    }
}

#[cfg(test)]
mod test_node_index {
    use super::*;

    #[test]
    fn test_generation() {
        let nodes: Vec<Node> = vec![3.into(), 7.into(), 2.into()];
        let index = NodeIndex::init(&nodes);

        assert_eq!(index[&3.into()], 0);
        assert_eq!(index[&7.into()], 1);
        assert_eq!(index[&2.into()], 2);
    }

    #[test]
    fn test_add() {
        let nodes: Vec<Node> = vec![3.into(), 7.into(), 2.into()];
        let mut index = NodeIndex::init(&nodes);

        index.add_node(5.into());
        assert_eq!(index[&5.into()], 3);

        index.add_node(1.into());
        assert_eq!(index[&1.into()], 4);

        index.add_node(10.into());
        assert_eq!(index[&10.into()], 5);
    }
}
