use std::{
    cmp::Ordering,
    collections::BTreeSet,
    hash::{Hash, Hasher},
    iter::FromIterator,
};

/// A node in a graph. It is identified by an unique unsigned integer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Node(usize);

impl Node {
    pub fn new(id: usize) -> Self {
        Node(id)
    }

    pub fn id(&self) -> usize {
        self.0
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Node) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Node) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl From<usize> for Node {
    fn from(id: usize) -> Self {
        Node::new(id)
    }
}

/// A set of nodes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeSet(BTreeSet<Node>);

impl NodeSet {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn empty() -> Self {
        Self(BTreeSet::default())
    }

    pub fn singleton(node: Node) -> Self {
        let mut set = Self::empty();
        set.insert(node);
        set
    }

    pub fn insert(&mut self, n: Node) -> bool {
        self.0.insert(n)
    }

    pub fn remove(&mut self, n: Node) {
        self.0.remove(&n);
    }

    pub fn contains(&self, n: &Node) -> bool {
        self.0.contains(n)
    }

    pub fn union<'a>(&'a self, other: &'a NodeSet) -> impl Iterator<Item = &'a Node> {
        self.0.union(&other.0)
    }

    pub fn intersection<'a>(&'a self, other: &'a NodeSet) -> impl Iterator<Item = &'a Node> {
        self.0.intersection(&other.0)
    }

    pub fn diff<'a>(&'a self, other: &'a NodeSet) -> impl Iterator<Item = &'a Node> {
        self.0.difference(&other.0)
    }

    pub fn to_sorted_vec(self) -> Vec<Node> {
        let mut vec: Vec<Node> = self.0.into_iter().collect();
        vec.sort();
        vec
    }

    pub fn to_vec(&self) -> Vec<Node> {
        self.0.clone().into_iter().collect()
    }

    // pub fn as_slice<'a>(&'a self) -> &'a [Node] {
    //     self.to_vec().as_slice()
    // }
}

impl IntoIterator for NodeSet {
    type Item = Node;
    type IntoIter = std::collections::btree_set::IntoIter<Node>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a NodeSet {
    type Item = Node;
    type IntoIter = std::iter::Copied<std::collections::btree_set::Iter<'a, Node>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().copied()
    }
}

impl FromIterator<Node> for NodeSet {
    fn from_iter<T: IntoIterator<Item = Node>>(iter: T) -> Self {
        NodeSet(iter.into_iter().collect())
    }
}

impl std::fmt::Display for NodeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            self.clone()
                .to_sorted_vec()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<String>>()
                .join(",")
        )
    }
}
