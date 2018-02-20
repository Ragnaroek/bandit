
extern crate serde;

use std::path::{Path};
use std::hash::{Hash};
use std::io;

pub trait MultiArmedBandit<A: Hash + Clone + Identifiable> {
    fn select_arm(&self) -> A;
    fn update(&mut self, arm: A, reward: f64);

    /// stores the current state of the bandit algorithm in
    /// the supplied file. Every implementation has a corresponding
    /// load_bandit function.
    fn save_bandit(&self, path: &Path) -> io::Result<()>;
}

pub trait Identifiable {
    fn ident(&self) -> String;
}
