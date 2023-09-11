#![crate_name = "bandit"]
#![crate_type = "lib"]

#[macro_use]
extern crate serde_derive;
extern crate serde;

use std::path::{PathBuf, Path};
use std::hash::{Hash};
use std::io;

pub mod softmax;
pub mod ucb;
mod utils;

#[derive(Debug, PartialEq, Clone)]
pub struct BanditConfig {
    /// Log file for logging details about the bandit algorithm
    /// run. What will be logged depends on the bandit algorithm
    /// implementation.
    pub log_file: Option<PathBuf>
}

pub static DEFAULT_BANDIT_CONFIG : BanditConfig = BanditConfig{log_file: Option::None};

pub trait MultiArmedBandit<A: Hash + Clone + Identifiable> {
    fn select_arm(&self) -> A;
    fn update(&mut self, arm: A, reward: f64);
    /// additional function to update counts of selected arm in multi-threaded applications
    fn update_counts(&mut self, arm: &A);
    /// additional function to update rewards of selected arm in multi-threaded applications
    fn update_rewards(&mut self, arm: &A, reward: f64);

    /// stores the current state of the bandit algorithm in
    /// the supplied file. Every implementation has a corresponding
    /// load_bandit function.
    fn save_bandit(&self, path: &Path) -> io::Result<()>;
}

pub trait Identifiable {
    fn ident(&self) -> String;
}
