
use super::bandit::{MultiArmedBandit};
use std::collections::{HashMap};
use std::hash::{Hash};
use std::cmp::{Eq};

pub struct AnnealingSoftmax<A: Hash + Eq> {
    pub arms: Vec<A>,
    counts: HashMap<A, u32>
}

impl<A: Hash + Eq> AnnealingSoftmax<A> {
    pub fn new(arms: Vec<A>) -> AnnealingSoftmax<A> {
        return AnnealingSoftmax{arms, counts: HashMap::new()};
    }
}

impl<A: Clone + Hash + Eq> MultiArmedBandit<A> for AnnealingSoftmax<A> {

    fn select_arm(&self) -> A {
        return self.arms[0].clone();
    }

    fn update(&self, arm: A, reward: f32) {
    }
}
