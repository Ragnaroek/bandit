extern crate rand;

use super::bandit::{MultiArmedBandit};
use std::collections::{HashMap};
use std::hash::{Hash};
use std::cmp::{Eq};

pub struct AnnealingSoftmax<A: Hash + Eq> {
    pub arms: Vec<A>,
    counts: HashMap<A, u64>,
    values: HashMap<A, f64>
}

impl<A: Clone + Hash + Eq> AnnealingSoftmax<A> {
    pub fn new(arms: Vec<A>) -> AnnealingSoftmax<A> {
        let mut values = HashMap::new();
        for i in 0..arms.len() {
            values.insert(arms[i].clone(), 0.0);
        }
        return AnnealingSoftmax::new_with_values(arms, values);
    }

    pub fn new_with_values(arms: Vec<A>, values: HashMap<A, f64>) -> AnnealingSoftmax<A> {
        let mut counts = HashMap::new();
        for i in 0..arms.len() {
            counts.insert(arms[i].clone(), 0);
        }
        return AnnealingSoftmax{arms: arms, counts, values};
    }
}

impl<A: Clone + Hash + Eq> MultiArmedBandit<A> for AnnealingSoftmax<A> {

    fn select_arm(&self) -> A {

        let temperature = 1.0;

        let mut z = 0.0;
        for v in self.values.values() {
            z += (v / temperature).exp()
        }

        let rnd : f64 = rand::random();
        let mut cum_prob = 0.0;
        for (arm, v) in self.values.iter() {
            let prob = ((v / temperature).exp() / z);
            cum_prob += prob;
            if cum_prob > rnd {
                return arm.clone();
            }
        }
        return self.arms[self.arms.len()-1].clone();
    }

    fn update(&self, arm: A, reward: f32) {
    }
}
