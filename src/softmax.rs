extern crate rand;

use super::bandit::{MultiArmedBandit};
use std::collections::{HashMap};
use std::hash::{Hash};
use std::cmp::{Eq};
use std;

pub static DEFAULT_CONFIG : AnnealingSoftmaxConfig =  AnnealingSoftmaxConfig{cooldown_factor: 0.5};

const E : f64 = 2.71828_18284_59045_23536;

pub struct AnnealingSoftmax<A: Hash + Eq> {
    config: AnnealingSoftmaxConfig,
    pub arms: Vec<A>,
    counts: HashMap<A, u64>,
    values: HashMap<A, f64>
}

#[derive(Copy, Clone)]
pub struct AnnealingSoftmaxConfig {
    /// The higher the value the faster the algorithms tends toward selecting
    /// the arm with highest reward. Should be a number between [0, 1.0)
    pub cooldown_factor : f64
}



impl<A: Clone + Hash + Eq> AnnealingSoftmax<A> {
    pub fn new(arms: Vec<A>, config: AnnealingSoftmaxConfig) -> AnnealingSoftmax<A> {
        let mut values = HashMap::new();
        for i in 0..arms.len() {
            values.insert(arms[i].clone(), 0.0);
        }
        return AnnealingSoftmax::new_with_values(arms, config, values);
    }

    pub fn new_with_values(arms: Vec<A>, config: AnnealingSoftmaxConfig, values: HashMap<A, f64>) -> AnnealingSoftmax<A> {
        let mut counts = HashMap::new();
        for i in 0..arms.len() {
            counts.insert(arms[i].clone(), 0);
        }
        return AnnealingSoftmax{config, arms, counts, values};
    }
}

impl<A: Clone + Hash + Eq> MultiArmedBandit<A> for AnnealingSoftmax<A> {

    fn select_arm(&self) -> A {

        let mut t : u64 = 1;
        for v in self.counts.values() {
            t += v;
        }
        let temperature = 1.0 / (t as f64 + 0.0000001).ln();
        let cool_down = E*self.config.cooldown_factor;

        let mut z : f64 = 0.0;
        for v in self.values.values() {
            z += cool_down.powf(v / temperature)
        }

        if z.is_infinite() {
            let mut highest_reward_arm : Option<&A> = None;
            let mut highest_value = std::f64::MIN;
            for (arm, v) in self.values.iter() {
                if *v > highest_value {
                    highest_value = *v;
                    highest_reward_arm = Some(arm);
                }
            }
            if highest_reward_arm.is_some() {
                return highest_reward_arm.expect("highest reward arm").clone();
            } else {
                self.arms[self.arms.len()-1].clone();
            }
        }

        let rnd : f64 = rand::random();
        let mut cum_prob : f64 = 0.0;
        for (arm, v) in self.values.iter() {
            let mut prob = (cool_down.powf(v / temperature)) / z;

            if prob.is_nan() {
                prob = 0.0;
            }
            cum_prob += prob;
            if cum_prob > rnd {
                return arm.clone();
            }
        }
        return self.arms[self.arms.len()-1].clone();
    }

    fn update(&mut self, arm: A, reward: f64) {
        let n_ = self.counts.entry(arm.clone()).or_insert(0);
        *n_ += 1;
        let n = *n_ as f64;

        let val = self.values.entry(arm).or_insert(0.0);
        *val = ((n - 1.0) / n) * *val + (1.0 / n) * reward

    }
}
