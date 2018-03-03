extern crate rand;
extern crate serde;
extern crate serde_json;

use super::bandit::{MultiArmedBandit, Identifiable, BanditConfig};
use std::collections::{HashMap};
use std::hash::{Hash};
use std::cmp::{Eq};
use std::path::{Path};
use std::io::{Error, ErrorKind, Write, Read};
use std::io;
use std::fs::{File};
use std;

pub static DEFAULT_CONFIG : AnnealingSoftmaxConfig =  AnnealingSoftmaxConfig{cooldown_factor: 0.5};

const E : f64 = 2.71828_18284_59045_23536;

#[derive(Debug, PartialEq)]
pub struct AnnealingSoftmax<A: Hash + Eq + Identifiable> {
    config: AnnealingSoftmaxConfig,
    bandit_config: BanditConfig,
    pub arms: Vec<A>,
    counts: HashMap<A, u64>,
    values: HashMap<A, f64>
}

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct AnnealingSoftmaxConfig {
    /// The higher the value the faster the algorithms tends toward selecting
    /// the arm with highest reward. Should be a number between [0, 1.0)
    pub cooldown_factor : f64
}

impl<A: Clone + Hash + Eq + Identifiable> AnnealingSoftmax<A> {
    pub fn new(arms: Vec<A>, bandit_config: BanditConfig, config: AnnealingSoftmaxConfig) -> AnnealingSoftmax<A> {
        let mut values = HashMap::new();
        for i in 0..arms.len() {
            values.insert(arms[i].clone(), 0.0);
        }
        AnnealingSoftmax::new_with_values(arms, bandit_config, config, values)
    }

    pub fn new_with_values(arms: Vec<A>, bandit_config: BanditConfig, config: AnnealingSoftmaxConfig, values: HashMap<A, f64>) -> AnnealingSoftmax<A> {
        let mut counts = HashMap::new();
        for i in 0..arms.len() {
            counts.insert(arms[i].clone(), 0);
        }
        AnnealingSoftmax{config, bandit_config, arms, counts, values}
    }

    pub fn load_bandit(arms: Vec<A>, bandit_config: BanditConfig, path : &Path) -> io::Result<AnnealingSoftmax<A>> {

        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;

        let deser : ExternalFormat = serde_json::from_str(&content)?;

        let mut counts = HashMap::new();
        for (arm_ident, count) in deser.counts {
            let arm = find_arm(&arms, &arm_ident)?;
            counts.insert(arm.clone(), count);
        }
        let mut values = HashMap::new();
        for (arm_ident, val) in deser.values {
            let arm = find_arm(&arms, &arm_ident)?;
            values.insert(arm.clone(), val);
        }

        Ok(AnnealingSoftmax{config: deser.config, bandit_config: bandit_config, arms: arms, counts: counts, values: values})
    }
}

impl<A: Clone + Hash + Eq + Identifiable> MultiArmedBandit<A> for AnnealingSoftmax<A> {

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

    fn save_bandit(&self, path: &Path) -> io::Result<()> {

        let mut counts = HashMap::new();
        for (arm, count) in &self.counts {
            counts.insert(arm.ident(), *count);
        };

        let mut arms = Vec::with_capacity(self.arms.len());
        let mut values = HashMap::new();
        for (arm, value) in &self.values {
            let arm_ident = arm.ident();
            arms.push(arm_ident.clone());
            values.insert(arm_ident, *value);
        };

        let external_format = ExternalFormat {
            config: self.config.clone(),
            arms: arms,
            counts: counts,
            values: values,
        };
        let ser = serde_json::to_string(&external_format)?;

        let mut file = File::create(path)?;
        file.write_all(&ser.into_bytes())?;
        file.flush()
    }
}

fn find_arm<'a, A: Identifiable>(arms : &'a [A], ident: &str) -> io::Result<&'a A> {
    for arm in arms {
        if arm.ident() == ident {
            return Ok(arm);
        }
    }
    Err(Error::new(ErrorKind::NotFound, format!("arm {} not found", ident)))
}

#[derive(Serialize, Deserialize)]
struct ExternalFormat {
    config: AnnealingSoftmaxConfig,
    arms: Vec<String>,
    counts: HashMap<String, u64>,
    values: HashMap<String, f64>,
}
