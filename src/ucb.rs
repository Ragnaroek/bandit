use super::utils::{find_arm, log, log_command, select_argmax};
use super::{BanditConfig, Identifiable, MultiArmedBandit};
use std::cmp::Eq;
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io;
use std::io::{Read, Write};
use std::path::Path;

pub static DEFAULT_CONFIG: UcbConfig = UcbConfig { alpha: 0.5 };

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct UcbConfig {
    /// The higher the value the faster the algorithms tends toward selecting
    /// the arm with highest reward. Should be a number between [0, 1.0)
    pub alpha: f64,
}

#[derive(Debug, PartialEq)]
pub struct UCB<A: Hash + Eq + Identifiable> {
    config: UcbConfig,
    bandit_config: BanditConfig,
    pub arms: Vec<A>,
    counts: HashMap<A, u64>,
    rewards: HashMap<A, f64>,
    all_counts: u64,
    all_arms_played_at_least_once: bool,
}

impl<A: Clone + Hash + Eq + Identifiable> UCB<A> {
    pub fn new(arms: Vec<A>, bandit_config: BanditConfig, config: UcbConfig) -> UCB<A> {
        assert!(!arms.is_empty(), "Arms vector cannot be empty!");
        let mut rewards = HashMap::new();
        for arm in &arms {
            rewards.insert(arm.clone(), 0.);
        }

        let mut counts = HashMap::new();
        for arm in &arms {
            counts.insert(arm.clone(), 0);
        }
        Self::new_with_values(arms, bandit_config, config, counts, rewards)
    }

    pub fn new_with_values(
        arms: Vec<A>,
        bandit_config: BanditConfig,
        config: UcbConfig,
        counts: HashMap<A, u64>,
        rewards: HashMap<A, f64>,
    ) -> UCB<A> {
        let all_counts: u64 = counts.values().sum();
        let all_arms_played_at_least_once =
            all_counts > 0 && counts.values().filter(|value| **value == 0).count() == 0;
        UCB {
            config,
            bandit_config,
            arms,
            counts,
            rewards,
            all_counts,
            all_arms_played_at_least_once,
        }
    }

    pub fn load_bandit(
        arms: Vec<A>,
        bandit_config: BanditConfig,
        path: &Path,
    ) -> io::Result<UCB<A>> {
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;

        let deser: ExternalFormat = serde_json::from_str(&content)?;

        let mut counts = HashMap::new();
        for (arm_ident, count) in deser.counts {
            let arm = find_arm(&arms, &arm_ident)?;
            counts.insert(arm.clone(), count);
        }
        let all_counts: u64 = counts.values().sum();
        let mut values = HashMap::new();
        for (arm_ident, val) in deser.rewards {
            let arm = find_arm(&arms, &arm_ident)?;
            values.insert(arm.clone(), val);
        }

        let all_arms_played = counts.values().filter(|c| **c == 0).count() == 0;
        Ok(UCB {
            config: deser.config,
            bandit_config,
            arms,
            counts,
            rewards: values,
            all_counts,
            all_arms_played_at_least_once: all_arms_played,
        })
    }

    fn check_if_all_played(&self) -> bool {
        self.counts.values().filter(|c| **c == 0).count() == 0
    }

    fn log_update(&self, arm: &A) {
        log(&log_command("UPDATE", arm), &self.bandit_config.log_file);
    }

    fn log_select(&self, arm: &A) {
        log(&log_command("SELECT", arm), &self.bandit_config.log_file);
    }

    fn exploration(&self, arm_counts: f64) -> f64 {
        ((self.all_counts as f64).ln() / arm_counts).sqrt()
    }

    fn calculate_best_arm(&self) -> Option<A> {
        let mut arms_estimations = vec![];
        for arm in self.arms.iter() {
            let rewards = self.rewards.get(arm)?;
            let n_counts = *self.counts.get(arm)? as f64;
            let exploratory_factor = self.exploration(n_counts);
            let est = *rewards / n_counts + exploratory_factor;
            arms_estimations.push(est);
        }
        let argmax = select_argmax(&arms_estimations)?;
        Some(self.arms[argmax].clone())
    }

    fn get_next_unexplored(&self) -> Option<A> {
        let mut unexplored: Vec<_> = self
            .counts
            .iter()
            .filter(|(_, cnt)| **cnt == 0)
            .map(|(arm, _)| arm.clone())
            .collect();
        unexplored.pop()
    }
}

impl<A: Clone + Hash + Eq + Identifiable> MultiArmedBandit<A> for UCB<A> {
    fn select_arm(&self) -> A {
        let possible_arm_to_play = if self.all_arms_played_at_least_once {
            self.calculate_best_arm()
        } else {
            self.get_next_unexplored()
        };
        match possible_arm_to_play {
            Some(arm) => {
                self.log_select(&arm);
                arm
            }
            None => {
                let fallback_arm = self.arms[self.arms.len() - 1].clone();
                self.log_select(&fallback_arm);
                fallback_arm
            }
        }
    }

    fn update(&mut self, arm: A, reward: f64) {
        self.all_counts += 1;
        let n_ = self.counts.entry(arm.clone()).or_insert(0);
        *n_ += 1;
        self.all_arms_played_at_least_once = self.check_if_all_played();
        let val = self.rewards.entry(arm.clone()).or_insert(0.0);
        *val += reward;
        self.log_update(&arm);
    }

    fn update_counts(&mut self, arm: &A) {
        self.all_counts += 1;
        let n_ = self.counts.entry(arm.clone()).or_insert(0);
        *n_ += 1;
        self.all_arms_played_at_least_once = self.check_if_all_played();
        self.log_update(arm);
    }

    fn update_rewards(&mut self, arm: &A, reward: f64) {
        let val = self.rewards.entry(arm.clone()).or_insert(0.0);
        *val += reward;
        self.log_update(arm);
    }

    fn save_bandit(&self, path: &Path) -> io::Result<()> {
        let mut counts = HashMap::new();
        for (arm, count) in &self.counts {
            counts.insert(arm.ident(), *count);
        }

        let mut arms = Vec::with_capacity(self.arms.len());
        let mut values = HashMap::new();
        for (arm, value) in &self.rewards {
            let arm_ident = arm.ident();
            arms.push(arm_ident.clone());
            values.insert(arm_ident, *value);
        }

        let external_format = ExternalFormat {
            arms,
            counts,
            rewards: values,
            config: self.config,
        };
        let ser = serde_json::to_string(&external_format)?;

        let mut file = File::create(path)?;
        file.write_all(&ser.into_bytes())?;
        file.flush()
    }
}

#[derive(Serialize, Deserialize)]
struct ExternalFormat {
    arms: Vec<String>,
    counts: HashMap<String, u64>,
    rewards: HashMap<String, f64>,
    config: UcbConfig,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DEFAULT_BANDIT_CONFIG;

    #[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
    struct TestArm {
        num: u32,
    }

    impl Identifiable for TestArm {
        fn ident(&self) -> String {
            format!("arm:{}", self.num)
        }
    }

    #[test]
    fn creating_bandit_works() {
        let arms = vec![
            TestArm { num: 0 },
            TestArm { num: 1 },
            TestArm { num: 2 },
            TestArm { num: 3 },
        ];
        let _bandit = UCB::new(
            arms.clone(),
            DEFAULT_BANDIT_CONFIG.clone(),
            DEFAULT_CONFIG.clone(),
        );
    }

    #[test]
    #[should_panic]
    fn creating_bandit_fails_with_empty_arm_vector() {
        let arms: Vec<TestArm> = vec![];
        UCB::new(arms, DEFAULT_BANDIT_CONFIG.clone(), DEFAULT_CONFIG.clone());
    }

    #[test]
    fn select_next_unexplored_arm() {
        let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
        let mut bandit = UCB::new(arms.clone(), DEFAULT_BANDIT_CONFIG.clone(), DEFAULT_CONFIG.clone());
        assert!(!bandit.all_arms_played_at_least_once);

        let n_arms = 3;
        for _ in 0..=n_arms {
            let arm = bandit.select_arm();
            bandit.update_counts(&arm);
        }
        assert!(bandit.all_arms_played_at_least_once);
        let expected_counts = vec![
            (TestArm{num: 0}, 1), (TestArm{num: 1}, 1),
            (TestArm{num: 2}, 1), (TestArm{num: 3}, 1),
        ].into_iter().collect::<HashMap<TestArm, u64>>();
        assert_eq!(bandit.counts, expected_counts)
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct TestArm {
    num: u32,
}

impl Identifiable for TestArm {
    fn ident(&self) -> String {
        format!("arm:{}", self.num)
    }
}
