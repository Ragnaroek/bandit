
extern crate bandit;

use bandit::bandit::{MultiArmedBandit};
use bandit::softmax::{AnnealingSoftmax, AnnealingSoftmaxConfig, DEFAULT_CONFIG};
use std::collections::{HashMap};

const NUM_SELECTS : u32 = 100_000;
const EPSILON : u32 = (NUM_SELECTS as f64 * 0.005) as u32;

#[test]
pub fn test_select_arm() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm = AnnealingSoftmax::new(arms, DEFAULT_CONFIG);

    let mut selects : HashMap<TestArm, u32> = HashMap::new();
    for _ in 0..NUM_SELECTS {
        let arm_selected = sm.select_arm();
        *selects.entry(arm_selected).or_insert(0) += 1;
    }

    let expected_count = abs_select(0.25);
    for (arm, v) in selects {
        assert_prop(expected_count, v, arm);
    }
}

#[test]
fn test_moves_towards_arm_with_highest_reward_with_high_cooldown() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let arm_test_rewards = vec![98.0, 100.0, 99.0, 98.5];
    let mut sm = AnnealingSoftmax::new(arms.clone(), AnnealingSoftmaxConfig{cooldown_factor: 0.88});

    let num_iterations = 500;

    let mut selects = Vec::<[u64;4]>::with_capacity(num_iterations);
    for _ in 0..num_iterations {
        for i in 0..arms.len() {
            sm.update(arms[i], arm_test_rewards[i])
        }

        let mut draws = [0;4];
        for _ in 0..1000 {
            let selected_arm = sm.select_arm();
            draws[selected_arm.num as usize] += 1;
        }
        selects.push(draws);
    }

    assert!(selects[num_iterations-1][1] >= 998, format!("last round should favour highest reward, but did not {}", selects[num_iterations-1][1]));
}

#[test]
fn test_always_selects_highest_reward_if_totally_cooled_down() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let arm_test_rewards = vec![98.0, 100.0, 99.0, 98.5];
    let mut sm = AnnealingSoftmax::new(arms.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    let num_iterations = 1000;

    let mut selects = Vec::<[u64;4]>::with_capacity(num_iterations);
    for _ in 0..num_iterations {
        for i in 0..arms.len() {
            sm.update(arms[i], arm_test_rewards[i])
        }

        let mut draws = [0;4];
        for _ in 0..1000 {
            let selected_arm = sm.select_arm();
            draws[selected_arm.num as usize] += 1;
        }
        selects.push(draws);
    }

    assert_eq!(selects[num_iterations-1][1], 1000);
}

//Helper

fn abs_select(prop: f64) -> u32 {
    return (NUM_SELECTS as f64 * prop) as u32;
}

fn assert_prop(expected_count: u32, v: u32, arm: TestArm) {
    assert!(expected_count - EPSILON < v && v < expected_count + EPSILON, "expected {}+-{}, got {} arm {:?}", expected_count, EPSILON, v, arm);
}

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct TestArm {
    num: u32
}
