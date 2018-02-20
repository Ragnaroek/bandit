extern crate bandit;

use bandit::bandit::{MultiArmedBandit, Identifiable};
use bandit::softmax::{AnnealingSoftmax, AnnealingSoftmaxConfig, DEFAULT_CONFIG};
use std::collections::{HashMap};
use std::path::{Path};

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

    assert!(selects[num_iterations-1][1] >= 996, format!("last round should favour highest reward, but did not {}", selects[num_iterations-1][1]));
}

#[test]
fn test_eq() {
    let arms0 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm0 = AnnealingSoftmax::new(arms0.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    let arms0_2 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm0_2 = AnnealingSoftmax::new(arms0_2.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});
    sm0_2.select_arm(); //arm select does not change state
    sm0_2.select_arm();

    let arms1 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}, TestArm{num:4}];
    let sm1 = AnnealingSoftmax::new(arms1.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    let arms2 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}, TestArm{num:4}];
    let mut sm2 = AnnealingSoftmax::new(arms2.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});
    sm2.update(arms2[0], 34.32);

    let arms3 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let mut sm3 = AnnealingSoftmax::new(arms3.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});
    sm3.update(arms3[0], 34.32);
    sm3.update(arms3[2], 65.65);
    sm3.update(arms3[3], 12.49);

    assert_eq!(sm0, sm0_2);
    assert!(sm0 != sm1);
    assert!(sm1 != sm2);
    assert!(sm1 != sm3);
    assert!(sm2 != sm3);
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

#[test]
fn test_save_and_load_bandit() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let mut sm = AnnealingSoftmax::new(arms.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});
    sm.update(arms[0], 56.0);
    sm.update(arms[1], 63.22933432432171);
    //no update on arms[2]
    sm.update(arms[3], 733897263040475.72620335034262);

    let save_result = sm.save_bandit(Path::new("./tmp_bandit.json"));
    assert!(save_result.is_ok(), "save failed {:?}", save_result);

    let load_result = AnnealingSoftmax::load_bandit(arms, Path::new("./tmp_bandit.json"));
    assert!(load_result.is_ok(), "load failed {:?}", load_result);
    let sm_loaded : AnnealingSoftmax<TestArm> = load_result.unwrap();

    assert_eq!(sm, sm_loaded);
}

#[test]
fn test_save_and_load_bandit_with_missing_arm() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm = AnnealingSoftmax::new(arms.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    let save_result = sm.save_bandit(Path::new("./tmp_bandit_err.json"));
    assert!(save_result.is_ok(), "save failed {:?}", save_result);

    let arms_last_one_missing = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}];
    let load_result = AnnealingSoftmax::load_bandit(arms_last_one_missing, Path::new("./tmp_bandit.json"));
    assert!(load_result.is_err(), "load should fail, since TestArm{num: 3} could not be found");
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

impl Identifiable for TestArm {
    fn ident(&self) -> String {
        format!("arm:{}", self.num)
    }
}
