extern crate bandit;
extern crate regex;

mod common;

use bandit::{MultiArmedBandit, Identifiable, BanditConfig, DEFAULT_BANDIT_CONFIG};
use bandit::softmax::{AnnealingSoftmax, AnnealingSoftmaxConfig, DEFAULT_CONFIG};
use std::collections::{HashMap};
use std::path::{Path, PathBuf};
use std::fs::{File, remove_file};
use std::io::{Read};
use regex::{Regex};

use common::{TestArm, NUM_SELECTS};

const EPSILON : u32 = (NUM_SELECTS as f64 * 0.005) as u32;

#[test]
pub fn test_select_arm() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm = AnnealingSoftmax::new(arms, DEFAULT_BANDIT_CONFIG.clone(), DEFAULT_CONFIG);

    let mut selects : HashMap<TestArm, u32> = HashMap::new();
    for _ in 0..NUM_SELECTS {
        let arm_selected = sm.select_arm();
        *selects.entry(arm_selected).or_insert(0) += 1;
    }

    let expected_count = common::abs_select(0.25);
    for (arm, v) in selects {
        common::assert_prop(expected_count, v, arm);
    }
}

#[test]
fn test_moves_towards_arm_with_highest_reward_with_high_cooldown() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let arm_test_rewards = vec![98.0, 100.0, 99.0, 98.5];
    let mut sm = AnnealingSoftmax::new(arms.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 0.88});

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

    assert!(selects[num_iterations-1][1] >= 996, "last round should favour highest reward, but did not {}", selects[num_iterations-1][1]);
}

#[test]
fn test_eq() {
    let arms0 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm0 = AnnealingSoftmax::new(arms0.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    let arms0_2 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm0_2 = AnnealingSoftmax::new(arms0_2.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});
    sm0_2.select_arm(); //arm select does not change state
    sm0_2.select_arm();

    let arms1 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}, TestArm{num:4}];
    let sm1 = AnnealingSoftmax::new(arms1.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    let arms2 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}, TestArm{num:4}];
    let mut sm2 = AnnealingSoftmax::new(arms2.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});
    sm2.update(arms2[0], 34.32);

    let arms3 = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let mut sm3 = AnnealingSoftmax::new(arms3.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});
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
    let mut sm = AnnealingSoftmax::new(arms.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});

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
    let mut sm = AnnealingSoftmax::new(arms.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});
    sm.update(arms[0], 56.0);
    sm.update(arms[1], 63.22933432432171);
    //no update on arms[2]
    sm.update(arms[3], 733897263040475.72620335034262);

    let save_result = sm.save_bandit(Path::new("./tmp_bandit.json"));
    assert!(save_result.is_ok(), "save failed {:?}", save_result);

    let load_result = AnnealingSoftmax::load_bandit(arms, DEFAULT_BANDIT_CONFIG.clone(), Path::new("./tmp_bandit.json"));
    assert!(load_result.is_ok(), "load failed {:?}", load_result);
    let sm_loaded : AnnealingSoftmax<TestArm> = load_result.unwrap();

    assert_eq!(sm, sm_loaded);
}

#[test]
fn test_save_and_load_bandit_with_missing_arm() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm = AnnealingSoftmax::new(arms.clone(), DEFAULT_BANDIT_CONFIG.clone(), AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    let save_result = sm.save_bandit(Path::new("./tmp_bandit_err.json"));
    assert!(save_result.is_ok(), "save failed {:?}", save_result);

    let arms_last_one_missing = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}];
    let load_result = AnnealingSoftmax::load_bandit(arms_last_one_missing, DEFAULT_BANDIT_CONFIG.clone(), Path::new("./tmp_bandit.json"));
    assert!(load_result.is_err(), "load should fail, since TestArm{{num: 3}} could not be found, but was {:?}", load_result);
}

#[test]
fn test_logging_update() {

    let test_file = Path::new(common::LOG_UPDATE_FILE);
    if test_file.exists() {
        remove_file(test_file).unwrap();
    }

    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let bandit_config = BanditConfig{log_file: Some(PathBuf::from(common::LOG_UPDATE_FILE))};
    let mut sm = AnnealingSoftmax::new(arms.clone(), bandit_config, AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    sm.update(arms[0], 10.0);
    sm.update(arms[1], 20.0);
    sm.update(arms[2], 30.0);
    sm.update(arms[3], 40.0);

    let log_content = common::read_file_content(common::LOG_UPDATE_FILE);

    let re = Regex::new(
r#"^UPDATE;arm:0;\d{13};10
UPDATE;arm:1;\d{13};20
UPDATE;arm:2;\d{13};30
UPDATE;arm:3;\d{13};40
$"#).expect("compiled regex");

    assert!(re.is_match(&log_content), "log file did not match expected, was {}", &log_content);
}

#[test]
fn test_logging_select() {

    let test_file = Path::new(common::LOG_SELECT_FILE);
    if test_file.exists() {
        remove_file(test_file).unwrap();
    }

    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let bandit_config = BanditConfig{log_file: Some(PathBuf::from(common::LOG_SELECT_FILE))};
    let sm = AnnealingSoftmax::new(arms.clone(), bandit_config, AnnealingSoftmaxConfig{cooldown_factor: 1.0});

    let select1 = sm.select_arm();
    let select2 = sm.select_arm();
    let select3 = sm.select_arm();

    let log_content = common::read_file_content(common::LOG_SELECT_FILE);

    let re = Regex::new(&format!(
r#"^SELECT;{};\d{{13}}
SELECT;{};\d{{13}}
SELECT;{};\d{{13}}
$"#, select1.ident(), select2.ident(), select3.ident())).expect("compiled regex");

    assert!(re.is_match(&log_content), "log file did not match expected, was {}", &log_content);
}
