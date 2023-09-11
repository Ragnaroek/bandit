extern crate bandit;
extern crate regex;

mod common;

use bandit::ucb::{UcbConfig, DEFAULT_CONFIG, UCB};
use bandit::{BanditConfig, Identifiable, MultiArmedBandit, DEFAULT_BANDIT_CONFIG};
use regex::Regex;
use std::collections::HashMap;
use std::fs::remove_file;
use std::path::{Path, PathBuf};

use common::{TestArm, NUM_SELECTS};

#[test]
pub fn test_select_arm() {
    let arms = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let mut ucb = UCB::new(arms, DEFAULT_BANDIT_CONFIG.clone(), DEFAULT_CONFIG);

    let mut selects: HashMap<TestArm, u32> = HashMap::new();
    for _ in 0..NUM_SELECTS {
        let arm_selected = ucb.select_arm();
        *selects.entry(arm_selected).or_default() += 1;
        ucb.update_counts(&arm_selected);
    }

    let expected_count = common::abs_select(0.25);
    for (arm, v) in selects {
        common::assert_prop(expected_count, v, arm);
    }
}

#[test]
fn test_moves_towards_arm_with_highest_reward_with_low_alpha() {
    let arms = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let arm_test_rewards = vec![98.0, 100.0, 99.0, 98.5];
    let mut sm = UCB::new(
        arms.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 0.1 },
    );

    let num_iterations = 500;

    let mut selects = Vec::<[u64; 4]>::with_capacity(num_iterations);
    for _ in 0..num_iterations {
        for i in 0..arms.len() {
            sm.update_counts(&arms[i]);
            sm.update(arms[i], arm_test_rewards[i])
        }

        let mut draws = [0; 4];
        for _ in 0..1000 {
            let selected_arm = sm.select_arm();
            draws[selected_arm.num as usize] += 1;
        }
        selects.push(draws);
    }

    assert!(
        selects[num_iterations - 1][1] >= 996,
        "last round should favour highest reward, but did not {}",
        selects[num_iterations - 1][1]
    );
}

#[test]
fn test_eq() {
    let arms0 = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let ucb0 = UCB::new(
        arms0.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 1.0 },
    );

    let arms0_2 = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let ucb0_2 = UCB::new(
        arms0_2.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 1.0 },
    );
    ucb0_2.select_arm(); //arm select does not change state
    ucb0_2.select_arm();

    let arms1 = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
        TestArm { num: 4 },
    ];
    let ucb1 = UCB::new(
        arms1.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 1.0 },
    );

    let arms2 = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
        TestArm { num: 4 },
    ];
    let mut ucb2 = UCB::new(
        arms2.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 1.0 },
    );
    ucb2.update(arms2[0], 1.);

    let arms3 = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let mut ucb3 = UCB::new(
        arms3.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 1.0 },
    );
    ucb3.update(arms3[0], 34.32);
    ucb3.update(arms3[2], 1.);
    ucb3.update(arms3[3], 1.);

    assert_eq!(ucb0, ucb0_2);
    assert_ne!(ucb0, ucb1);
    assert_ne!(ucb1, ucb2);
    assert_ne!(ucb1, ucb3);
    assert_ne!(ucb2, ucb3);
}

#[test]
fn test_more_often_selects_highest_reward_if_alpha_is_zero() {
    let arms = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let reward_values = vec![10., 9000., 5., 1.];
    let mut counts = HashMap::new();
    let mut rewards = HashMap::new();
    for (id, arm) in arms.iter().enumerate() {
        counts.insert(arm.clone(), 10_000);
        rewards.insert(arm.clone(), reward_values[id]);
    }
    let ucb = UCB::new_with_values(
        arms.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 0.0 },
        counts,
        rewards,
    );

    let num_iterations = 1_000;
    let mut selects = Vec::<[u64; 4]>::with_capacity(num_iterations);
    for _ in 0..num_iterations {
        let mut draws = [0; 4];
        for _ in 0..num_iterations {
            let selected_arm = ucb.select_arm();
            draws[selected_arm.num as usize] += 1;
        }
        selects.push(draws);
    }

    assert_eq!(selects[num_iterations - 1][1] as usize, num_iterations);
}

#[test]
fn test_save_and_load_bandit() {
    let arms = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let mut ucb = UCB::new(
        arms.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 0.5 },
    );
    ucb.update(arms[0], 1.);
    ucb.update(arms[1], 1.);
    //no update on arms[2]
    ucb.update(arms[3], 1.);

    let save_result = ucb.save_bandit(Path::new("./tmp_bandit.json"));
    assert!(save_result.is_ok(), "save failed {:?}", save_result);

    let load_result = UCB::load_bandit(
        arms,
        DEFAULT_BANDIT_CONFIG.clone(),
        Path::new("./tmp_bandit.json"),
    );
    assert!(load_result.is_ok(), "load failed {:?}", load_result);
    let ucb_loaded: UCB<TestArm> = load_result.unwrap();

    assert_eq!(ucb, ucb_loaded);
}

#[test]
fn test_save_and_load_bandit_with_missing_arm() {
    let arms = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let ucb = UCB::new(
        arms.clone(),
        DEFAULT_BANDIT_CONFIG.clone(),
        UcbConfig { alpha: 1.0 },
    );

    let save_result = ucb.save_bandit(Path::new("./tmp_bandit_err.json"));
    assert!(save_result.is_ok(), "save failed {:?}", save_result);

    let arms_last_one_missing = vec![TestArm { num: 0 }, TestArm { num: 1 }, TestArm { num: 2 }];
    let load_result = UCB::load_bandit(
        arms_last_one_missing,
        DEFAULT_BANDIT_CONFIG.clone(),
        Path::new("./tmp_bandit.json"),
    );
    assert!(
        load_result.is_err(),
        "load should fail, since TestArm{{num: 3}} could not be found, but was {:?}",
        load_result
    );
}

#[test]
fn test_logging_update() {
    let test_file = Path::new(common::LOG_UPDATE_FILE);
    if test_file.exists() {
        remove_file(test_file).unwrap();
    }

    let arms = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let bandit_config = BanditConfig {
        log_file: Some(PathBuf::from(common::LOG_UPDATE_FILE)),
    };
    let mut ucb = UCB::new(arms.clone(), bandit_config, UcbConfig { alpha: 1.0 });

    ucb.update(arms[0], 1.);
    ucb.update(arms[1], 1.);
    ucb.update(arms[2], 1.);
    ucb.update(arms[3], 1.);

    let log_content = common::read_file_content(common::LOG_UPDATE_FILE);

    let re = Regex::new(
        r#"^UPDATE;arm:0;\d{13}
UPDATE;arm:1;\d{13}
UPDATE;arm:2;\d{13}
UPDATE;arm:3;\d{13}
$"#,
    )
    .expect("compiled regex");

    assert!(
        re.is_match(&log_content),
        "log file did not match expected, was {}",
        &log_content
    );
}

#[test]
fn test_logging_select() {
    let test_file = Path::new(common::LOG_SELECT_FILE);
    if test_file.exists() {
        remove_file(test_file).unwrap();
    }

    let arms = vec![
        TestArm { num: 0 },
        TestArm { num: 1 },
        TestArm { num: 2 },
        TestArm { num: 3 },
    ];
    let bandit_config = BanditConfig {
        log_file: Some(PathBuf::from(common::LOG_SELECT_FILE)),
    };
    let ucb = UCB::new(arms.clone(), bandit_config, UcbConfig { alpha: 1.0 });

    let select1 = ucb.select_arm();
    let select2 = ucb.select_arm();
    let select3 = ucb.select_arm();

    let log_content = common::read_file_content(common::LOG_SELECT_FILE);

    let re = Regex::new(&format!(
        r#"^SELECT;{};\d{{13}}
SELECT;{};\d{{13}}
SELECT;{};\d{{13}}
$"#,
        select1.ident(),
        select2.ident(),
        select3.ident()
    ))
    .expect("compiled regex");

    assert!(
        re.is_match(&log_content),
        "log file did not match expected, was {}",
        &log_content
    );
}
