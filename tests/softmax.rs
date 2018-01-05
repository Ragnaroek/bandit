
extern crate bandit;

use bandit::bandit::{MultiArmedBandit};
use bandit::softmax::{AnnealingSoftmax};
use std::collections::{HashMap};

const NUM_SELECTS : u32 = 100_000;
const EPSILON : u32 = (NUM_SELECTS as f64 * 0.005) as u32;

#[test]
pub fn test_select_arm() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm = AnnealingSoftmax::new(arms);

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
fn test_select_arm_with_given_values() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let mut values = HashMap::new();
    values.insert(arms[0].clone(), 0.2);
    values.insert(arms[1].clone(), 0.2);
    values.insert(arms[2].clone(), 0.1);
    values.insert(arms[3].clone(), 0.5);
    let sm = AnnealingSoftmax::new_with_values(arms.clone(), values);

    let mut selects = HashMap::new();
    for _ in 0..NUM_SELECTS {
        let arm_selected = sm.select_arm();
        *selects.entry(arm_selected).or_insert(0) += 1;
    }

    assert!(selects[&arms[3]] > selects[&arms[0]]);
    assert!(selects[&arms[3]] > selects[&arms[1]]);

    assert!(selects[&arms[0]] > selects[&arms[2]]);
    assert!(selects[&arms[1]] > selects[&arms[2]]);
}

//Helper

fn abs_select(prop: f64) -> u32 {
    return (NUM_SELECTS as f64 * prop) as u32;
}

fn assert_prop(expected_count: u32, v: u32, arm: TestArm) {
    assert!(expected_count - EPSILON < v && v < expected_count + EPSILON, "expected {}+-{}, got {} arm {:?}", expected_count, EPSILON, v, arm);
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct TestArm {
    num: u32
}
