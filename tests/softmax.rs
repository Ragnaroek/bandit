
extern crate bandit;

use bandit::bandit::{MultiArmedBandit};
use bandit::softmax::{AnnealingSoftmax};

#[derive(Hash, PartialEq, Eq, Clone)]
struct TestArm {
    num: u32
}

#[test]
pub fn test_select_arm() {
    let arms = vec![TestArm{num: 0}, TestArm{num: 1}, TestArm{num: 2}, TestArm{num: 3}];
    let sm = AnnealingSoftmax::new(arms);

    //TODO do some rounds of arm selection and check that arm is selected according to
    // current probability (ignoring annealing + count = fixed for this test)
    let arm_selected = sm.select_arm();
}
