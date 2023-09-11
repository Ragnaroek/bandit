extern crate bandit;
extern crate regex;

use bandit::Identifiable;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const NUM_SELECTS: u32 = 100_000;
pub static LOG_UPDATE_FILE: &str = "./tmp_log_update.csv";
pub static LOG_SELECT_FILE: &str = "./tmp_log_select.csv";
const EPSILON: u32 = (NUM_SELECTS as f64 * 0.005) as u32;

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct TestArm {
    pub num: u32,
}

impl Identifiable for TestArm {
    fn ident(&self) -> String {
        format!("arm:{}", self.num)
    }
}

pub fn abs_select(prop: f64) -> u32 {
    (f64::from(NUM_SELECTS) * prop) as u32
}

pub fn read_file_content(path: &str) -> String {
    let mut file = File::open(Path::new(path)).unwrap();
    let mut log_content = String::new();
    file.read_to_string(&mut log_content).unwrap();
    log_content
}

pub fn assert_prop(expected_count: u32, v: u32, arm: TestArm) {
    assert!(
        expected_count - EPSILON < v && v < expected_count + EPSILON,
        "expected {}+-{}, got {} arm {:?}",
        expected_count,
        EPSILON,
        v,
        arm
    );
}
