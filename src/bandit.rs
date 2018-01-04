
pub trait MultiArmedBandit<A: Clone> {
    fn select_arm(&self) -> A;
    fn update(&self, arm: A, reward: f32);
}
