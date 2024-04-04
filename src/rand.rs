static mut SEED: u32 = 123;

pub fn set_seed(seed: u32) {
    unsafe {
        SEED = seed;
    }
}

pub fn get_seed() -> u32 {
    unsafe { SEED }
}

#[derive(Clone)]
pub struct Random {
    seed: u32,
}

impl Default for Random {
    fn default() -> Self {
        Self { seed: get_seed() }
    }
}

pub fn new() -> Random {
    Default::default()
}

pub fn from_seed(seed: u32) -> Random {
    Random::from_seed(seed)
}

impl Random {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from_seed(seed: u32) -> Self {
        Self { seed }
    }

    pub fn nextu32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_add(0x9E3779B9);
        set_seed(self.seed);
        let mut x = self.seed;
        x ^= x >> 16;
        x = x.wrapping_mul(0x21F0AAAD);
        x ^= x >> 15;
        x = x.wrapping_mul(0x735A2D97);
        x ^= x >> 15;
        x
    }

    pub fn gen(&mut self) -> f32 {
        self.nextu32() as f32 / u32::MAX as f32
    }

    pub fn bool(&mut self) -> bool {
        self.nextu32() % 2 == 0
    }

    pub fn norm(&mut self) -> f32 {
        (-2.0 * self.gen().ln()).sqrt() * (std::f32::consts::TAU * self.gen()).cos()
    }

    pub fn gen_range(&mut self, range: std::ops::Range<f32>) -> f32 {
        self.gen() * (range.end - range.start) + range.start
    }

    pub fn gen_range_u32(&mut self, range: std::ops::Range<u32>) -> u32 {
        self.nextu32() % (range.end - range.start) + range.start
    }

    pub fn gen_range_i32(&mut self, range: std::ops::Range<i32>) -> i32 {
        (self.nextu32() % (range.end - range.start) as u32) as i32 + range.start
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }
}

pub trait Shuffle {
    fn shuffle(&mut self, r: &mut Random);
}

impl<T> Shuffle for [T] {
    fn shuffle(&mut self, rng: &mut Random) {
        for i in 0..self.len() {
            let j = rng.gen_range_u32(i as u32..self.len() as u32);
            self.swap(i, j as usize);
        }
    }
}

impl<I: Iterator> Shuffle for I {
    fn shuffle(&mut self, rng: &mut Random) {
        let slf = self as *mut Self;
        let len = self.size_hint().1.unwrap() as u32;
        for (i, mut a) in unsafe { &mut *slf }.by_ref().enumerate() {
            let j = rng.gen_range_u32(i as u32..len) as usize;
            let b = &mut self.nth(j).unwrap();
            std::mem::swap(&mut a, b);
        }
    }
}
