static mut SEED: u32 = 0;

pub fn set(seed: u32) {
    unsafe {
        SEED = seed;
    }
}

pub fn get() -> u32 {
    unsafe { SEED }
}
