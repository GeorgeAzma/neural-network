#![allow(dead_code)]
#![feature(portable_simd)]
pub mod rand;
pub mod tensor;
use std::time::Instant;

pub use tensor::Slice;
pub use tensor::Tensor;
pub use tensor::TensorRef;
pub mod nn;
pub mod seed;
pub use nn::*;
pub mod layer;
pub use layer::*;
pub mod gpu;
pub use gpu::*;
pub mod data;
pub use data::*;

fn main() {
    let (mut inputs, mut targets) = load_mnist();

    let learning_rate = 0.0005;
    let mut rng = rand::new();
    // let test_inputs = inputs.slice(inputs.size(0) - 32, 0);
    // let test_targets = targets.slice(targets.size(0) - 32, 0);

    let mut accuracy;

    let mut model = Model::new();
    model.add_layer(Linear::new(inputs.size(1), 1024));
    model.add_layer(Relu {});
    model.add_layer(Linear::output(targets.size(1)));
    model.add_layer(Softmax {});

    inputs.resize_(&[inputs.size(0) / 32 - 1, 32, inputs.size(1)]);
    targets.resize_(&[targets.size(0) / 32 - 1, 32, targets.size(1)]);

    for epoch in 0..16 {
        let batches = 64;
        let pos = rng.gen_range_u32(0..inputs.size(0) as u32 - batches as u32) as usize;
        accuracy = 0.0;
        let time = Instant::now();
        for idx in pos..pos + batches {
            for batch_idx in 0..inputs.size(1) {
                let x = inputs.at(&[idx, batch_idx]);
                let y = targets.at(&[idx, batch_idx]);
                let a2 = model.step(&x, &y);

                accuracy += y[a2.argmax()];
                if batch_idx == 0 && idx == pos + batches - 1 {
                    print!("Epoch {}: {}", epoch, -a2[y.argmax()].max(1e-8).ln() as f32);
                }
            }
            // Update
            if idx == pos + batches - 1 {
                accuracy /= (batches * inputs.size(1)) as f32;
                print!(" ({:.3}%)", accuracy * 100.0);
            }
            model.update(learning_rate);
        }
        let elapsed = Instant::now().duration_since(time);
        println!("  {:.3}ms", elapsed.as_secs_f32() * 1000.0);
    }

    // for i in 0..32 {
    //     let x = test_inputs.idx(i);
    //     let a2 = model.forward(&x);
    //     println!(
    //         "Guess: {} | Target: {}",
    //         a2.argmax(),
    //         test_targets.idx(i).argmax()
    //     );
    // }
}
