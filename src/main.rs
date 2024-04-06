#![allow(dead_code)]
#![feature(portable_simd)]
pub mod rand;
pub mod tensor;
use std::time::Instant;

pub use tensor::Tensor;
pub mod nn;
pub use nn::*;
pub mod layer;
pub use layer::*;
pub mod data;
pub use data::*;

fn main() {
    let learning_rate = 2e-4;

    let mut dataset = load_mnist();
    dataset.batch(32);
    let (mut train, test) = dataset.split_train_test_size_unbatched(dataset.batches() - 1);

    let mut model = Model::new();
    model.add_layer(Layer::lin(train.input_size(), 8));
    model.add_layer(Layer::Gelu);
    model.add_layer(Layer::lin_out(train.target_size()));

    for epoch in 0..32 {
        let mut accuracy = 0.0;
        let time = Instant::now();
        train.shuffle(&mut rand::new());
        for batch in train.iter() {
            for (x, y) in batch.iter() {
                let a2 = model.step(&x, &y);
                accuracy += if y.argmax() == a2.argmax() { 1.0 } else { 0.0 };
            }
            model.update(learning_rate);
        }

        accuracy /= (train.batches() * train.batch_size()) as f32;
        let elapsed = Instant::now().duration_since(time).as_secs_f32() * 1000.0;
        println!(
            "Epoch {}: ({:.3}%) | {:.3}ms",
            epoch,
            accuracy * 100.0,
            elapsed
        );
    }

    let mut guesses = Vec::new();
    let mut targets = Vec::new();
    for (x, y) in test.iter() {
        guesses.push(model.forward(&x).argmax());
        targets.push(y.argmax());
    }
    println!("Guesses: {:?}", guesses);
    println!("Targets: {:?}", targets);
}
