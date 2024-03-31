#![allow(dead_code)]
pub mod rand;
pub mod tensor;
pub use tensor::Slice;
pub use tensor::Tensor;
pub use tensor::TensorRef;
pub mod nn;
pub mod seed;
pub use nn::*;
pub mod layer;
pub use layer::*;

/// Returns a tuple of (images, labels)
fn load_mnist() -> (Tensor, Tensor) {
    let images = std::fs::read("data/images.dat").unwrap();
    let magic = u32::from_be_bytes([images[0], images[1], images[2], images[3]]);
    assert_eq!(magic, 2051, "Invalid MNIST image file");
    let image_len = u32::from_be_bytes([images[4], images[5], images[6], images[7]]) as usize;
    let width = u32::from_be_bytes([images[8], images[9], images[10], images[11]]) as usize;
    let height = u32::from_be_bytes([images[12], images[13], images[14], images[15]]) as usize;
    let size = width * height;

    let labels = std::fs::read("data/labels.dat").unwrap();
    let magic = u32::from_be_bytes([labels[0], labels[1], labels[2], labels[3]]);
    assert_eq!(magic, 2049, "Invalid MNIST label file");
    let label_len = u32::from_be_bytes([labels[4], labels[5], labels[6], labels[7]]) as usize;
    assert_eq!(
        image_len, label_len,
        "MNIST image and label files must have the same number of entries"
    );

    let input = Tensor::raw(
        images[16..]
            .into_iter()
            .map(|x| (*x as f32) / 255.0)
            .collect::<Vec<f32>>(),
        vec![image_len, size],
    );
    let mut targets = Tensor::zeros(&[image_len, 10]);
    for i in 0..image_len {
        targets[i * 10 + labels[8 + i] as usize] = 1.0;
    }

    (input, targets)
}

fn main() {
    let (mut inputs, mut targets) = load_mnist();

    let learning_rate = 0.003;
    let mut rng = rand::new();
    let test_inputs = inputs.slice(inputs.size(0) - 32, 0);
    let test_targets = targets.slice(targets.size(0) - 32, 0);

    let mut accuracy;

    let mut model = Model::new();
    model.add_layer(Linear::new(inputs.size(1), 16));
    model.add_layer(Relu {});
    model.add_layer(Linear::output(targets.size(1)));
    model.add_layer(Softmax {});

    inputs.resize_(&[inputs.size(0) / 32 - 1, 32, inputs.size(1)]);
    targets.resize_(&[targets.size(0) / 32 - 1, 32, targets.size(1)]);

    for epoch in 0..8 {
        let batches = 256;
        let pos = rng.gen_range_u32(0..inputs.size(0) as u32 - batches as u32) as usize;
        accuracy = 0.0;
        for idx in pos..pos + batches {
            for batch_idx in 0..inputs.size(1) {
                let x = inputs.at(&[idx, batch_idx]);
                let y = targets.at(&[idx, batch_idx]);
                let a2 = model.step(&x, &y);

                accuracy += y[a2.argmax()];
                if batch_idx == 0 && idx == pos + batches - 1 {
                    print!("Epoch {}: {}", epoch, -a2[y.argmax()].max(1e-5).ln() as f32);
                }
            }
            // Update
            if idx == pos + batches - 1 {
                accuracy /= (batches * inputs.size(1)) as f32;
                println!(" ({:.3}%)", accuracy * 100.0);
            }
            model.update(learning_rate);
        }
    }

    for i in 0..32 {
        let x = test_inputs.idx(i);
        let a2 = model.forward(&x);
        println!(
            "Guess: {} | Target: {}",
            a2.argmax(),
            test_targets.idx(i).argmax()
        );
    }
}
