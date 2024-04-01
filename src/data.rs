use crate::tensor::Tensor;

/// Returns a tuple of (images, labels)
pub fn load_mnist() -> (Tensor, Tensor) {
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
