use crate::rand;
use crate::tensor::Tensor;

pub struct Dataset {
    input: Tensor,
    target: Tensor,
}

impl Dataset {
    pub fn new(input: Tensor, target: Tensor) -> Self {
        assert_eq!(
            input.dim(),
            2,
            "dataset input must have shape (input_count, input_size)"
        );
        assert_eq!(
            target.dim(),
            2,
            "dataset target must have shape (target_count, target_size)"
        );
        assert_eq!(
            input.size(0),
            target.size(0),
            "dataset must have same number of inputs as targets"
        );
        Self { input, target }
    }

    pub fn split_train_test_size(&self, train_size: usize) -> (Dataset, Dataset) {
        (
            Dataset {
                input: self.input.slice(..train_size),
                target: self.target.slice(..train_size),
            },
            Dataset {
                input: self.input.slice(train_size..),
                target: self.target.slice(train_size..),
            },
        )
    }

    pub fn split_train_test(&self, train_ratio: f32) -> (Dataset, Dataset) {
        let train_size = (self.batches() as f32 * train_ratio) as usize;
        self.split_train_test_size(train_size)
    }

    pub fn split_train_test_size_unbatched(&self, train_size: usize) -> (Dataset, Batch) {
        let test_size = (self.batches() - train_size) * self.batch_size();
        let mut test_input = self.input.slice(train_size..);
        let mut test_target = self.target.slice(train_size..);
        test_input.reshape_(&[test_size, 0]);
        test_target.reshape_(&[test_size, 0]);
        (
            Dataset {
                input: self.input.slice(..train_size),
                target: self.target.slice(..train_size),
            },
            Batch {
                input: test_input,
                target: test_target,
            },
        )
    }

    pub fn split_train_test_unbatched(&self, train_ratio: f32) -> (Dataset, Batch) {
        let train_size = (self.batches() as f32 * train_ratio) as usize;
        self.split_train_test_size_unbatched(train_size)
    }

    pub fn batch(&mut self, batch_size: usize) {
        let input_batches = self.input.size(0) / batch_size;
        let target_batches = self.target.size(0) / batch_size;
        self.input.reshape_(&[input_batches, batch_size, 0]);
        self.target.reshape_(&[target_batches, batch_size, 0]);
    }

    pub fn shuffle(&mut self, rng: &mut rand::Random) {
        let mut rng2 = rng.clone();
        self.input.shuffle_at_(0, rng);
        self.target.shuffle_at_(0, &mut rng2);
    }

    pub fn input_size(&self) -> usize {
        self.input.size(2)
    }

    pub fn target_size(&self) -> usize {
        self.target.size(2)
    }

    pub fn batch_size(&self) -> usize {
        self.input.size(1)
    }

    pub fn batches(&self) -> usize {
        self.input.size(0)
    }

    pub fn iter(&self) -> DatasetIter {
        self.into_iter()
    }
}

pub struct Batch {
    input: Tensor,
    target: Tensor,
}

impl Batch {
    pub fn iter(&self) -> BatchIter {
        self.into_iter()
    }
}

pub struct BatchIter<'a> {
    batch: &'a Batch,
    index: usize,
}

impl<'a> Iterator for BatchIter<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.batch.input.size(0) {
            None
        } else {
            let input = self.batch.input.idx(self.index);
            let target = self.batch.target.idx(self.index);
            self.index += 1;
            Some((input, target))
        }
    }
}

impl<'a> IntoIterator for &'a Batch {
    type Item = (Tensor, Tensor);
    type IntoIter = BatchIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        BatchIter {
            batch: self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = Batch;
    type IntoIter = DatasetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DatasetIter {
            dataset: self,
            index: 0,
        }
    }
}

pub struct DatasetIter<'a> {
    dataset: &'a Dataset,
    index: usize,
}

impl<'a> Iterator for DatasetIter<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.input.size(0) {
            None
        } else {
            let input = self.dataset.input.idx(self.index);
            let target = self.dataset.target.idx(self.index);
            self.index += 1;
            Some(Batch { input, target })
        }
    }
}

impl From<(Tensor, Tensor)> for Dataset {
    fn from((input, target): (Tensor, Tensor)) -> Self {
        Self::new(input, target)
    }
}

impl From<Dataset> for (Tensor, Tensor) {
    fn from(dataset: Dataset) -> (Tensor, Tensor) {
        (dataset.input, dataset.target)
    }
}

pub fn load_mnist() -> Dataset {
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
            .iter()
            .map(|x| (*x as f32) / 255.0)
            .collect::<Vec<f32>>(),
        vec![image_len, size],
    );
    let mut target = Tensor::zeros(&[image_len, 10]);
    for i in 0..image_len {
        target[i * 10 + labels[8 + i] as usize] = 1.0;
    }

    (input, target).into()
}

pub fn xor(n: usize) -> Dataset {
    let mut rng = rand::new();
    let mut input = Tensor::zeros(&[n, 2]);
    let mut target = Tensor::zeros(&[n, 1]);
    for i in 0..n {
        let r = rng.nextu32();
        let a = r & 1;
        let b = (r & 2) >> 1;
        input[i * 2] = a as f32;
        input[i * 2 + 1] = b as f32;
        target[i] = (a ^ b) as f32;
    }
    (input, target).into()
}
