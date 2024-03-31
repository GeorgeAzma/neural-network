use crate::rand::Random;
use crate::rand::Shuffle;
use core::fmt;
use std::ops;

pub trait Slice<T> {
    fn as_slice(&self) -> &[T];
    fn as_slice_mut(&mut self) -> &mut [T];
}

impl Slice<f32> for &[f32] {
    fn as_slice(&self) -> &[f32] {
        *self
    }

    fn as_slice_mut(&mut self) -> &mut [f32] {
        panic!("Cannot mutate immutable slice");
    }
}

impl Slice<usize> for &[usize] {
    fn as_slice(&self) -> &[usize] {
        *self
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        panic!("Cannot mutate immutable slice");
    }
}

impl Slice<f32> for &mut [f32] {
    fn as_slice(&self) -> &[f32] {
        *self
    }

    fn as_slice_mut(&mut self) -> &mut [f32] {
        *self
    }
}

impl Slice<usize> for &mut [usize] {
    fn as_slice(&self) -> &[usize] {
        *self
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        *self
    }
}

impl Slice<f32> for Vec<f32> {
    fn as_slice(&self) -> &[f32] {
        self
    }

    fn as_slice_mut(&mut self) -> &mut [f32] {
        self
    }
}

impl Slice<usize> for Vec<usize> {
    fn as_slice(&self) -> &[usize] {
        self
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        self
    }
}

#[derive(Clone, Default)]
pub struct Tensor<T: Slice<f32> = Vec<f32>, S: Slice<usize> = Vec<usize>> {
    data: T,
    shape: S,
}

pub type TensorRef<'a> = Tensor<&'a [f32], &'a [usize]>;
pub type TensorMut<'a> = Tensor<&'a mut [f32], &'a mut [usize]>;

impl<T: Slice<f32>, S: Slice<usize>> Tensor<T, S> {
    pub fn view(data: T, shape: S) -> Self {
        Self { data, shape }
    }

    pub fn size(&self, dim: usize) -> usize {
        self.shape.as_slice()[dim]
    }

    pub fn dim(&self) -> usize {
        self.shape.as_slice().len()
    }

    pub fn numel(&self) -> usize {
        self.data.as_slice().len()
    }

    pub fn iter(&self) -> std::slice::Iter<f32> {
        self.data.as_slice().iter()
    }

    pub fn matmul<T2: Slice<f32>, S2: Slice<usize>>(&self, other: &Tensor<T2, S2>) -> Tensor {
        let s1 = self.shape.as_slice();
        let s2 = other.shape.as_slice();
        if s1.len() == 0 || s2.len() == 0 {
            return Tensor::default();
        }
        let dim0 = if s1.len() == 1 { 1 } else { s1[s1.len() - 2] };
        let dim1 = s1[s1.len() - 1];
        let other_dim0 = if s2.len() == 1 { 1 } else { s2[s2.len() - 2] };
        let other_dim1 = s2[s2.len() - 1];
        assert!(
            dim1 == other_dim0,
            "cannot multiply matrices with incompatible shapes ({}x{}) * ({}x{})",
            dim0,
            dim1,
            other_dim0,
            other_dim1,
        );
        let mut result = Tensor::zeros(&[dim0, other_dim1]);
        for i in 0..dim0 {
            for j in 0..other_dim1 {
                for k in 0..dim1 {
                    result.data[i * dim0 + j] += self.data.as_slice()[i * dim1 + k]
                        * other.data.as_slice()[k * other_dim1 + j];
                }
            }
        }
        result
    }

    pub fn outer<T2: Slice<f32>, S2: Slice<usize>>(&self, other: &Tensor<T2, S2>) -> Tensor {
        let mut result = Tensor::zeros(&[self.numel(), other.numel()]);
        for i in 0..self.numel() {
            for j in 0..other.numel() {
                result.data[i * other.numel() + j] =
                    self.data.as_slice()[i] * other.data.as_slice()[j];
            }
        }
        result
    }

    pub fn take(&self, index: &[usize]) -> Tensor {
        let mut data = Vec::new();
        data.resize(index.len(), 0.0);
        for i in 0..index.len() {
            data[i] = self.data.as_slice()[index[i]];
        }
        Tensor {
            data,
            shape: self.shape.as_slice().to_vec(),
        }
    }

    pub fn as_ref(&self) -> Tensor<&[f32], &[usize]> {
        Tensor::<&[f32], &[usize]> {
            data: self.data.as_slice(),
            shape: self.shape.as_slice(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn at(&self, index: &[usize]) -> TensorRef {
        let mut numel: usize = self.numel();
        let mut idx_start = 0;
        for (i, idx) in index.into_iter().enumerate() {
            numel /= self.shape.as_slice()[i];
            idx_start += idx * numel;
        }
        Tensor::view(
            &self.data.as_slice()[idx_start..idx_start + numel],
            &self.shape.as_slice()[index.len()..],
        )
    }

    pub fn to_owned(&self) -> Tensor {
        Tensor::new(self.data.as_slice(), self.shape.as_slice())
    }

    pub fn slice(&self, start: usize, end: usize) -> Tensor {
        let mut result = self.to_owned();
        result.slice_(start, end);
        result
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let mut result = self.to_owned();
        result.reshape_(shape);
        result
    }

    pub fn flatten(&self) -> Tensor {
        let mut result = self.to_owned();
        result.flatten_();
        result
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let mut result = self.to_owned();
        result.transpose_(dim0, dim1);
        result
    }

    pub fn t(&self) -> Tensor {
        let mut result = self.to_owned();
        result.t_();
        result
    }

    pub fn squeeze(&self) -> Tensor {
        let mut result = self.to_owned();
        result.squeeze_();
        result
    }

    pub fn squeeze_idx(&self, dim: usize) -> Tensor {
        let mut result = self.to_owned();
        result.squeeze_idx_(dim);
        result
    }

    pub fn squeeze_at(&self, dims: &[usize]) -> Tensor {
        let mut result = self.to_owned();
        result.squeeze_at_(dims);
        result
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let mut result = self.to_owned();
        result.unsqueeze_(dim);
        result
    }

    pub fn cat(&self, other: &Tensor<T, S>, dim: usize) -> Tensor {
        let mut result = self.to_owned();
        result.cat_(other, dim);
        result
    }

    pub fn apply(&self, f: impl FnMut(f32) -> f32) -> Tensor {
        let mut result = self.to_owned();
        result.apply_(f);
        result
    }

    pub fn softmax(&self) -> Tensor {
        let mut result = self.to_owned();
        result.softmax_();
        result
    }

    pub fn value(&self, index: &[usize]) -> f32 {
        assert_eq!(
            index.len(),
            self.dim(),
            "index must have the same length as the tensor dimensions"
        );
        self.at(index)[0]
    }

    pub fn shuffle(&self, rng: &mut Random) -> Tensor {
        let mut result = self.to_owned();
        result.shuffle_(rng);
        result
    }

    pub fn resize(&self, shape: &[usize]) -> Tensor {
        let mut result = self.to_owned();
        result.resize_(shape);
        result
    }

    pub fn idx(&self, index: usize) -> Tensor<&[f32], &[usize]> {
        let numel: usize = self.numel() / self.shape.as_slice()[0];
        let idx = index * numel;
        Tensor::view(
            &self.data.as_slice()[idx..idx + numel],
            &self.shape.as_slice()[1..],
        )
    }

    pub fn fill(&mut self, value: f32) -> Tensor {
        let mut result = self.to_owned();
        result.fill_(value);
        result
    }

    pub fn set(&mut self, data: &[f32]) -> Tensor {
        let mut result = self.to_owned();
        result.set_(data);
        result
    }

    pub fn argmax(&self) -> usize {
        let mut max = f32::NEG_INFINITY;
        let mut idx = 0;
        for (i, &d) in self.data.as_slice().iter().enumerate() {
            if d > max {
                max = d;
                idx = i;
            }
        }
        idx
    }
}

impl Tensor<&mut [f32], &mut [usize]> {
    pub fn iter_mut(&mut self) -> std::slice::IterMut<f32> {
        self.data.iter_mut()
    }

    pub fn at_mut(&mut self, index: &[usize]) -> Tensor<&mut [f32], &mut [usize]> {
        let mut numel: usize = self.numel();
        let mut idx_start = 0;
        for (i, idx) in index.into_iter().enumerate() {
            numel /= self.shape[i];
            idx_start += idx * numel;
        }
        Tensor::view(
            &mut self.data[idx_start..idx_start + numel],
            &mut self.shape[index.len()..],
        )
    }

    pub fn value_mut(&mut self, index: &[usize]) -> &mut f32 {
        assert_eq!(
            index.len(),
            self.dim(),
            "index must have the same length as the tensor dimensions"
        );
        &mut self.at_mut(index).data[0]
    }

    pub fn idx_mut(&mut self, index: usize) -> Tensor<&mut [f32], &mut [usize]> {
        let numel: usize = self.numel() / self.shape[0];
        let idx = index * numel;
        Tensor::view(&mut self.data[idx..idx + numel], &mut self.shape[1..])
    }
}

impl Tensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        Self {
            data: data.to_vec(),
            shape: shape.to_vec(),
        }
    }

    pub fn raw(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn array(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
            shape: vec![data.len()],
        }
    }

    pub fn matrix_from_vec(data: &[Vec<f32>]) -> Self {
        Self {
            data: data.into_iter().cloned().flatten().collect(),
            shape: vec![data.len(), data[0].len()],
        }
    }

    pub fn matrix<const N: usize>(data: &[[f32; N]]) -> Self {
        Self {
            data: data.into_iter().cloned().flatten().collect(),
            shape: vec![data.len(), N],
        }
    }

    pub fn zeros(dims: &[usize]) -> Self {
        Self {
            data: vec![0.0; dims.iter().product()],
            shape: dims.to_vec(),
        }
    }

    pub fn ones(dims: &[usize]) -> Self {
        Self {
            data: vec![1.0; dims.iter().product()],
            shape: dims.to_vec(),
        }
    }

    pub fn rand(dims: &[usize]) -> Self {
        let mut rng = crate::rand::new();
        Self {
            data: (0..dims.iter().product()).map(|_| rng.gen()).collect(),
            shape: dims.to_vec(),
        }
    }

    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let n = ((end - start) / step).round() as usize;
        data.resize(n, Default::default());
        let mut x = start;
        for i in 0..n {
            data[i] = x;
            x += step;
        }
        Self {
            data,
            shape: vec![n],
        }
    }

    pub fn linspace(start: f32, end: f32, steps: usize) -> Self {
        let step = (end - start) / (steps - 1) as f32;
        Self::arange(start, end, step)
    }

    pub fn eye(n: usize, m: usize) -> Self {
        let mut result = Self::zeros(&[n, m]);
        let dim = n.min(m);
        for i in 0..dim {
            result.data[i * m + i] = 1.0;
        }
        result
    }

    pub fn identity(dim: usize) -> Self {
        Self::eye(dim, dim)
    }

    pub fn full(shape: &[usize], fill_value: f32) -> Self {
        Self {
            data: vec![fill_value; shape.iter().product()],
            shape: shape.to_vec(),
        }
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<f32> {
        self.data.iter_mut()
    }

    pub fn into_iter(self) -> std::vec::IntoIter<f32> {
        self.data.into_iter()
    }

    pub fn idx_mut(&mut self, index: usize) -> Tensor<&mut [f32], &mut [usize]> {
        let numel: usize = self.numel() / self.shape[0];
        let idx = index * numel;
        Tensor::view(&mut self.data[idx..idx + numel], &mut self.shape[1..])
    }

    pub fn at_mut(&mut self, index: &[usize]) -> Tensor<&mut [f32], &mut [usize]> {
        let mut numel: usize = self.numel();
        let mut idx_start = 0;
        for (i, idx) in index.into_iter().enumerate() {
            numel /= self.shape[i];
            idx_start += idx * numel;
        }
        Tensor::view(
            &mut self.data[idx_start..idx_start + numel],
            &mut self.shape[index.len()..],
        )
    }

    pub fn as_mut(&mut self) -> Tensor<&mut [f32], &mut [usize]> {
        Tensor::<&mut [f32], &mut [usize]> {
            data: self.data.as_slice_mut(),
            shape: self.shape.as_slice_mut(),
        }
    }

    pub fn value_mut(&mut self, index: &[usize]) -> &mut f32 {
        assert_eq!(
            index.len(),
            self.dim(),
            "index must have the same length as the tensor dimensions"
        );
        &mut self.at_mut(index).data[0]
    }
}

// In-place operations
impl Tensor {
    pub fn slice_(&mut self, start: usize, mut end: usize) {
        if end == 0 {
            end = self.shape[0];
        }
        assert!(start < end, "start must be less than end");
        let start_idx = start * self.numel() / self.shape[0];
        let end_idx = end * self.numel() / self.shape[0];
        self.shape[0] = end - start;
        self.data = self.data[start_idx..end_idx].to_vec();
    }

    pub fn flatten_(&mut self) {
        self.shape = vec![self.numel()];
    }

    pub fn matmul_<T: Slice<f32>, S: Slice<usize>>(&mut self, other: &Tensor<T, S>) {
        *self = self.matmul(other);
    }

    pub fn reshape_(&mut self, shape: &[usize]) {
        assert_eq!(
            self.numel(),
            shape.iter().product(),
            "new shape must have the same number of elements as the old shape"
        );
        self.shape = shape.to_vec();
    }

    pub fn transpose_(&mut self, dim0: usize, dim1: usize) {
        assert!(
            self.shape.len() >= 2,
            "cannot transpose tensor with less than 2 dimensions"
        );
        for i in 0..self.shape[dim0] {
            for j in 0..self.shape[dim1] {
                let idx = i * self.shape[dim1] + j;
                let idx_t = j * self.shape[dim0] + i;
                self.data.swap(idx, idx_t);
            }
        }
        self.shape.swap(dim0, dim1);
    }

    pub fn t_(&mut self) {
        assert!(
            self.shape.len() >= 2,
            "cannot transpose tensor with less than 2 dimensions"
        );
        let dim0 = self.shape.len() - 2;
        let dim1 = self.shape.len() - 1;
        self.transpose_(dim0, dim1);
    }

    pub fn squeeze_(&mut self) {
        self.shape.retain(|&x| x != 1);
    }

    pub fn squeeze_idx_(&mut self, dim: usize) {
        if self.shape[dim] == 1 {
            self.shape.remove(dim);
        }
    }

    pub fn squeeze_at_(&mut self, dims: &[usize]) {
        self.shape = self
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| {
                if dims.contains(&i) && x == 1 {
                    None
                } else {
                    Some(x)
                }
            })
            .collect();
    }

    pub fn unsqueeze_(&mut self, dim: usize) {
        self.shape.insert(dim, 1);
    }

    pub fn cat_<T: Slice<f32>, S: Slice<usize>>(&mut self, other: &Tensor<T, S>, dim: usize) {
        assert_eq!(
            self.shape[dim],
            other.shape.as_slice()[dim],
            "cannot concatenate tensors with different dimensions"
        );
        self.data.extend_from_slice(other.data.as_slice());
        self.shape[dim] += other.shape.as_slice()[dim];
    }

    pub fn apply_(&mut self, mut f: impl FnMut(f32) -> f32) {
        self.data.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn softmax_(&mut self) {
        let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        self.apply_(|x| (x - max).exp());
        *self /= self.data.iter().sum::<f32>();
    }

    pub fn shuffle_(&mut self, rng: &mut Random) {
        let numel = self.numel() / self.shape[0];
        for i in 0..self.shape[0] {
            let r = rng.gen_range_u32(i as u32..self.shape[0] as u32);
            for j in 0..numel {
                let idx = i * numel + j;
                let idx_t = r as usize * numel + j;
                self.data.swap(idx, idx_t);
            }
        }
    }

    pub fn resize_(&mut self, shape: &[usize]) {
        let numel: usize = shape.iter().product();
        if numel == self.numel() {
            self.reshape_(shape);
            return;
        }
        self.data.resize(numel, 0.0);
        self.shape = shape.to_vec();
    }

    pub fn fill_(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn set_(&mut self, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.numel(),
            "data must have the same number of elements as the tensor"
        );
        self.data.copy_from_slice(data);
    }
}

impl Tensor<&mut [f32], &mut [usize]> {
    pub fn apply_(&mut self, mut f: impl FnMut(f32) -> f32) {
        self.data.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn softmax_(&mut self) {
        let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        self.apply_(|x| (x - max).exp());
        *self /= self.data.iter().sum::<f32>();
    }

    pub fn shuffle_(&mut self, rng: &mut Random) {
        self.data.shuffle(rng);
    }

    pub fn fill_(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn set_(&mut self, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.numel(),
            "data must have the same number of elements as the tensor"
        );
        self.data.copy_from_slice(data);
    }
}

impl AsRef<[f32]> for Tensor {
    fn as_ref(&self) -> &[f32] {
        &self.data
    }
}

impl<I: std::slice::SliceIndex<[f32]>, T: Slice<f32>, S: Slice<usize>> ops::Index<I>
    for Tensor<T, S>
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.data.as_slice()[index]
    }
}

impl<I: std::slice::SliceIndex<[f32]>> ops::IndexMut<I> for Tensor {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<I: std::slice::SliceIndex<[f32]>> ops::IndexMut<I> for Tensor<&mut [f32], &mut [usize]> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl From<&Vec<f32>> for Tensor {
    fn from(data: &Vec<f32>) -> Self {
        Self::array(data)
    }
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        Self::array(&data)
    }
}

impl From<&Vec<Vec<f32>>> for Tensor {
    fn from(data: &Vec<Vec<f32>>) -> Self {
        Self::matrix_from_vec(data)
    }
}

impl From<&[Vec<f32>]> for Tensor {
    fn from(data: &[Vec<f32>]) -> Self {
        Self::matrix_from_vec(data)
    }
}

impl<const N: usize> From<&[[f32; N]]> for Tensor {
    fn from(data: &[[f32; N]]) -> Self {
        Self::matrix(data)
    }
}

impl From<Vec<Vec<f32>>> for Tensor {
    fn from(data: Vec<Vec<f32>>) -> Self {
        Self::matrix_from_vec(&data)
    }
}

impl From<&[f32]> for Tensor {
    fn from(data: &[f32]) -> Self {
        Self::array(data)
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(data: [f32; N]) -> Self {
        Self::array(&data)
    }
}

impl<T1, S1, T2, S2> ops::Mul<&Tensor<T2, S2>> for &Tensor<T1, S1>
where
    T1: Slice<f32>,
    S1: Slice<usize>,
    T2: Slice<f32>,
    S2: Slice<usize>,
{
    type Output = Tensor;

    fn mul(self, other: &Tensor<T2, S2>) -> Self::Output {
        assert_eq!(
            self.shape.as_slice(),
            other.shape.as_slice(),
            "cannot multiply tensors with different shapes"
        );
        let mut result = Tensor::zeros(&[self.numel()]);
        for i in 0..self.numel() {
            result.data[i] = self.data.as_slice()[i] * other.data.as_slice()[i];
        }
        result
    }
}

impl<T1, S1, T2, S2> ops::Div<&Tensor<T2, S2>> for &Tensor<T1, S1>
where
    T1: Slice<f32>,
    S1: Slice<usize>,
    T2: Slice<f32>,
    S2: Slice<usize>,
{
    type Output = Tensor;

    fn div(self, other: &Tensor<T2, S2>) -> Self::Output {
        let other = Tensor {
            data: other
                .data
                .as_slice()
                .iter()
                .map(|x| 1.0 / x)
                .collect::<Vec<f32>>(),
            shape: other.shape.as_slice().to_vec(),
        };
        self * &other
    }
}

impl<T1, S1, T2, S2> ops::Add<&Tensor<T2, S2>> for &Tensor<T1, S1>
where
    T1: Slice<f32>,
    S1: Slice<usize>,
    T2: Slice<f32>,
    S2: Slice<usize>,
{
    type Output = Tensor;

    fn add(self, other: &Tensor<T2, S2>) -> Self::Output {
        assert_eq!(
            self.shape.as_slice(),
            other.shape.as_slice(),
            "cannot add tensors with different shapes"
        );
        Tensor {
            data: self
                .data
                .as_slice()
                .iter()
                .zip(other.data.as_slice().iter())
                .map(|(a, b)| a + b)
                .collect(),
            shape: self.shape.as_slice().to_vec(),
        }
    }
}

impl<T1, S1, T2, S2> ops::Sub<&Tensor<T2, S2>> for &Tensor<T1, S1>
where
    T1: Slice<f32>,
    S1: Slice<usize>,
    T2: Slice<f32>,
    S2: Slice<usize>,
{
    type Output = Tensor;

    fn sub(self, other: &Tensor<T2, S2>) -> Self::Output {
        assert_eq!(
            self.shape.as_slice(),
            other.shape.as_slice(),
            "cannot subtract tensors with different shapes"
        );
        Tensor {
            data: self
                .data
                .as_slice()
                .iter()
                .zip(other.data.as_slice().iter())
                .map(|(a, b)| a - b)
                .collect(),
            shape: self.shape.as_slice().to_vec(),
        }
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Mul<f32> for &Tensor<T, S> {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Self::Output {
        Tensor {
            data: self.data.as_slice().iter().map(|x| x * scalar).collect(),
            shape: self.shape.as_slice().to_vec(),
        }
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Div<f32> for &Tensor<T, S> {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Self::Output {
        Tensor {
            data: self.data.as_slice().iter().map(|x| x / scalar).collect(),
            shape: self.shape.as_slice().to_vec(),
        }
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Add<f32> for &Tensor<T, S> {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Self::Output {
        Tensor {
            data: self.data.as_slice().iter().map(|x| x + scalar).collect(),
            shape: self.shape.as_slice().to_vec(),
        }
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Sub<f32> for &Tensor<T, S> {
    type Output = Tensor;

    fn sub(self, scalar: f32) -> Self::Output {
        Tensor {
            data: self.data.as_slice().iter().map(|x| x - scalar).collect(),
            shape: self.shape.as_slice().to_vec(),
        }
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Mul<&Tensor<T, S>> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: &Tensor<T, S>) -> Self::Output {
        tensor * self
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Div<&Tensor<T, S>> for f32 {
    type Output = Tensor;

    fn div(self, tensor: &Tensor<T, S>) -> Self::Output {
        Tensor {
            data: tensor.data.as_slice().iter().map(|x| self / *x).collect(),
            shape: tensor.shape.as_slice().to_vec(),
        }
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Add<&Tensor<T, S>> for f32 {
    type Output = Tensor;

    fn add(self, tensor: &Tensor<T, S>) -> Self::Output {
        tensor + self
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Sub<&Tensor<T, S>> for f32 {
    type Output = Tensor;

    fn sub(self, tensor: &Tensor<T, S>) -> Self::Output {
        Tensor {
            data: tensor.data.as_slice().iter().map(|x| self - x).collect(),
            shape: tensor.shape.as_slice().to_vec(),
        }
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::Neg for &Tensor<T, S> {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl ops::AddAssign<f32> for Tensor {
    fn add_assign(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|x| *x += scalar);
    }
}

impl ops::SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|x| *x -= scalar);
    }
}

impl ops::MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|x| *x *= scalar);
    }
}

impl ops::DivAssign<f32> for Tensor {
    fn div_assign(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|x| *x /= scalar);
    }
}

impl ops::AddAssign<f32> for Tensor<&mut [f32], &mut [usize]> {
    fn add_assign(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|x| *x += scalar);
    }
}

impl ops::SubAssign<f32> for Tensor<&mut [f32], &mut [usize]> {
    fn sub_assign(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|x| *x -= scalar);
    }
}

impl ops::MulAssign<f32> for Tensor<&mut [f32], &mut [usize]> {
    fn mul_assign(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|x| *x *= scalar);
    }
}

impl ops::DivAssign<f32> for Tensor<&mut [f32], &mut [usize]> {
    fn div_assign(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|x| *x /= scalar);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::AddAssign<&Tensor<T, S>> for Tensor {
    fn add_assign(&mut self, other: &Tensor<T, S>) {
        assert_eq!(
            self.shape,
            other.shape.as_slice(),
            "cannot add tensors with different shapes"
        );
        self.data
            .iter_mut()
            .zip(other.data.as_slice().iter())
            .for_each(|(a, b)| *a += b);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::SubAssign<&Tensor<T, S>> for Tensor {
    fn sub_assign(&mut self, other: &Tensor<T, S>) {
        assert_eq!(
            self.shape,
            other.shape.as_slice(),
            "cannot subtract tensors with different shapes"
        );
        self.data
            .iter_mut()
            .zip(other.data.as_slice().iter())
            .for_each(|(a, b)| *a -= b);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::MulAssign<&Tensor<T, S>> for Tensor {
    fn mul_assign(&mut self, other: &Tensor<T, S>) {
        assert_eq!(
            self.shape,
            other.shape.as_slice(),
            "cannot multiply tensors with different shapes"
        );
        self.data
            .iter_mut()
            .zip(other.data.as_slice().iter())
            .for_each(|(a, b)| *a *= b);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::DivAssign<&Tensor<T, S>> for Tensor {
    fn div_assign(&mut self, other: &Tensor<T, S>) {
        assert_eq!(
            self.shape,
            other.shape.as_slice(),
            "cannot divide tensors with different shapes"
        );
        self.data
            .iter_mut()
            .zip(other.data.as_slice().iter())
            .for_each(|(a, b)| *a /= b);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::AddAssign<&Tensor<T, S>>
    for Tensor<&mut [f32], &mut [usize]>
{
    fn add_assign(&mut self, other: &Tensor<T, S>) {
        assert_eq!(
            self.shape,
            other.shape.as_slice(),
            "cannot add tensors with different shapes"
        );
        self.data
            .iter_mut()
            .zip(other.data.as_slice().iter())
            .for_each(|(a, b)| *a += b);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::SubAssign<&Tensor<T, S>>
    for Tensor<&mut [f32], &mut [usize]>
{
    fn sub_assign(&mut self, other: &Tensor<T, S>) {
        assert_eq!(
            self.shape,
            other.shape.as_slice(),
            "cannot subtract tensors with different shapes"
        );
        self.data
            .iter_mut()
            .zip(other.data.as_slice().iter())
            .for_each(|(a, b)| *a -= b);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::MulAssign<&Tensor<T, S>>
    for Tensor<&mut [f32], &mut [usize]>
{
    fn mul_assign(&mut self, other: &Tensor<T, S>) {
        assert_eq!(
            self.shape,
            other.shape.as_slice(),
            "cannot multiply tensors with different shapes"
        );
        self.data
            .iter_mut()
            .zip(other.data.as_slice().iter())
            .for_each(|(a, b)| *a *= b);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> ops::DivAssign<&Tensor<T, S>>
    for Tensor<&mut [f32], &mut [usize]>
{
    fn div_assign(&mut self, other: &Tensor<T, S>) {
        assert_eq!(
            self.shape,
            other.shape.as_slice(),
            "cannot divide tensors with different shapes"
        );
        self.data
            .iter_mut()
            .zip(other.data.as_slice().iter())
            .for_each(|(a, b)| *a /= b);
    }
}

impl<T: Slice<f32>, S: Slice<usize>> fmt::Display for Tensor<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T: Slice<f32>, S: Slice<usize>> fmt::Debug for Tensor<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn vals<T: Slice<f32>, S: Slice<usize>>(
            slf: &Tensor<T, S>,
            depth: usize,
            i: usize,
            f: &fmt::Formatter<'_>,
        ) -> String {
            let s = slf.shape.as_slice();
            let d = slf.data.as_slice();
            let p: usize = if depth + 1 < s.len() {
                s[depth + 1..].iter().product()
            } else {
                1
            };
            let mut str = String::new();
            let pad = " ".repeat(depth * 2);
            for j in 0..s[depth] {
                if j == 0 {
                    if f.alternate() {
                        str += &pad;
                    }
                    str += "[";
                }
                if f.alternate() && depth != s.len() - 1 {
                    str += "\n";
                }
                let val = if depth + 1 < s.len() {
                    vals(slf, depth + 1, i + j * p, f)
                } else {
                    d[i + j * p].to_string()
                };
                str += &val;
                if j == s[depth] - 1 {
                    if f.alternate() && depth != s.len() - 1 {
                        str += "\n";
                        str += &pad;
                    }
                    str += "]";
                } else {
                    str += ", ";
                }
            }
            str
        }

        f.write_str(&vals(self, 0, 0, f))
    }
}
