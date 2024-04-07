use crate::rand::Random;
use crate::rand::Shuffle;
use core::fmt;
use std::ops;
use std::ops::{Bound, RangeBounds};

#[macro_export]
macro_rules! tensor {
    (x: literal) => {
        Tensor::array(&[x as f32])
    };

    ($x: literal; $n: literal) => {
        Tensor::splat(&[$n as usize], $x as f32)
    };

    ($value:expr; $($dim:expr),*) => {{
        let shape = vec![$($dim),*];
        let data = vec![$value; shape.iter().product::<usize>()];
        Tensor::raw(data, shape)
    }};

    [$($x: literal), *] => {
        Tensor::array(&[$($x as f32),*])
    };

    ($([$([$([$($x:expr),* $(,)*]),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        let data = vec![$([$([$([$($x,)*],)*],)*],)*];
        let dim0 = data.len();
        let dim1 = if dim0 > 0 { data[0].len() } else { 0 };
        let dim2 = if dim1 > 0 { data[0][0].len() } else { 0 };
        let dim3 = if dim2 > 0 { data[0][0][0].len() } else { 0 };
        let flattened_data: Vec<f32> = data.into_iter().flatten().flatten().flatten().map(|x| x as f32).collect();
        Tensor::raw(flattened_data, vec![dim0, dim1, dim2, dim3])
    }};

    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        let data = vec![$([$([$($x,)*],)*],)*];
        let dim0 = data.len();
        let dim1 = if dim0 > 0 { data[0].len() } else { 0 };
        let dim2 = if dim1 > 0 { data[0][0].len() } else { 0 };
        Tensor::raw(data.into_iter().flatten().flatten().map(|x| x as f32).collect(), vec![dim0, dim1, dim2])
    }};

    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        let data = vec![$([$($x,)*],)*];
        let dim0 = data.len();
        let dim1 = if dim0 > 0 { data[0].len() } else { 0 };
        Tensor::raw(data.into_iter().flatten().map(|x| x as f32).collect(), vec![dim0, dim1])
    }};
}

#[derive(Clone, Default)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        Self::raw(data.to_vec(), shape.to_vec())
    }

    pub fn raw(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "data and shape mismatch"
        );
        Self { data, shape }
    }

    pub fn deep_clone(&self) -> Self {
        Self::raw(self.data.clone(), self.shape.clone())
    }

    pub fn array(data: &[f32]) -> Self {
        Self::new(data, &[data.len()])
    }

    pub fn matrix<const N: usize>(data: &[[f32; N]]) -> Self {
        Self::raw(
            data.iter().cloned().flatten().collect(),
            vec![data.len(), data[0].len()],
        )
    }

    pub fn zeros(dims: &[usize]) -> Self {
        Self::splat(dims, 0.0)
    }

    pub fn ones(dims: &[usize]) -> Self {
        Self::splat(dims, 1.0)
    }

    pub fn rand(dims: &[usize]) -> Self {
        let mut rng = crate::rand::new();
        Self::splat_with(dims, || rng.gen())
    }

    pub fn randn(dims: &[usize]) -> Self {
        let mut rng = crate::rand::new();
        Self::splat_with(dims, || rng.norm())
    }

    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let n = ((end - start) / step).round() as usize;
        data.resize(n, Default::default());
        let mut x = start;
        for d in data.iter_mut() {
            *d = x;
            x += step;
        }
        Self::raw(data, vec![n])
    }

    pub fn linspace(start: f32, end: f32, steps: usize) -> Self {
        let step = (end - start) / (steps - 1) as f32;
        Self::arange(start, end + step, step)
    }

    pub fn eye(n: usize, m: usize) -> Self {
        let mut result = Self::zeros(&[n, m]);
        let dim = n.min(m);
        for i in 0..dim {
            result[i * m + i] = 1.0;
        }
        result
    }

    pub fn identity(dim: usize) -> Self {
        Self::eye(dim, dim)
    }

    pub fn splat(shape: &[usize], fill_value: f32) -> Self {
        Self::raw(vec![fill_value; shape.iter().product()], shape.to_vec())
    }

    pub fn splat_with(shape: &[usize], f: impl FnMut() -> f32) -> Self {
        let mut data = vec![];
        data.resize_with(shape.iter().product(), f);
        Self::raw(data, shape.to_vec())
    }

    pub fn size(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn empty(&self) -> bool {
        self.numel() == 0
    }

    pub fn idx(&self, index: usize) -> Tensor {
        let m = self.numel() / self.size(0);
        Self::new(
            &self[index * m..][..m],
            if self.dim() == 1 {
                &[1]
            } else {
                &self.shape[1..]
            },
        )
    }

    pub fn slice<I: RangeBounds<usize>>(&self, index: I) -> Tensor {
        let mut shape = self.shape.clone();
        let start = match index.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start,
            Bound::Unbounded => 0,
        };
        let end = match index.end_bound() {
            Bound::Included(&end) => end + 1,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => shape[0],
        };
        shape[0] = end - start;
        let m = self.numel() / self.size(0);
        Self::new(&self[start * m..][..shape[0] * m], &shape)
    }

    pub fn index(&self, index: &[usize]) -> usize {
        let mut m = 1;
        let mut idx = 0;
        for (i, &indx) in index.iter().enumerate().rev() {
            idx += indx * m;
            m *= self.size(i);
        }
        idx
    }

    pub fn at(&self, index: &[usize]) -> Tensor {
        let new_shape = if index.len() < self.dim() {
            &self.shape[index.len()..]
        } else {
            &[1]
        };
        Tensor::new(
            &self[self.index(index)..][..new_shape.iter().product()],
            new_shape,
        )
    }

    pub fn get(&self, index: &[usize]) -> f32 {
        self[self.index(index)]
    }

    pub fn get_mut(&mut self, index: &[usize]) -> &mut f32 {
        let idx = self.index(index);
        &mut self[idx]
    }

    pub fn iter(&self) -> std::slice::Iter<f32> {
        self[..].iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<f32> {
        self[..].iter_mut()
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let mut result = self.deep_clone();
        result.reshape_(shape);
        result
    }

    pub fn resize(&self, shape: &[usize], value: f32) -> Tensor {
        let mut result = self.deep_clone();
        result.resize_(shape, value);
        result
    }

    pub fn resize_with(&self, shape: &[usize], f: impl FnMut() -> f32) -> Tensor {
        let mut result = self.deep_clone();
        result.resize_with_(shape, f);
        result
    }

    pub fn flatten(&self) -> Tensor {
        let mut result = self.deep_clone();
        result.flatten_();
        result
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let mut result = self.deep_clone();
        result.transpose_(dim0, dim1);
        result
    }

    pub fn t(&self) -> Tensor {
        let mut result = self.deep_clone();
        result.t_();
        result
    }

    pub fn squeeze(&self) -> Tensor {
        let mut result = self.deep_clone();
        result.squeeze_();
        result
    }

    pub fn squeeze_idx(&self, dim: usize) -> Tensor {
        let mut result = self.deep_clone();
        result.squeeze_idx_(dim);
        result
    }

    pub fn squeeze_at(&self, dims: &[usize]) -> Tensor {
        let mut result = self.deep_clone();
        result.squeeze_at_(dims);
        result
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let mut result = self.deep_clone();
        result.unsqueeze_(dim);
        result
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(
            self.dim() <= 2 && other.dim() <= 2,
            "cannot matrix multiply matrices with more than 2 dimensions"
        );
        assert!(
            self.dim() != 0 && other.dim() != 0,
            "cannot matrix multiply matrices with zero dimensions"
        );
        let dim0 = if self.dim() == 1 {
            1
        } else {
            self.size(self.dim() - 2)
        };
        let dim1 = self.size(self.dim() - 1);
        let other_dim0 = if other.dim() == 1 {
            1
        } else {
            other.size(other.dim() - 2)
        };
        let other_dim1 = other.size(other.dim() - 1);
        assert!(
            dim1 == other_dim0,
            "cannot matrix multiply matrices with incompatible shapes ({}x{}) * ({}x{})",
            dim0,
            dim1,
            other_dim0,
            other_dim1,
        );

        let mut result = Tensor::zeros(&[dim0, other_dim1]);
        for i in 0..dim0 {
            for j in 0..dim1 {
                result[i * dim1..][..other_dim1]
                    .iter_mut()
                    .zip(other[j * other_dim1..].iter())
                    .for_each(|(a, b)| *a += self[i * dim1 + j] * b);
            }
        }

        result
    }

    pub fn dot(&self, other: &Tensor) -> Tensor {
        // TODO: Use broadcasting instead
        assert!(
            self.dim() != 0 && other.dim() != 0,
            "cannot dot matrices with zero dimensions"
        );
        let dim0 = if self.dim() == 1 {
            1
        } else {
            self.size(self.dim() - 2)
        };
        let dim1 = self.size(self.dim() - 1);
        let other_dim0 = if other.dim() == 1 {
            1
        } else {
            other.size(other.dim() - 2)
        };
        let other_dim1 = other.size(other.dim() - 1);
        assert!(
            (dim0 == other_dim0 || dim0 == 1 || other_dim0 == 1) && dim1 == other_dim1,
            "cannot dot matrices with incompatible shapes ({}x{}) * ({}x{})",
            dim0,
            dim1,
            other_dim0,
            other_dim1,
        );

        let n = dim0.max(other_dim0);
        let mut result = Tensor::zeros(&[n]);
        if dim0 == other_dim0 {
            for i in 0..n * dim1 {
                result[i / dim1] += self[i] * other[i];
            }
        } else if dim0 == 1 {
            for i in 0..n {
                result[i] = other[i * dim1..][..dim1]
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| x * self[j])
                    .sum();
            }
        } else {
            for i in 0..n {
                result[i] = self[i * dim1..][..dim1]
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| x * other[j])
                    .sum();
            }
        }

        result
    }

    pub fn outer(&self, other: &Tensor) -> Tensor {
        let mut result = tensor!(0.0; self.numel(), other.numel());
        for i in 0..self.numel() {
            result[i * other.numel()..][..other.numel()]
                .iter_mut()
                .zip(other.iter())
                .for_each(|(a, b)| *a = self[i] * b);
        }
        result
    }

    pub fn take(&self, index: &[usize]) -> Tensor {
        let mut data = Vec::new();
        data.resize(index.len(), 0.0);
        for i in 0..index.len() {
            data[i] = self[index[i]];
        }
        Tensor::raw(data, self.shape.to_vec())
    }

    pub fn diag(&self) -> Tensor {
        assert!(
            self.dim() >= 2,
            "cannot extract diagonal from tensor with less than 2 dimensions"
        );
        let n = self.size(self.dim() - 1);
        let m = self.size(self.dim() - 2);
        let s = n.min(m);
        let diags = self.numel() / (n * m);
        let mut shape = self.shape.clone();
        shape.pop();
        *shape.last_mut().unwrap() = s;
        let mut result = Tensor::zeros(&shape);
        for d in 0..diags {
            for i in 0..s {
                result[d * s + i] = self[i * n + i];
            }
        }
        result
    }

    pub fn cat(&self, other: &Tensor, dim: usize) -> Tensor {
        let mut result = self.deep_clone();
        result.cat_(other, dim);
        result
    }

    pub fn apply(&self, f: impl FnMut(f32) -> f32) -> Tensor {
        let mut result = self.deep_clone();
        result.apply_(f);
        result
    }

    pub fn softmax(&self) -> Tensor {
        let mut result = self.deep_clone();
        result.softmax_();
        result
    }

    pub fn shuffle(&self, rng: &mut Random) -> Tensor {
        let mut result = self.deep_clone();
        result.shuffle_(rng);
        result
    }

    pub fn shuffle_at(&self, dim: usize, rng: &mut Random) -> Tensor {
        let mut result = self.deep_clone();
        result.shuffle_at_(dim, rng);
        result
    }

    pub fn fill(&mut self, value: f32) -> Tensor {
        let mut result = self.deep_clone();
        result.fill_(value);
        result
    }

    pub fn min(&self) -> f32 {
        self.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    pub fn max(&self) -> f32 {
        self.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn sum(&self) -> f32 {
        self.iter().sum()
    }

    pub fn prod(&self) -> f32 {
        self.iter().product()
    }

    pub fn argmin(&self) -> usize {
        let mut min = f32::INFINITY;
        let mut idx = 0;
        for (i, &d) in self.iter().enumerate() {
            if d < min {
                min = d;
                idx = i;
            }
        }
        idx
    }

    pub fn argmax(&self) -> usize {
        let mut max = f32::NEG_INFINITY;
        let mut idx = 0;
        for (i, &d) in self.iter().enumerate() {
            if d > max {
                max = d;
                idx = i;
            }
        }
        idx
    }
}

// In-place operations
impl Tensor {
    pub fn flatten_(&mut self) {
        self.shape = vec![self.numel()];
    }

    pub fn reshape_(&mut self, shape: &[usize]) {
        self.shape = shape.to_vec();
        let new_numel = shape.iter().product();
        if self.numel() != new_numel {
            let zero_pos = shape
                .iter()
                .position(|x| *x == 0)
                .expect("new shape must have the same number of elements as the old shape");
            let new_numel: usize = shape.iter().filter(|&x| *x != 0).product();
            assert!(
                new_numel != 0,
                "new shape contains multiple zeros, only single zero is allowed for auto reshaping"
            );
            assert!(
                self.numel() % new_numel == 0,
                "new shape contains zero, but it is not divisible for auto reshaping"
            );
            self.shape[zero_pos] = self.numel() / new_numel;
        }
        assert_eq!(
            self.numel(),
            self.shape.iter().product(),
            "new shape must have the same number of elements as the old shape"
        );
    }

    pub fn resize_(&mut self, shape: &[usize], value: f32) {
        self.shape = shape.to_vec();
        self.data.resize(shape.iter().product(), value);
    }

    pub fn resize_with_(&mut self, shape: &[usize], f: impl FnMut() -> f32) {
        self.shape = shape.to_vec();
        self.data.resize_with(shape.iter().product(), f);
    }

    pub fn transpose_(&mut self, dim0: usize, dim1: usize) {
        assert!(
            self.shape.len() >= 2,
            "cannot transpose tensor with less than 2 dimensions"
        );
        let n = self.size(dim0);
        let m = self.size(dim1);
        if n == m {
            for i in 0..n {
                for j in i + 1..m {
                    self.data.swap(i * m + j, j * n + i);
                }
            }
        } else {
            // TODO: slow, maybe use transposed strides instead
            let data = self.data.clone();
            self.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = data[(i % n) * m + i / n])
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
        if self.size(dim) == 1 {
            self.shape.remove(dim);
        }
    }

    pub fn squeeze_at_(&mut self, dims: &[usize]) {
        self.shape = self
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| {
                if x == 1 && dims.contains(&i) {
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

    pub fn cat_(&mut self, other: &Tensor, dim: usize) {
        // TODO: Only works for dim=0
        assert_eq!(
            self.size(dim),
            other.size(dim),
            "cannot concatenate tensors with different dimensions"
        );
        self.data.extend_from_slice(&other[..]);
        self.shape[dim] += other.size(dim);
    }

    pub fn broadcast_(&mut self, other_shape: &[usize]) {
        // TODO: Fix this and use it for matmul/add/sub/div/mul/dot
        if self.shape() == other_shape {
            return;
        }
        let (mut self_shape, other_shape) = match self.dim().cmp(&other_shape.len()) {
            std::cmp::Ordering::Less => {
                let mut ones = vec![1usize; self.dim() - other_shape.len()];
                ones.extend_from_slice(other_shape);
                (ones, self.shape.as_slice())
            }
            std::cmp::Ordering::Greater => {
                let mut ones = vec![1usize; other_shape.len() - self.dim()];
                ones.extend_from_slice(&self.shape);
                (ones, other_shape)
            }
            _ => (self.shape.to_vec(), other_shape),
        };

        for (self_dim, other_dim) in self_shape.iter_mut().zip(other_shape.iter()).rev() {
            match (*self_dim, *other_dim) {
                (a, b) if a == b => {}
                (_, 1) => {}
                (1, _) => {
                    *self_dim = *other_dim;
                }
                _ => panic!("incompatible tensor shapes for broadcasting"),
            }
        }
        self.shape = self_shape;
    }

    pub fn repeat_(&mut self, repeats: &[usize]) {
        // TODO: Only works for dim=0
        assert!(
            repeats.len() >= self.dim(),
            "repeat dimensions can not be smaller than tensor dimensions"
        );
        self.data = self[..].repeat(repeats.iter().product());
        if repeats.len() > self.dim() {
            let mut ones = vec![1usize; repeats.len() - self.dim()];
            ones.extend_from_slice(self.shape.as_slice());
            self.shape = ones;
        }
        self.shape
            .iter_mut()
            .zip(repeats.iter())
            .for_each(|(x, y)| *x *= y)
    }

    pub fn apply_(&mut self, mut f: impl FnMut(f32) -> f32) {
        self.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn softmax_(&mut self) {
        let max = self.max();
        self.apply_(|x| (x - max).exp());
        *self /= self.sum();
    }

    pub fn shuffle_(&mut self, rng: &mut Random) {
        self[..].shuffle(rng);
    }

    pub fn shuffle_at_(&mut self, dim: usize, rng: &mut Random) {
        let self_ptr = &mut self[..] as *mut [f32];
        let size: usize = if dim + 1 < self.dim() {
            self.shape[dim + 1..].iter().product()
        } else {
            1
        };
        for i in 0..self.size(dim) {
            let j = rng.gen_range_u32(i as u32..self.size(dim) as u32) as usize;
            let a = &mut unsafe { &mut *self_ptr }[i * size..][..size];
            let b = &mut unsafe { &mut *self_ptr }[j * size..][..size];
            if i != j {
                a.swap_with_slice(b);
            }
        }
    }

    pub fn fill_(&mut self, value: f32) {
        self[..].fill(value);
    }

    pub fn mul_(&mut self, other: &Tensor) {
        assert_eq!(
            self.shape.as_slice(),
            other.shape.as_slice(),
            "cannot multiply tensors with different shapes"
        );
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a *= b);
    }

    pub fn div_(&mut self, other: &Tensor) {
        assert_eq!(
            self.shape.as_slice(),
            other.shape.as_slice(),
            "cannot divide tensors with different shapes"
        );
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a /= b);
    }

    pub fn add_(&mut self, other: &Tensor) {
        assert_eq!(
            self.shape.as_slice(),
            other.shape.as_slice(),
            "cannot add tensors with different shapes"
        );
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b);
    }

    pub fn sub_(&mut self, other: &Tensor) {
        assert_eq!(
            self.shape.as_slice(),
            other.shape.as_slice(),
            "cannot subtract tensors with different shapes"
        );
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a -= b);
    }

    pub fn mul_scalar_(&mut self, scalar: f32) {
        self.iter_mut().for_each(|x| *x *= scalar);
    }

    pub fn div_scalar_(&mut self, scalar: f32) {
        self.iter_mut().for_each(|x| *x /= scalar);
    }

    pub fn add_scalar_(&mut self, scalar: f32) {
        self.iter_mut().for_each(|x| *x += scalar);
    }

    pub fn sub_scalar_(&mut self, scalar: f32) {
        self.iter_mut().for_each(|x| *x -= scalar);
    }

    pub fn neg_(&mut self) {
        self.iter_mut().for_each(|x| *x = -*x);
    }
}

impl IntoIterator for Tensor {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl AsRef<[f32]> for Tensor {
    fn as_ref(&self) -> &[f32] {
        &self[..]
    }
}

impl<I: std::slice::SliceIndex<[f32]>> ops::Index<I> for Tensor {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.data[index]
    }
}

impl<I: std::slice::SliceIndex<[f32]>> ops::IndexMut<I> for Tensor {
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

impl<const N: usize> From<&[[f32; N]]> for Tensor {
    fn from(data: &[[f32; N]]) -> Self {
        Self::matrix(data)
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

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        self * &other
    }
}

impl ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        self / &other
    }
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        self + &other
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        self - &other
    }
}

impl ops::Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(mut self, other: &Tensor) -> Self::Output {
        self.mul_(other);
        self
    }
}

impl ops::Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(mut self, other: &Tensor) -> Self::Output {
        self.div_(other);
        self
    }
}

impl ops::Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(mut self, other: &Tensor) -> Self::Output {
        self.add_(other);
        self
    }
}

impl ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(mut self, other: &Tensor) -> Self::Output {
        self.sub_(other);
        self
    }
}

impl ops::Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(mut self, scalar: f32) -> Self::Output {
        self.mul_scalar_(scalar);
        self
    }
}

impl ops::Div<f32> for Tensor {
    type Output = Tensor;

    fn div(mut self, scalar: f32) -> Self::Output {
        self.div_scalar_(scalar);
        self
    }
}

impl ops::Add<f32> for Tensor {
    type Output = Tensor;

    fn add(mut self, scalar: f32) -> Self::Output {
        self.add_scalar_(scalar);
        self
    }
}

impl ops::Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(mut self, scalar: f32) -> Self::Output {
        self.sub_scalar_(scalar);
        self
    }
}

impl ops::Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: Tensor) -> Self::Output {
        tensor * self
    }
}

impl ops::Div<Tensor> for f32 {
    type Output = Tensor;

    fn div(self, mut tensor: Tensor) -> Self::Output {
        tensor.iter_mut().for_each(|x| *x = self / *x);
        tensor
    }
}

impl ops::Add<Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: Tensor) -> Self::Output {
        tensor + self
    }
}

impl ops::Sub<Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, mut tensor: Tensor) -> Self::Output {
        tensor.iter_mut().for_each(|x| *x = self - *x);
        tensor
    }
}

impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(mut self) -> Self::Output {
        self.neg_();
        self
    }
}

impl ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        let mut result = self.clone();
        result.mul_(other);
        result
    }
}

impl ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Self::Output {
        let mut result = self.clone();
        result.div_(other);
        result
    }
}

impl ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Self::Output {
        let mut result = self.clone();
        result.add_(other);
        result
    }
}

impl ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        let mut result = self.clone();
        result.sub_(other);
        result
    }
}

impl ops::Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Self::Output {
        let mut result = self.clone();
        result.mul_scalar_(scalar);
        result
    }
}

impl ops::Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Self::Output {
        let mut result = self.clone();
        result.div_scalar_(scalar);
        result
    }
}

impl ops::Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Self::Output {
        let mut result = self.clone();
        result.add_scalar_(scalar);
        result
    }
}

impl ops::Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f32) -> Self::Output {
        let mut result = self.clone();
        result.sub_scalar_(scalar);
        result
    }
}

impl ops::Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: &Tensor) -> Self::Output {
        tensor * self
    }
}

impl ops::Div<&Tensor> for f32 {
    type Output = Tensor;

    fn div(self, tensor: &Tensor) -> Self::Output {
        let mut result = tensor.clone();
        result.iter_mut().for_each(|x| *x = self / *x);
        result
    }
}

impl ops::Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: &Tensor) -> Self::Output {
        tensor + self
    }
}

impl ops::Sub<&Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, tensor: &Tensor) -> Self::Output {
        let mut result = tensor.clone();
        result.iter_mut().for_each(|x| *x = self - *x);
        result
    }
}

impl ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let mut result = self.clone();
        result.iter_mut().for_each(|x| *x = -*x);
        result
    }
}

impl ops::AddAssign<f32> for Tensor {
    fn add_assign(&mut self, scalar: f32) {
        self.add_scalar_(scalar);
    }
}

impl ops::SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, scalar: f32) {
        self.sub_scalar_(scalar);
    }
}

impl ops::MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, scalar: f32) {
        self.mul_scalar_(scalar);
    }
}

impl ops::DivAssign<f32> for Tensor {
    fn div_assign(&mut self, scalar: f32) {
        self.div_scalar_(scalar);
    }
}

impl ops::AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        self.add_(other);
    }
}

impl ops::SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        self.sub_(other);
    }
}

impl ops::MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        self.mul_(other);
    }
}

impl ops::DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        self.div_(other);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl Eq for Tensor {}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn vals(slf: &Tensor, depth: usize, i: usize, f: &fmt::Formatter<'_>) -> String {
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
                    format!("{:.4}", d[i + j * p])
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand;

    macro_rules! assert_approx_eq {
        ($a: expr, $b: expr, $($message:tt)+) => {
            let a: Vec<f32> = $a.iter_mut().map(|x| (*x * 1e6f32).round() / 1e6).collect();
            let b: Vec<f32> = $b.iter_mut().map(|x| (*x * 1e6f32).round() / 1e6).collect();
            assert_eq!(a, b, $message);
        };
        ($a: expr, $b: expr) => {
            let a: Vec<f32> = $a.iter_mut().map(|x| (*x * 1e6f32).round() / 1e6).collect();
            let b: Vec<f32> = $b.iter_mut().map(|x| (*x * 1e6f32).round() / 1e6).collect();
            assert_eq!(a, b);
        };
    }

    #[test]
    fn init_macro() {
        assert_eq!(tensor![3.0][..], [3.0]);
        assert_eq!(tensor![5.0; 3][..], [5.0, 5.0, 5.0]);
        assert_eq!(tensor![2.0; 3, 2][..], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        assert_eq!(tensor![2.0; 3, 2].shape(), [3, 2]);
        assert_eq!(tensor![0.0, 1.0, 2.0][..], [0.0, 1.0, 2.0]);
        assert_eq!(
            tensor![[3.0, 2.0], [4.0, 5.0], [7.0, 1.0]][..],
            [3.0, 2.0, 4.0, 5.0, 7.0, 1.0]
        );
        assert_eq!(tensor![[[3.0, 4.0], [2.0, 1.0]]][..], [3.0, 4.0, 2.0, 1.0]);
        assert_eq!(tensor![[[3.0, 4.0], [2.0, 1.0]]].shape(), [1, 2, 2]);
        assert_eq!(
            tensor![[[[1.0], [4.0]], [[3.0], [2.0]]]].shape(),
            [1, 2, 2, 1]
        );
        assert_eq!(tensor![[3.0, 2.0], [4.0, 5.0], [7.0, 1.0]].shape(), [3, 2]);
    }

    #[test]
    fn equality() {
        assert_eq!(tensor![4.0, 6.0, 3.0], tensor![4.0, 6.0, 3.0]);
        assert_ne!(tensor![0.0, 1.0], tensor![0.0, 1.4]);
    }

    #[test]
    fn init() {
        assert_eq!(Tensor::new(&[3.0, 2.0], &[2])[..], [3.0, 2.0]);
        assert_eq!(Tensor::array(&[3.0, 2.0])[..], [3.0, 2.0]);
        assert_eq!(
            Tensor::matrix(&[[3.0, 2.0], [4.0, 5.0]])[..],
            [3.0, 2.0, 4.0, 5.0]
        );
        assert_eq!(
            Tensor::matrix(&[[3.0, 2.0], [7.0, 6.0], [4.0, 5.0]]).shape(),
            [3, 2]
        );

        assert_eq!(Tensor::zeros(&[3, 2])[..], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(Tensor::ones(&[3, 2])[..], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let seed = 518012623;
        let mut r = rand::Random::from_seed(seed);
        let rands: Vec<f32> = (0..6).into_iter().map(|_| r.gen()).collect();
        rand::set_seed(seed);
        assert_eq!(&Tensor::rand(&[3, 2])[..], &rands);

        let seed = r.seed();
        let rands: Vec<f32> = (0..6).into_iter().map(|_| r.norm()).collect();
        rand::set_seed(seed);
        assert_eq!(&Tensor::randn(&[2, 3])[..], &rands);

        assert_approx_eq!(Tensor::arange(0.0, 3.0, 1.0)[..], [0.0, 1.0, 2.0]);
        assert_approx_eq!(Tensor::arange(0.1, 2.8, 1.0)[..], [0.1, 1.1, 2.1]);
        assert_approx_eq!(
            Tensor::arange(-1.3, 1.0, 0.6),
            tensor![-1.3, -0.7, -0.1, 0.5]
        );
        assert_approx_eq!(Tensor::linspace(0.0, 3.0, 3)[..], [0.0, 1.5, 3.0]);
        assert_approx_eq!(Tensor::linspace(0.1, 2.8, 4)[..], [0.1, 1.0, 1.9, 2.8]);
        assert_approx_eq!(
            Tensor::linspace(-1.4, 1.0, 7),
            tensor![-1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
        );

        assert_eq!(
            Tensor::eye(3, 4),
            tensor![[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        );
        assert_eq!(
            Tensor::identity(3),
            tensor![[1, 0, 0], [0, 1, 0], [0, 0, 1,]]
        );

        assert_eq!(Tensor::splat(&[1, 2], 2.0)[..], [2.0, 2.0]);
        let mut count = 0.0;
        assert_eq!(
            Tensor::splat_with(&[3, 2], || {
                count += 1.0;
                count
            })[..],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn memory() {
        let a = Tensor::array(&[3.0, 2.0]);
        let b = a.deep_clone();
        assert!(!std::ptr::addr_eq(
            std::ptr::addr_of!(a.data),
            std::ptr::addr_of!(b.data)
        ));
    }

    #[test]
    fn metadata() {
        assert_eq!(tensor![3.0, 2.0].shape(), &[2]);
        assert_eq!(tensor![[3.0, 2.0, 0.0], [4.0, 5.0, 1.0]].shape(), &[2, 3]);
        assert_eq!(tensor![[3.0], [2.0]].size(0), 2);
        assert_eq!(tensor![[3.0], [2.0]].size(1), 1);
        assert_eq!(tensor![[[3.0]], [[2.0]]].dim(), 3);
        assert_eq!(tensor![[3.0, 2.0, 0.0], [4.0, 5.0, 1.0]].numel(), 6);
    }

    #[test]
    fn slicing() {
        assert_eq!(
            tensor![[3.0, 2.0, 0.0], [4.0, 5.0, 1.0]].idx(1),
            tensor![4.0, 5.0, 1.0]
        );
        assert_eq!(
            tensor![[3.0, 2.0, 0.0], [4.0, 5.0, 1.0]].idx(1).idx(2),
            tensor![1.0]
        );
        assert_eq!(tensor![[3.0, 2.0, 0.0], [4.0, 5.0, 1.0]].idx(1).dim(), 1);
        assert_eq!(tensor![[3.0, 2.0, 0.0], [4.0, 5.0, 1.0]].idx(1).numel(), 3);
        assert_eq!(
            tensor![[3.0, 2.0, 0.0], [4.0, 5.0, 1.0]].idx(1).shape(),
            &[3]
        );

        assert_eq!(
            tensor![[3.0, 2.0], [0.0, 1.0], [4.0, 5.0]].slice(1..3),
            tensor![[0.0, 1.0], [4.0, 5.0]]
        );
        assert_eq!(
            tensor![[3.0, 2.0], [0.0, 1.0], [4.0, 5.0]]
                .slice(1..2)
                .shape(),
            [1, 2]
        );

        assert_eq!(
            tensor![[3.0, 2.0], [0.0, 1.0], [4.0, 5.0]].at(&[2, 1]),
            tensor![5.0]
        );

        assert_eq!(
            tensor![[3.0, 2.0], [0.0, 1.0], [4.0, 5.0]].get(&[2, 1]),
            5.0
        );
        assert_approx_eq!(tensor![3.0, 2.0].iter().collect::<Vec<_>>(), [3.0, 2.0]);
        assert_approx_eq!(
            tensor![3.0, 2.0].into_iter().collect::<Vec<_>>(),
            [3.0, 2.0]
        );
    }

    #[test]
    fn transpose() {
        assert_eq!(
            tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].t(),
            tensor![[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]
        );
        assert_eq!(
            tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].t(),
            tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        );
        assert_eq!(
            tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]].t(),
            tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        );
    }

    #[test]
    fn matmul() {
        let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = tensor![[1.0, 2.0], [3.0, 4.0]];
        assert_approx_eq!(
            a.matmul(&b),
            tensor![[7.0, 10.0], [15.0, 22.0], [23.0, 34.0]]
        );

        // TODO: implement broadcasting first
        // let a = tensor![
        //     [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        //     [[2.0, 2.0], [6.0, 4.0], [10.0, 6.0]]
        // ];
        // let b = tensor![[1.0, 2.0], [3.0, 4.0]];
        // assert_approx_eq!(
        //     a.matmul(&b),
        //     tensor![[8.0, 12.0], [18.0, 28.0], [28.0, 44.0]]
        // );
    }
}
