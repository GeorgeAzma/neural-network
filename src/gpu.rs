use crate::tensor::Slice;
use crate::tensor::Tensor;
use lazy_static::lazy_static;

lazy_static! {
    static ref INSTANCE: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
}

lazy_static! {
    static ref ADAPTER: wgpu::Adapter =
        futures::executor::block_on(INSTANCE.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .unwrap();
}

lazy_static! {
    static ref DEVICE_AND_QUEUE: (wgpu::Device, wgpu::Queue) = {
        let (device, queue) = futures::executor::block_on(ADAPTER.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                required_limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        ))
        .unwrap();
        (device, queue)
    };
}

pub fn device() -> &'static wgpu::Device {
    &DEVICE_AND_QUEUE.0
}

pub fn queue() -> &'static wgpu::Queue {
    &DEVICE_AND_QUEUE.1
}

lazy_static! {
    static ref SHADER_MODULE: wgpu::ShaderModule =
        device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(
                std::fs::read_to_string("data/shaders/matmul.wgsl").unwrap_or_else(|err| {
                    panic!(
                        "Failed to open shader file data/shaders/matmul.wgsl: {}",
                        err
                    )
                }),
            )),
        });
}

lazy_static! {
    static ref BIND_GROUP_LAYOUT: wgpu::BindGroupLayout =
        device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
        });
}

lazy_static! {
    static ref PIPELINE_LAYOUT: wgpu::PipelineLayout =
        device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        });
}

lazy_static! {
    static ref PIPELINE: wgpu::ComputePipeline =
        device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&PIPELINE_LAYOUT),
            module: &SHADER_MODULE,
            entry_point: "main",
        });
}

pub fn matmul<T1: Slice<f32>, S1: Slice<usize>, T2: Slice<f32>, S2: Slice<usize>>(
    a: &Tensor<T1, S1>,
    b: &Tensor<T2, S2>,
) -> Tensor {
    use wgpu::util::DeviceExt;
    let dim0 = if a.dim() == 1 { 1 } else { a.size(a.dim() - 2) };
    let dim1 = a.size(a.dim() - 1);
    let dim1_b = b.size(b.dim() - 1);
    let mut input = vec![
        f32::from_bits(dim0 as u32),
        f32::from_bits(dim1 as u32),
        f32::from_bits(dim1_b as u32),
    ];
    input.extend_from_slice(a.data());
    input.extend_from_slice(b.data());
    let input_buffer = device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (dim0 * dim1_b * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &BIND_GROUP_LAYOUT,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(input_buffer.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(output_buffer.as_entire_buffer_binding()),
            },
        ],
    });

    let mut encoder = device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&PIPELINE);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((dim0 as u32 + 7) / 8, (dim1_b as u32 + 7) / 8, 1);
    }
    // let start = std::time::Instant::now();
    queue().submit(std::iter::once(encoder.finish()));
    output_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});
    device().poll(wgpu::MaintainBase::Wait);
    let output_range = output_buffer.slice(..).get_mapped_range();
    let output_data = unsafe {
        std::slice::from_raw_parts(
            output_range.as_ptr() as *const f32,
            output_range.len() / std::mem::size_of::<f32>(),
        )
    };
    // println!("{} ms", start.elapsed().as_secs_f32() * 1000.0);
    Tensor::new(output_data, &[dim0, dim1_b])
}
