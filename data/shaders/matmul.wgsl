struct Input {
    dim0: u32,
    dim1: u32,
    dim1_b: u32,
    data: array<f32>,
}

@group(0) @binding(0) var<storage, read> input: Input;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= input.dim0 || gid.y >= input.dim1_b {
        return;
    }
    output[gid.x * input.dim0 + gid.y] = 0.0;
    for (var y = 0u; y < input.dim1; y++) {
        let p = vec3u(gid.x, y, gid.y);
        output[p.x * input.dim0 + p.z] += input.data[p.x * input.dim1 + p.y] * 
        input.data[input.dim0 * input.dim1 + p.y * input.dim1_b + p.z];
    }
}