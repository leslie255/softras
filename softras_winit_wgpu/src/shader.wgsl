@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var sampler_: sampler;

const vertices = array<vec2<f32>, 6>(
    vec2<f32>(0., 0.),
    vec2<f32>(1., 0.),
    vec2<f32>(1., 1.),
    vec2<f32>(0., 0.),
    vec2<f32>(1., 1.),
    vec2<f32>(0., 1.),
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let position = vertices[index];
    var result: VertexOutput;
    result.position = vec4(position.x * 2. - 1., position.y * 2. - 1., 0., 1.);
    result.uv = vec2(position.x, 1. - position.y);
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(texture, sampler_, vertex.uv);
}
