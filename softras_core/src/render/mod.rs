use std::{array, f32};

use bytemuck::{Pod, Zeroable};
use glam::*;

mod color;
mod shader;

pub use color::*;
use obj::Obj;
pub use shader::*;

use crate::*;

#[derive(Debug, Default, Clone, Copy)]
pub struct RenderOptions {
    pub cull_face: CullFaceMode,
    pub disable_chunking: bool,
}

/// Option for culling triangles according to their winding direction.
/// 3D models are conventionally clockwise-winded, therefore the default value of this `enum` is
/// `CullCCW`. #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CullFaceMode {
    /// Cull counter-clockwise triangles (the default).
    #[default]
    CullCounterClockwise,
    /// Cull clockwise triangles (the default).
    CullClockwise,
    /// No face culling based on winding directions.
    NoCulling,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct GbufferPixel {
    pub albedo: Rgb,
    pub specular: f32,
    pub position: Vec3,
    pub normal: Vec3,
}

#[derive(derive_more::Debug, Clone)]
pub struct Canvas {
    width: u32,
    height: u32,
    #[debug(skip)]
    frame_buffer: Vec<RgbaU8>,
    #[debug(skip)]
    depth_buffer: Vec<f32>,
    #[debug(skip)]
    gbuffer: Vec<GbufferPixel>,
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

impl Canvas {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            frame_buffer: Vec::new(),
            depth_buffer: Vec::new(),
            gbuffer: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[allow(dead_code)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[allow(dead_code)]
    pub fn frame_buffer(&self) -> &[RgbaU8] {
        &self.frame_buffer
    }

    #[allow(dead_code)]
    pub fn gbuffer(&self) -> &[GbufferPixel] {
        &self.gbuffer
    }

    pub fn clear(&mut self) {
        self.frame_buffer.fill(RgbaU8::zeroed());
        self.depth_buffer.fill(1.0f32);
        self.gbuffer.fill(GbufferPixel::zeroed());
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.frame_buffer.resize(self.n_pixels(), RgbaU8::zeroed());
        self.depth_buffer.resize(self.n_pixels(), 0.0f32);
        self.gbuffer.resize(
            self.n_pixels(),
            GbufferPixel {
                albedo: Rgb::zeroed(),
                specular: 0.0f32,
                position: Vec3::zeroed(),
                normal: Vec3::zeroed(),
            },
        );
    }

    pub fn n_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

pub fn postprocess<P: Postprocessor + ?Sized>(canvas: &mut Canvas, postprocessor: &P) {
    debug_assert!(canvas.frame_buffer.len() == canvas.n_pixels());
    debug_assert!(canvas.gbuffer.len() == canvas.n_pixels());
    debug_assert!(canvas.depth_buffer.len() == canvas.n_pixels());

    for x_pixel in 0..canvas.width {
        for y_pixel in 0..canvas.height {
            let i_pixel = y_pixel as usize * canvas.width as usize + x_pixel as usize;
            let result = unsafe { canvas.frame_buffer.get_unchecked_mut(i_pixel) };
            let gbuffer_pixel = unsafe { *canvas.gbuffer.get_unchecked(i_pixel) };
            let depth = unsafe { *canvas.depth_buffer.get_unchecked(i_pixel) };
            let material_input = PostprocessInput {
                albedo: gbuffer_pixel.albedo,
                specular: gbuffer_pixel.specular,
                position: gbuffer_pixel.position,
                normal: gbuffer_pixel.normal,
                depth,
            };
            *result = postprocessor.postprocess(material_input).into();
        }
    }
}

/// Draws a triangle to a canvas.
///
/// * `canvas`           - the canvas to draw on
/// * `options`          - the render options
/// * `mvp`              - the model-view-projection matrix, equals to projection * view * model
/// * `model`            - the model matrix
/// * `normal_transform` - the matrix for transforming normals, equals to
///   `transpose(inverse(model))`
/// * `vertices`         - the vertices of the triangle
pub fn draw_triangle<S: Material + ?Sized>(
    canvas: &mut Canvas,
    options: RenderOptions,
    mvp: Mat4,
    model: Mat4,
    normal_transform: Mat4,
    material: &S,
    vertices: [Vertex; 3],
) {
    let vertices_clip: [VertexClip; 3] = vertices.map(|vertex| {
        let position_clip = mvp * vertex.position.extend(1.);
        VertexClip {
            position_world: model.transform_point3(vertex.position),
            position: position_clip.xyz(),
            w: position_clip.w,
            uv: vertex.uv,
            normal: normal_transform.transform_vector3(vertex.normal),
        }
    });

    #[rustfmt::skip]
    match vertices_clip.map(|vertex| vertex.position.z >= 0.) {
        // All points are in front of near plane, no near plane clipping needed.
        [true, true, true] => after_near_clipping(canvas, options, vertices_clip, material),

        // All points are behind near plane.
        [false, false, false] => (),

        // Clip Case 1: One point is behind the near plane, the other two points are in front.
        [false, true, true] => clip_case_1::<_, 0>(canvas, options, vertices_clip, material),
        [true, false, true] => clip_case_1::<_, 1>(canvas, options, vertices_clip, material),
        [true, true, false] => clip_case_1::<_, 2>(canvas, options, vertices_clip, material),

        // Clip Case 2: Two points are behind the near plane, the other point is in front.
        [true, false, false] => clip_case_2::<_, 0>(canvas, options, vertices_clip, material),
        [false, true, false] => clip_case_2::<_, 1>(canvas, options, vertices_clip, material),
        [false, false, true] => clip_case_2::<_, 2>(canvas, options, vertices_clip, material),
    };
}

// Clip Case 1: One point is behind the near plane, the other two points are in front.
#[cold]
fn clip_case_1<S: Material + ?Sized, const I_BACK: usize>(
    canvas: &mut Canvas,
    options: RenderOptions,
    vertices: [VertexClip; 3],
    material: &S,
) {
    const { assert!(I_BACK < 3) };
    let [v0, v1, v2] = vertices;
    let [v0_vb, v1_vb, v2_vb] = vertices.map(|p| near_plane_intersection(p, vertices[I_BACK]));
    let result_vertices: [[VertexClip; 3]; 2] = match I_BACK {
        0 => [[v1_vb, v1, v2], [v1_vb, v2, v2_vb]],
        1 => [[v2_vb, v2, v0], [v2_vb, v0, v0_vb]],
        2 => [[v0_vb, v0, v1], [v0_vb, v1, v1_vb]],
        _ => unreachable!(),
    };
    after_near_clipping(canvas, options, result_vertices[0], material);
    after_near_clipping(canvas, options, result_vertices[1], material);
}

// Clip Case 2: Two points are behind the near plane, the other point is in front.
#[cold]
#[inline(never)]
fn clip_case_2<S: Material + ?Sized, const I_FRONT: usize>(
    canvas: &mut Canvas,
    options: RenderOptions,
    vertices: [VertexClip; 3],
    material: &S,
) {
    const { assert!(I_FRONT < 3) };
    let [v0, v1, v2] = vertices;
    let [v0_vf, v1_vf, v2_vf] = vertices.map(|p| near_plane_intersection(vertices[I_FRONT], p));
    let positions: [VertexClip; 3] = match I_FRONT {
        0 => [v0, v1_vf, v2_vf],
        1 => [v0_vf, v1, v2_vf],
        2 => [v0_vf, v1_vf, v2],
        _ => unreachable!(),
    };
    after_near_clipping(canvas, options, positions, material);
}

fn near_plane_intersection(v_front: VertexClip, v_back: VertexClip) -> VertexClip {
    let t = v_front.position.z / (v_front.position.z - v_back.position.z);
    VertexClip {
        position_world: lerp_vec3(v_front.position_world, v_back.position_world, t),
        position: lerp_vec3(v_front.position, v_back.position, t),
        w: lerp(v_front.w, v_back.w, t),
        uv: lerp_vec2(v_front.uv, v_back.uv, t),
        normal: lerp_vec3(v_front.normal, v_back.normal, t),
    }
}

/// Handle the rendering from after clipping (but before perspective division).
fn after_near_clipping<S: Material + ?Sized>(
    canvas: &mut Canvas,
    options: RenderOptions,
    vertices_clip: [VertexClip; 3],
    material: &S,
) {
    let positions_div_w = vertices_clip.map(|vertex| vertex.position.xyz() / vertex.w);
    let positions_world_div_w = vertices_clip.map(|vertex| vertex.position_world / vertex.w);
    let positions_ndc = positions_div_w.map(|p| p.xy());

    match options.cull_face {
        CullFaceMode::CullCounterClockwise if !is_clockwise_winding(positions_ndc) => return,
        CullFaceMode::CullClockwise if is_clockwise_winding(positions_ndc) => return,
        _ => (),
    }

    let vertices_ndc: [VertexNdc; 3] = array::from_fn(|i| {
        let vertex_clip = vertices_clip[i];
        let w = vertex_clip.w;
        VertexNdc {
            w_inv: 1. / w,
            position_div_w: positions_div_w[i],
            position_world_div_w: positions_world_div_w[i],
            uv_div_w: vertex_clip.uv / w,
            normal: w * vertex_clip.normal,
        }
    });

    // Bounding box of pixels that we need to sample.
    let bounds @ [x_min, x_max, y_min, y_max]: [u32; 4] = {
        let [p0, p1, p2] = positions_ndc;
        let min_ndc = p0.min(p1).min(p2);
        let max_ndc = p0.max(p1).max(p2);
        let width = canvas.width;
        let height = canvas.height;
        // Y min/max need to be inverted because NDC is down-to-up whereas pixel coordinate is up-
        // to-down.
        [
            (ndc_to_pixel_x(min_ndc.x, width).floor().max(0.) as u32).min(width - 1),
            (ndc_to_pixel_x(max_ndc.x, width).ceil().max(0.) as u32).min(width - 1),
            (ndc_to_pixel_y(max_ndc.y, height).floor().max(0.) as u32).min(height - 1),
            (ndc_to_pixel_y(min_ndc.y, height).ceil().max(0.) as u32).min(height - 1),
        ]
    };
    let bounds_area = (x_max - x_min) * (y_max - y_min);
    let chunk_size = 32u32;
    if !options.disable_chunking && chunk_size.pow(2) < bounds_area {
        // Chunking case.
        // (Chunking is basically only an optimization measure for dealing with triangles that
        // occupy big screen spaces, so if you are checking out the code base just to understand
        // the workings of rasterization process, simply assume the other no chunking path).
        rasterize_chunked(canvas, options, vertices_ndc, material, bounds, chunk_size);
    } else {
        // No chunking case.
        rasterize(canvas, options, vertices_ndc, material, bounds);
    }
}

fn rasterize<M: Material + ?Sized>(
    canvas: &mut Canvas,
    _options: RenderOptions,
    vertices_ndc: [VertexNdc; 3],
    material: &M,
    [x_min, x_max, y_min, y_max]: [u32; 4],
) {
    debug_assert!(canvas.gbuffer.len() == canvas.n_pixels());
    debug_assert!(canvas.depth_buffer.len() == canvas.n_pixels());

    let positions_ndc: [Vec2; 3] = vertices_ndc.map(|p| p.position_div_w.xy());
    let positions_world: [Vec3; 3] = vertices_ndc.map(|p| p.position_world_div_w);
    let depths: [f32; 3] = vertices_ndc.map(|p| p.position_div_w.z);
    let uvs_div_w: [Vec2; 3] = vertices_ndc.map(|p| p.uv_div_w);
    let normals: [Vec3; 3] = vertices_ndc.map(|p| p.normal);
    let w_invs: [f32; 3] = vertices_ndc.map(|p| p.w_inv);

    for x_pixel in x_min..=x_max {
        for y_pixel in y_min..=y_max {
            let i_pixel = y_pixel as usize * canvas.width as usize + x_pixel as usize;
            let gbuffer_sample = unsafe { canvas.gbuffer.get_unchecked_mut(i_pixel) };
            let depth_sample = unsafe { canvas.depth_buffer.get_unchecked_mut(i_pixel) };

            let p_ndc = vec2(
                pixel_to_ndc_x(x_pixel, canvas.width),
                pixel_to_ndc_y(y_pixel, canvas.height),
            );
            let Some(weights) = barycentric_weights(positions_ndc, p_ndc) else {
                continue;
            };
            let depth = terp(weights, depths);
            if *depth_sample <= depth {
                continue;
            }
            let w = 1. / terp(weights, w_invs);
            let uv = w * terp_vec2(weights, uvs_div_w);
            let normal = terp_vec3(weights, normals);
            let fragment_input = FragmentInput {
                position: w * vec3(p_ndc.x, p_ndc.y, depth),
                depth,
                uv,
                normal,
            };

            let fragment_output = material.fragment(fragment_input);
            *gbuffer_sample = GbufferPixel {
                albedo: fragment_output.albedo,
                specular: fragment_output.specular,
                position: w * terp_vec3(weights, positions_world),
                normal,
            };
            *depth_sample = depth;
        }
    }
}

fn rasterize_chunked<M: Material + ?Sized>(
    canvas: &mut Canvas,
    options: RenderOptions,
    vertices_ndc: [VertexNdc; 3],
    material: &M,
    [x_min, x_max, y_min, y_max]: [u32; 4],
    chunk_size: u32,
) {
    let positions_pixel = vertices_ndc.map(|p_ndc| {
        vec2(
            ndc_to_pixel_x(p_ndc.position_div_w.x, canvas.width),
            ndc_to_pixel_y(p_ndc.position_div_w.y, canvas.height),
        )
    });
    for y_min_ in step_range(y_min, (y_max + 1).next_multiple_of(chunk_size), chunk_size) {
        for x_min_ in step_range(x_min, (x_max + 1).next_multiple_of(chunk_size), chunk_size) {
            let x_max_ = (x_min_ + chunk_size).min(canvas.width - 1);
            let y_max_ = (y_min_ + chunk_size).min(canvas.height - 1);
            let bounds_chunk = [x_min_, x_max_, y_min_, y_max_];
            if rect_triangle_overlap(positions_pixel, bounds_chunk.map(|u| u as f32)) {
                rasterize(canvas, options, vertices_ndc, material, bounds_chunk);
            }
        }
    }
}

fn rect_triangle_overlap(
    triangle @ [p0, p1, p2]: [Vec2; 3],
    rect @ [x_min, x_max, y_min, y_max]: [f32; 4],
) -> bool {
    if point_inside_rect(p0, rect) || point_inside_rect(p1, rect) || point_inside_rect(p2, rect) {
        return true;
    }
    if point_in_triangle(triangle, vec2(x_min, y_min))
        || point_in_triangle(triangle, vec2(x_min, y_max))
        || point_in_triangle(triangle, vec2(x_max, y_min))
        || point_in_triangle(triangle, vec2(x_max, y_max))
    {
        return true;
    }
    line_rect_intersects(p0, p1, rect)
        || line_rect_intersects(p1, p2, rect)
        || line_rect_intersects(p2, p0, rect)
}

fn point_inside_rect(p: Vec2, [x_min, x_max, y_min, y_max]: [f32; 4]) -> bool {
    (x_min..=x_max).contains(&p.x) || (y_min..=y_max).contains(&p.y)
}

fn line_rect_intersects(p0: Vec2, p1: Vec2, [x_min, x_max, y_min, y_max]: [f32; 4]) -> bool {
    // Liang-Barsky algorithm.
    let x0 = p0.x;
    let y0 = p0.y;
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let p = [
        -dx, // p1
        dx,  // p2
        -dy, // p3
        dy,  // p4
    ];
    let q = [
        x0 - x_min, // q1
        x_max - x0, // q2
        y0 - y_min, // q3
        y_max - y0, // q4
    ];
    let mut u1 = 0.0f32;
    let mut u2 = 1.0f32;
    for i in 0..4 {
        if p[i].abs() < 0.001 {
            if q[i] < 0. {
                return false;
            }
        } else {
            let t = q[i] / p[i];
            if p[i] < 0. {
                u1 = u1.max(t);
            } else if p[i] > 0. {
                u2 = u2.min(t);
            }
        }
    }
    u1 <= u2 && u2 >= 0. && u1 <= 1.
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn lerp_vec2(a: Vec2, b: Vec2, t: f32) -> Vec2 {
    a + t * (b - a)
}

fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + t * (b - a)
}

fn point_in_triangle([a, b, c]: [Vec2; 3], p: Vec2) -> bool {
    barycentric_weights([a, b, c], p).is_some()
}

/// Triangular interpolation.
fn terp([w0, w1, w2]: [f32; 3], [x0, x1, x2]: [f32; 3]) -> f32 {
    w0 * x0 + w1 * x1 + w2 * x2
}

/// Triangular interpolation.
fn terp_vec2(weights: [f32; 3], [v0, v1, v2]: [Vec2; 3]) -> Vec2 {
    vec2(
        terp(weights, [v0.x, v1.x, v2.x]),
        terp(weights, [v0.y, v1.y, v2.y]),
    )
}

/// Triangular interpolation.
fn terp_vec3(weights: [f32; 3], [v0, v1, v2]: [Vec3; 3]) -> Vec3 {
    vec3(
        terp(weights, [v0.x, v1.x, v2.x]),
        terp(weights, [v0.y, v1.y, v2.y]),
        terp(weights, [v0.z, v1.z, v2.z]),
    )
}

fn ndc_to_pixel_x(x_ndc: f32, width: u32) -> f32 {
    let width_f = width as f32;
    0.5 * width_f * (x_ndc + 1.)
}

fn ndc_to_pixel_y(y_ndc: f32, height: u32) -> f32 {
    let height_f = height as f32;
    -0.5 * height_f * (y_ndc - 1.)
}

fn pixel_to_ndc_x(x_pixel: u32, width: u32) -> f32 {
    let x_pixel_f = x_pixel.min(width - 1) as f32;
    let width_f = width as f32;
    2. * x_pixel_f / width_f - 1.
}

fn pixel_to_ndc_y(y_pixel: u32, height: u32) -> f32 {
    let y_pixel_f = y_pixel.min(height - 1) as f32;
    let height_f = height as f32;
    -2. * y_pixel_f / height_f + 1.
}

fn is_clockwise_winding([a, b, c]: [Vec2; 3]) -> bool {
    fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        0.5 * (c - a).dot((b - a).perp())
    }
    signed_area(a, b, c) >= 0.
}

/// If point `p` is inside the triangle formed by XP components of points `a`, `b`, and `c`,
/// returns the weights of `a`, `b`, and `c` for triangular-interpolation.
fn barycentric_weights([a, b, c]: [Vec2; 3], p: Vec2) -> Option<[f32; 3]> {
    let [a, b, c] = [a, b, c];
    fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        0.5 * (c - a).dot((b - a).perp())
    }
    let area_bcp = signed_area(b, c, p);
    let area_cap = signed_area(c, a, p);
    let area_abp = signed_area(a, b, p);
    if (area_bcp > 0.) == (area_cap > 0.) && (area_cap > 0.) == (area_abp > 0.) {
        let area_total = area_bcp + area_cap + area_abp;
        Some([
            (1. / area_total) * area_bcp,
            (1. / area_total) * area_cap,
            (1. / area_total) * area_abp,
        ])
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub pitch_degrees: f32,
    pub yaw_degrees: f32,
    pub fov_y_degrees: f32,
}

impl Camera {
    pub fn direction(self) -> Vec3 {
        let pitch = self.pitch_degrees.to_radians();
        let yaw = self.yaw_degrees.to_radians();
        vec3(
            f32::sin(yaw) * f32::cos(pitch),
            f32::sin(pitch),
            -f32::cos(yaw) * f32::cos(pitch),
        )
    }

    pub fn move_(&mut self, delta: Vec3) {
        let direction = self.direction();
        let forward = direction.with_y(0.).normalize();
        let up = vec3(0., 1., 0.);
        let right = forward.cross(up).normalize();
        let forward_scaled = delta.z * forward;
        let right_scaled = delta.x * right;
        let up_scaled = delta.y * up;
        self.position += forward_scaled + right_scaled + up_scaled;
    }

    pub fn view_matrix(self) -> Mat4 {
        Mat4::look_to_rh(
            self.position,    // eye
            self.direction(), // dir
            vec3(0., 1., 0.), // up
        )
    }

    pub fn projection_matrix(self, frame_width: f32, frame_height: f32) -> Mat4 {
        let aspect = frame_width / frame_height;
        let fovy = self.fov_y_degrees.to_radians();
        let near = 0.1;

        Mat4::perspective_infinite_rh(fovy, aspect, near)
    }
}

/// A vertex in NDC space.
#[derive(Debug, Clone, Copy, PartialEq)]
struct VertexNdc {
    position_world_div_w: Vec3,
    position_div_w: Vec3,
    w_inv: f32,
    uv_div_w: Vec2,
    normal: Vec3,
}

/// A vertex in Clip space.
#[derive(Debug, Clone, Copy, PartialEq)]
struct VertexClip {
    position_world: Vec3,
    position: Vec3,
    w: f32,
    uv: Vec2,
    normal: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex<Position: Copy = Vec3> {
    pub position: Position,
    pub uv: Vec2,
    pub normal: Vec3,
}

impl Vertex {
    pub const fn new(
        [x, y, z]: [f32; 3],
        [u, v]: [f32; 2],
        [x_normal, y_normal, z_normal]: [f32; 3],
    ) -> Self {
        Self {
            position: vec3(x, y, z),
            uv: vec2(u, v),
            normal: vec3(x_normal, y_normal, z_normal),
        }
    }
}

pub trait ObjectVertex: Copy {
    fn into_vertex(selfs: [Self; 3]) -> [Vertex; 3];
}

impl ObjectVertex for obj::Position {
    fn into_vertex(positions: [Self; 3]) -> [Vertex; 3] {
        let positions @ [p0, p1, p2] = positions.map(|p| Vec3::from_array(p.position));
        let normal = (p1 - p0).cross(p2 - p0);
        positions.map(|position| Vertex {
            position,
            uv: vec2(0., 0.),
            normal,
        })
    }
}

impl ObjectVertex for obj::Vertex {
    fn into_vertex(vertices: [Self; 3]) -> [Vertex; 3] {
        vertices.map(|vertex| Vertex {
            position: Vec3::from_array(vertex.position),
            uv: vec2(0., 0.),
            normal: Vec3::from_array(vertex.normal),
        })
    }
}

impl ObjectVertex for obj::TexturedVertex {
    fn into_vertex(vertices: [Self; 3]) -> [Vertex; 3] {
        vertices.map(|vertex| Vertex {
            position: Vec3::from_array(vertex.position),
            uv: vec2(vertex.texture[0], vertex.texture[1]),
            normal: Vec3::from_array(vertex.normal),
        })
    }
}

impl<T: Into<Vertex> + Copy> ObjectVertex for T {
    fn into_vertex(selfs: [Self; 3]) -> [Vertex; 3] {
        selfs.map(Into::into)
    }
}

#[allow(dead_code)]
pub trait ObjectIndex: Copy {
    fn to_usize(self) -> usize;
    fn from_usize(u: usize) -> Option<Self>;
}

impl ObjectIndex for u16 {
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(u: usize) -> Option<Self> {
        Self::try_from(u).ok()
    }
}

impl ObjectIndex for u32 {
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(u: usize) -> Option<Self> {
        Self::try_from(u).ok()
    }
}

#[allow(dead_code)]
pub fn draw_vertices<M: Material + ?Sized, V: ObjectVertex>(
    canvas: &mut Canvas,
    options: RenderOptions,
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    material: &M,
    vertices: &[V],
) {
    let mvp = projection * view * model;
    let normal_transform = model.inverse().transpose();
    for vertices_raw in vertices.iter().copied().array_chunks::<3>() {
        let vertices = V::into_vertex(vertices_raw);
        draw_triangle(
            canvas,
            options,
            mvp,
            model,
            normal_transform,
            material,
            vertices,
        );
    }
}

#[allow(dead_code)]
#[expect(clippy::too_many_arguments)]
pub fn draw_vertices_indexed<M: Material + ?Sized, V: ObjectVertex, I: ObjectIndex>(
    canvas: &mut Canvas,
    options: RenderOptions,
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    material: &M,
    vertices: &[V],
    indices: &[I],
) {
    let mvp = projection * view * model;
    let normal_transform = model.inverse().transpose();
    for indices in indices.iter().copied().array_chunks::<3>() {
        let vertices_raw = indices.map(|i| vertices[i.to_usize()]);
        let vertices = V::into_vertex(vertices_raw);
        draw_triangle(
            canvas,
            options,
            mvp,
            model,
            normal_transform,
            material,
            vertices,
        );
    }
}

#[allow(dead_code)]
#[expect(clippy::too_many_arguments)]
pub unsafe fn draw_vertices_indexed_unchecked<
    M: Material + ?Sized,
    V: ObjectVertex,
    I: ObjectIndex,
>(
    canvas: &mut Canvas,
    options: RenderOptions,
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    material: &M,
    vertices: &[V],
    indices: &[I],
) {
    let mvp = projection * view * model;
    let normal_transform = model.inverse().transpose();
    for indices in indices.iter().copied().array_chunks::<3>() {
        let vertices_raw = indices.map(|i| unsafe { *vertices.get_unchecked(i.to_usize()) });
        let vertices = V::into_vertex(vertices_raw);
        draw_triangle(
            canvas,
            options,
            mvp,
            model,
            normal_transform,
            material,
            vertices,
        );
    }
}

#[allow(dead_code)]
pub fn draw_object<M: Material + ?Sized, V: ObjectVertex, I: ObjectIndex>(
    canvas: &mut Canvas,
    options: RenderOptions,
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    material: &M,
    object: &Obj<V, I>,
) {
    draw_vertices_indexed(
        canvas,
        options,
        model,
        view,
        projection,
        material,
        &object.vertices,
        &object.indices,
    );
}

#[allow(dead_code)]
pub unsafe fn draw_object_unchecked<M: Material + ?Sized, V: ObjectVertex, I: ObjectIndex>(
    canvas: &mut Canvas,
    options: RenderOptions,
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    material: &M,
    object: &Obj<V, I>,
) {
    unsafe {
        draw_vertices_indexed_unchecked(
            canvas,
            options,
            model,
            view,
            projection,
            material,
            &object.vertices,
            &object.indices,
        );
    }
}
