use std::f32;

use bytemuck::Zeroable as _;
use glam::*;

mod color;
mod shader;

pub use color::*;
pub use shader::*;

#[derive(Clone)]
pub struct Canvas {
    width: u32,
    height: u32,
    frame_buffer: Vec<RgbaU8>,
    depth_buffer: Vec<f32>,
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
    pub fn depth_buffer(&self) -> &[f32] {
        &self.depth_buffer
    }

    pub fn clear(&mut self, clear_color: RgbaU8) {
        self.frame_buffer.fill(clear_color);
        self.depth_buffer.fill(1.0f32);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.frame_buffer.resize(self.n_pixels(), RgbaU8::zeroed());
        self.depth_buffer.resize(self.n_pixels(), f32::zeroed());
    }

    pub fn n_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Draws a triangle to a canvas.
pub fn draw_triangle<S: Shader + ?Sized>(
    canvas: &mut Canvas,
    model_view: Mat4,
    projection: Mat4,
    vertices: [Vertex; 3],
    shader: &S,
) {
    let mvp = projection * model_view;
    let vertices_clip: [Vertex<Vec4>; 3] = vertices.map(|vertex| {
        let position = mvp * vertex.position.extend(1.);
        vertex.with_position(position)
    });

    #[rustfmt::skip]
    match vertices_clip.map(|vertex| vertex.position.z >= 0.) {
        // All points are in front of near plane, no near plane clipping needed.
        [true, true, true] => after_near_clipping(canvas, model_view, vertices_clip, shader),

        // All points are behind near plane.
        [false, false, false] => (),

        // Clip Case 1: One point is behind the near plane, the other two points are in front.
        [false, true, true] => clip_case_1::<_, 0>(canvas, model_view, vertices_clip, shader),
        [true, false, true] => clip_case_1::<_, 1>(canvas, model_view, vertices_clip, shader),
        [true, true, false] => clip_case_1::<_, 2>(canvas, model_view, vertices_clip, shader),

        // Clip Case 2: Two points are behind the near plane, the other point is in front.
        [true, false, false] => clip_case_2::<_, 0>(canvas, model_view, vertices_clip, shader),
        [false, true, false] => clip_case_2::<_, 1>(canvas, model_view, vertices_clip, shader),
        [false, false, true] => clip_case_2::<_, 2>(canvas, model_view, vertices_clip, shader),
    };
}

// Clip Case 1: One point is behind the near plane, the other two points are in front.
#[cold]
#[inline(never)]
fn clip_case_1<S: Shader + ?Sized, const I_BACK: usize>(
    canvas: &mut Canvas,
    model_view: Mat4,
    vertices: [Vertex<Vec4>; 3],
    shader: &S,
) {
    const { assert!(I_BACK < 3) };
    let [v0, v1, v2] = vertices;
    let [v0_vb, v1_vb, v2_vb] = vertices.map(|p| near_plane_intersection(p, vertices[I_BACK]));
    let result_vertices: [[Vertex<Vec4>; 3]; 2] = match I_BACK {
        0 => [[v1_vb, v1, v2], [v1_vb, v2, v2_vb]],
        1 => [[v2_vb, v2, v0], [v2_vb, v0, v0_vb]],
        2 => [[v0_vb, v0, v1], [v0_vb, v1, v1_vb]],
        _ => unreachable!(),
    };
    after_near_clipping(canvas, model_view, result_vertices[0], shader);
    after_near_clipping(canvas, model_view, result_vertices[1], shader);
}

// Clip Case 2: Two points are behind the near plane, the other point is in front.
#[cold]
#[inline(never)]
fn clip_case_2<S: Shader + ?Sized, const I_FRONT: usize>(
    canvas: &mut Canvas,
    model_view: Mat4,
    vertices: [Vertex<Vec4>; 3],
    shader: &S,
) {
    const { assert!(I_FRONT < 3) };
    let [v0, v1, v2] = vertices;
    let [v0_vf, v1_vf, v2_vf] = vertices.map(|p| near_plane_intersection(vertices[I_FRONT], p));
    let positions: [Vertex<Vec4>; 3] = match I_FRONT {
        0 => [v0, v1_vf, v2_vf],
        1 => [v0_vf, v1, v2_vf],
        2 => [v0_vf, v1_vf, v2],
        _ => unreachable!(),
    };
    after_near_clipping(canvas, model_view, positions, shader);
}

fn near_plane_intersection(v_front: Vertex<Vec4>, v_back: Vertex<Vec4>) -> Vertex<Vec4> {
    let t = v_front.position.z / (v_front.position.z - v_back.position.z);
    Vertex {
        position: v_front.position + t * (v_back.position - v_front.position),
        uv: v_front.uv + t * (v_back.uv - v_front.uv),
        normal: v_front.normal + t * (v_back.normal - v_front.normal),
    }
}

/// Handle the rendering from after clipping (but before perspective division).
#[inline(always)]
fn after_near_clipping<S: Shader + ?Sized>(
    canvas: &mut Canvas,
    model_view: Mat4,
    vertices_clip: [Vertex<Vec4>; 3],
    shader: &S,
) {
    debug_assert!(canvas.frame_buffer.len() == canvas.n_pixels());
    debug_assert!(canvas.depth_buffer.len() == canvas.n_pixels());

    // Perspective division.

    let vertices_ndc: [Vertex; 3] = vertices_clip.map(|vertex| {
        let position = vertex.position.xyz() / vertex.position.w;
        vertex.with_position(position)
    });
    let positions_ndc: [Vec2; 3] = vertices_ndc.map(|vertex| vertex.position.xy());
    let depths: [f32; 3] = vertices_ndc.map(|vertex| vertex.position.z);
    let normal_transform = model_view.inverse().transpose();
    let normals: [Vec3; 3] = [
        vertices_clip[0].position.w * normal_transform.transform_vector3(vertices_clip[0].normal),
        vertices_clip[1].position.w * normal_transform.transform_vector3(vertices_clip[1].normal),
        vertices_clip[2].position.w * normal_transform.transform_vector3(vertices_clip[2].normal),
    ];
    // let normals: [Vec3; 3] = [
    //     vertices_clip[0].position.w * vertices_clip[0].normal,
    //     vertices_clip[1].position.w * vertices_clip[1].normal,
    //     vertices_clip[2].position.w * vertices_clip[2].normal,
    // ];
    if !is_clockwise_winding(positions_ndc) {
        return;
    }

    // Rasterization.

    // Bounding box of pixels that we need to sample.
    let [x_min, x_max, y_min, y_max]: [u32; 4] = {
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
    for x_pixel in x_min..=x_max {
        for y_pixel in y_min..=y_max {
            let i_pixel = y_pixel as usize * canvas.width as usize + x_pixel as usize;
            let depth_sample = unsafe { canvas.depth_buffer.get_unchecked_mut(i_pixel) };
            let pixel_sample = unsafe { canvas.frame_buffer.get_unchecked_mut(i_pixel) };
            let p_ndc = vec2(
                pixel_to_ndc_x(x_pixel, canvas.width),
                pixel_to_ndc_y(y_pixel, canvas.height),
            );
            let Some(weights) = triangle_weights(positions_ndc, p_ndc) else {
                continue;
            };
            let depth = triangular_interpolate(weights, depths);
            if *depth_sample <= depth {
                continue;
            }
            *depth_sample = depth;
            let fragment_input = FragmentInput {
                position: vec3(p_ndc.x, p_ndc.y, depth),
                depth,
                uv: triangular_interpolate_vec2(weights, vertices_clip.map(|v| v.uv)),
                normal: triangular_interpolate_vec3(weights, normals),
            };
            let fragment_result = shader.fragment(fragment_input);
            *pixel_sample = RgbaU8::from(fragment_result);
        }
    }
}

fn triangular_interpolate([w0, w1, w2]: [f32; 3], [x0, x1, x2]: [f32; 3]) -> f32 {
    w0 * x0 + w1 * x1 + w2 * x2
}

fn triangular_interpolate_vec2(weights: [f32; 3], [v0, v1, v2]: [Vec2; 3]) -> Vec2 {
    vec2(
        triangular_interpolate(weights, [v0.x, v1.x, v2.x]),
        triangular_interpolate(weights, [v0.y, v1.y, v2.y]),
    )
}

fn triangular_interpolate_vec3(weights: [f32; 3], [v0, v1, v2]: [Vec3; 3]) -> Vec3 {
    vec3(
        triangular_interpolate(weights, [v0.x, v1.x, v2.x]),
        triangular_interpolate(weights, [v0.y, v1.y, v2.y]),
        triangular_interpolate(weights, [v0.z, v1.z, v2.z]),
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
fn triangle_weights([a, b, c]: [Vec2; 3], p: Vec2) -> Option<[f32; 3]> {
    fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        0.5 * (c - a).dot((b - a).perp())
    }
    let area_bcp = signed_area(b, c, p);
    let area_cap = signed_area(c, a, p);
    let area_abp = signed_area(a, b, p);
    let area_total = area_bcp + area_cap + area_abp;
    if (area_bcp > 0.) == (area_cap > 0.) && (area_cap > 0.) == (area_abp > 0.) {
        // Inside.
        let inv_area_total = 1. / area_total;
        Some([
            inv_area_total * area_bcp, // weight of A
            inv_area_total * area_cap, // weight of B
            inv_area_total * area_abp, // weight of C
        ])
    } else {
        // Outside.
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

impl<T: Copy> Vertex<T> {
    pub const fn with_position<U: Copy>(self, position: U) -> Vertex<U> {
        Vertex {
            position,
            uv: self.uv,
            normal: self.normal,
        }
    }
}
