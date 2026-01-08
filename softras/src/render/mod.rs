use std::f32;

use bytemuck::Zeroable;
use glam::*;

mod color;
mod shader;

pub use color::*;
use obj::Obj;
pub use shader::*;

use crate::*;

#[derive(Clone)]
pub struct Canvas {
    width: u32,
    height: u32,
    frame_buffer: Vec<RgbaU8>,
    albedo_buffer: Vec<Rgba>,
    position_buffer: Vec<Vec3>,
    normal_buffer: Vec<Vec3>,
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
            albedo_buffer: Vec::new(),
            position_buffer: Vec::new(),
            normal_buffer: Vec::new(),
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
        self.albedo_buffer.fill(Rgba::zeroed());
        self.position_buffer.fill(Vec3::zeroed());
        self.normal_buffer.fill(Vec3::zeroed());
        self.depth_buffer.fill(1.0f32);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.frame_buffer.resize(self.n_pixels(), Zeroable::zeroed());
        self.albedo_buffer.resize(self.n_pixels(), Zeroable::zeroed());
        self.position_buffer.resize(self.n_pixels(), Zeroable::zeroed());
        self.normal_buffer.resize(self.n_pixels(), Zeroable::zeroed());
        self.depth_buffer.resize(self.n_pixels(), Zeroable::zeroed());
    }

    pub fn n_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

pub fn postprocess<P: Postprocessor + ?Sized>(canvas: &mut Canvas, postprocessor: &P) {
    debug_assert!(canvas.frame_buffer.len() == canvas.n_pixels());
    debug_assert!(canvas.albedo_buffer.len() == canvas.n_pixels());
    debug_assert!(canvas.normal_buffer.len() == canvas.n_pixels());
    debug_assert!(canvas.depth_buffer.len() == canvas.n_pixels());

    for x_pixel in 0..canvas.width {
        for y_pixel in 0..canvas.height {
            let i_pixel = y_pixel as usize * canvas.width as usize + x_pixel as usize;
            let result = unsafe { canvas.frame_buffer.get_unchecked_mut(i_pixel) };
            let albedo = unsafe { *canvas.albedo_buffer.get_unchecked_mut(i_pixel) };
            let position = unsafe { *canvas.position_buffer.get_unchecked_mut(i_pixel) };
            let normal = unsafe { *canvas.normal_buffer.get_unchecked_mut(i_pixel) };
            let depth = unsafe { *canvas.depth_buffer.get_unchecked_mut(i_pixel) };
            let material_input = PostprocessInput {
                albedo,
                position,
                depth,
                normal,
            };
            *result = postprocessor.postprocess(material_input).into();
        }
    }
}

/// Draws a triangle to a canvas.
pub fn draw_triangle<S: Material + ?Sized>(
    canvas: &mut Canvas,
    model_view: Mat4,
    projection: Mat4,
    material: &S,
    vertices: [Vertex; 3],
) {
    let mvp = projection * model_view;
    let vertices_clip: [Vertex<Vec4>; 3] = vertices.map(|vertex| {
        let position = mvp * vertex.position.extend(1.);
        vertex.with_position(position)
    });

    #[rustfmt::skip]
    match vertices_clip.map(|vertex| vertex.position.z >= 0.) {
        // All points are in front of near plane, no near plane clipping needed.
        [true, true, true] => after_near_clipping(canvas, model_view, vertices_clip, material),

        // All points are behind near plane.
        [false, false, false] => (),

        // Clip Case 1: One point is behind the near plane, the other two points are in front.
        [false, true, true] => clip_case_1::<_, 0>(canvas, model_view, vertices_clip, material),
        [true, false, true] => clip_case_1::<_, 1>(canvas, model_view, vertices_clip, material),
        [true, true, false] => clip_case_1::<_, 2>(canvas, model_view, vertices_clip, material),

        // Clip Case 2: Two points are behind the near plane, the other point is in front.
        [true, false, false] => clip_case_2::<_, 0>(canvas, model_view, vertices_clip, material),
        [false, true, false] => clip_case_2::<_, 1>(canvas, model_view, vertices_clip, material),
        [false, false, true] => clip_case_2::<_, 2>(canvas, model_view, vertices_clip, material),
    };
}

// Clip Case 1: One point is behind the near plane, the other two points are in front.
#[cold]
fn clip_case_1<S: Material + ?Sized, const I_BACK: usize>(
    canvas: &mut Canvas,
    model_view: Mat4,
    vertices: [Vertex<Vec4>; 3],
    material: &S,
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
    after_near_clipping(canvas, model_view, result_vertices[0], material);
    after_near_clipping(canvas, model_view, result_vertices[1], material);
}

// Clip Case 2: Two points are behind the near plane, the other point is in front.
#[cold]
#[inline(never)]
fn clip_case_2<S: Material + ?Sized, const I_FRONT: usize>(
    canvas: &mut Canvas,
    model_view: Mat4,
    vertices: [Vertex<Vec4>; 3],
    material: &S,
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
    after_near_clipping(canvas, model_view, positions, material);
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
fn after_near_clipping<S: Material + ?Sized>(
    canvas: &mut Canvas,
    model_view: Mat4,
    vertices_clip: [Vertex<Vec4>; 3],
    material: &S,
) {
    // === Perspective Division ===

    let vertices_ndc: [Vertex; 3] = vertices_clip.map(|vertex| {
        let position = vertex.position.xyz() / vertex.position.w;
        vertex.with_position(position)
    });
    // Convenience declarations.
    let positions_ndc: [Vec2; 3] = vertices_ndc.map(|vertex| vertex.position.xy());
    let depths: [f32; 3] = vertices_ndc.map(|vertex| vertex.position.z);

    if !is_clockwise_winding(positions_ndc) {
        return;
    }

    let normal_transform = model_view.inverse().transpose();
    let normals: [Vec3; 3] = [
        vertices_clip[0].position.w * normal_transform.transform_vector3(vertices_clip[0].normal),
        vertices_clip[1].position.w * normal_transform.transform_vector3(vertices_clip[1].normal),
        vertices_clip[2].position.w * normal_transform.transform_vector3(vertices_clip[2].normal),
    ];

    // === Rasterization ===

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
    if chunk_size.pow(2) < bounds_area {
        // Chunking case.
        // (Chunking is basically only an optimization measure for dealing with triangles that
        // occupy big screen spaces, so if you are checking out the code base just to understand
        // the workings of rasterization process, simply assume the other no chunking path).
        rasterize_chunked(
            canvas,
            vertices_clip,
            material,
            bounds,
            positions_ndc,
            depths,
            normals,
            chunk_size,
        );
    } else {
        // No chunking case.
        rasterize(
            canvas,
            vertices_clip,
            material,
            bounds,
            positions_ndc,
            depths,
            normals,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn rasterize_chunked<S: Material + ?Sized>(
    canvas: &mut Canvas,
    vertices_clip: [Vertex<Vec4>; 3],
    material: &S,
    [x_min, x_max, y_min, y_max]: [u32; 4],
    positions_ndc: [Vec2; 3],
    depths: [f32; 3],
    normals: [Vec3; 3],
    chunk_size: u32,
) {
    let positions_pixel = positions_ndc.map(|p_ndc| {
        vec2(
            ndc_to_pixel_x(p_ndc.x, canvas.width),
            ndc_to_pixel_y(p_ndc.y, canvas.height),
        )
    });
    for y_min in step_range(y_min, (y_max + 1).next_multiple_of(chunk_size), chunk_size) {
        for x_min in step_range(x_min, (x_max + 1).next_multiple_of(chunk_size), chunk_size) {
            let x_max = (x_min + chunk_size).min(canvas.width - 1);
            let y_max = (y_min + chunk_size).min(canvas.height - 1);
            if rect_triangle_overlap(
                positions_pixel,
                [x_min, x_max, y_min, y_max].map(|u| u as f32),
            ) {
                rasterize(
                    canvas,
                    vertices_clip,
                    material,
                    [x_min, x_max, y_min, y_max],
                    positions_ndc,
                    depths,
                    normals,
                );
            }
        }
    }
}

fn rasterize<M: Material + ?Sized>(
    canvas: &mut Canvas,
    vertices_clip: [Vertex<Vec4>; 3],
    material: &M,
    [x_min, x_max, y_min, y_max]: [u32; 4],
    positions_ndc: [Vec2; 3],
    depths: [f32; 3],
    normals: [Vec3; 3],
) {
    debug_assert!(canvas.albedo_buffer.len() == canvas.n_pixels());
    debug_assert!(canvas.normal_buffer.len() == canvas.n_pixels());
    debug_assert!(canvas.depth_buffer.len() == canvas.n_pixels());

    for x_pixel in x_min..=x_max {
        for y_pixel in y_min..=y_max {
            let i_pixel = y_pixel as usize * canvas.width as usize + x_pixel as usize;
            let albedo_sample = unsafe { canvas.albedo_buffer.get_unchecked_mut(i_pixel) };
            let position_sample = unsafe { canvas.position_buffer.get_unchecked_mut(i_pixel) };
            let normal_sample = unsafe { canvas.normal_buffer.get_unchecked_mut(i_pixel) };
            let depth_sample = unsafe { canvas.depth_buffer.get_unchecked_mut(i_pixel) };
            let p_ndc = vec2(
                pixel_to_ndc_x(x_pixel, canvas.width),
                pixel_to_ndc_y(y_pixel, canvas.height),
            );
            let Some(weights) = barycentric_weights_inside(positions_ndc, p_ndc) else {
                continue;
            };
            let depth = triangular_interpolate(weights, depths);
            if *depth_sample <= depth {
                continue;
            }
            let fragment_input = FragmentInput {
                position: vec3(p_ndc.x, p_ndc.y, depth),
                depth,
                uv: triangular_interpolate_vec2(weights, vertices_clip.map(|v| v.uv)),
                normal: triangular_interpolate_vec3(weights, normals),
            };
            let fragment_output = material.fragment(fragment_input);
            *albedo_sample = fragment_output.albedo;
            *position_sample = fragment_input.position;
            *normal_sample = fragment_output.normal;
            *depth_sample = depth;
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

fn point_in_triangle([a, b, c]: [Vec2; 3], p: Vec2) -> bool {
    barycentric_weights_inside([a, b, c], p).is_some()
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

fn barycentric_weights([a, b, c]: [Vec2; 3], p: Vec2) -> [f32; 3] {
    fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        0.5 * (c - a).dot((b - a).perp())
    }
    let area_bcp = signed_area(b, c, p);
    let area_cap = signed_area(c, a, p);
    let area_abp = signed_area(a, b, p);
    let area_total = area_bcp + area_cap + area_abp;
    [
        (1. / area_total) * area_bcp,
        (1. / area_total) * area_cap,
        (1. / area_total) * area_abp,
    ]
}

/// If point `p` is inside the triangle formed by XP components of points `a`, `b`, and `c`,
/// returns the weights of `a`, `b`, and `c` for triangular-interpolation.
fn barycentric_weights_inside([a, b, c]: [Vec2; 3], p: Vec2) -> Option<[f32; 3]> {
    let weights @ [w_a, w_b, w_c] = barycentric_weights([a, b, c], p);
    if (w_a > 0.) == (w_b > 0.) && (w_b > 0.) == (w_c > 0.) {
        Some(weights)
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

pub trait IntoVertex: Copy {
    fn into_vertex(selfs: [Self; 3]) -> [Vertex; 3];
}

impl IntoVertex for obj::Position {
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

impl IntoVertex for obj::Vertex {
    fn into_vertex(vertices: [Self; 3]) -> [Vertex; 3] {
        vertices.map(|vertex| Vertex {
            position: Vec3::from_array(vertex.position),
            uv: vec2(0., 0.),
            normal: Vec3::from_array(vertex.normal),
        })
    }
}

impl IntoVertex for obj::TexturedVertex {
    fn into_vertex(vertices: [Self; 3]) -> [Vertex; 3] {
        vertices.map(|vertex| Vertex {
            position: Vec3::from_array(vertex.position),
            uv: vec2(vertex.texture[0], vertex.texture[1]),
            normal: Vec3::from_array(vertex.normal),
        })
    }
}

impl<T: Into<Vertex> + Copy> IntoVertex for T {
    fn into_vertex(selfs: [Self; 3]) -> [Vertex; 3] {
        selfs.map(Into::into)
    }
}

#[allow(dead_code)]
pub fn draw_vertices<M: Material + ?Sized, V: IntoVertex>(
    canvas: &mut Canvas,
    model_view: Mat4,
    projection: Mat4,
    material: &M,
    vertices: &[V],
) {
    for vertices_raw in vertices.iter().copied().array_chunks::<3>() {
        let vertices = V::into_vertex(vertices_raw);
        draw_triangle(canvas, model_view, projection, material, vertices);
    }
}

#[allow(dead_code)]
pub fn draw_vertices_indexed<M: Material + ?Sized, V: IntoVertex>(
    canvas: &mut Canvas,
    model_view: Mat4,
    projection: Mat4,
    material: &M,
    vertices: &[V],
    indices: &[u16],
) {
    for indices in indices.iter().copied().array_chunks::<3>() {
        let vertices_raw = indices.map(|i| vertices[i as usize]);
        let vertices = V::into_vertex(vertices_raw);
        draw_triangle(canvas, model_view, projection, material, vertices);
    }
}

#[allow(dead_code)]
pub unsafe fn draw_vertices_indexed_unchecked<M: Material + ?Sized, V: IntoVertex>(
    canvas: &mut Canvas,
    model_view: Mat4,
    projection: Mat4,
    material: &M,
    vertices: &[V],
    indices: &[u16],
) {
    for indices in indices.iter().copied().array_chunks::<3>() {
        let vertices_raw = indices.map(|i| unsafe { *vertices.get_unchecked(i as usize) });
        let vertices = V::into_vertex(vertices_raw);
        draw_triangle(canvas, model_view, projection, material, vertices);
    }
}

#[allow(dead_code)]
pub fn draw_object<M: Material + ?Sized, V: IntoVertex>(
    canvas: &mut Canvas,
    model_view: Mat4,
    projection: Mat4,
    material: &M,
    object: &Obj<V>,
) {
    draw_vertices_indexed(
        canvas,
        model_view,
        projection,
        material,
        &object.vertices,
        &object.indices,
    );
}

#[allow(dead_code)]
pub unsafe fn draw_object_unchecked<M: Material + ?Sized, V: IntoVertex>(
    canvas: &mut Canvas,
    model_view: Mat4,
    projection: Mat4,
    material: &M,
    object: &Obj<V>,
) {
    unsafe {
        draw_vertices_indexed_unchecked(
            canvas,
            model_view,
            projection,
            material,
            &object.vertices,
            &object.indices,
        );
    }
}
