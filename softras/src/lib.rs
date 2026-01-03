use std::time::SystemTime;

use bytemuck::{Pod, Zeroable};
use glam::*;

mod key_code;
pub use key_code::*;

mod color;
use crate::color::*;

#[derive(Debug, Clone, Copy)]
pub struct FrameOutput<'game> {
    pub display_buffer: &'game [u8],
    pub display_width: u32,
    pub display_height: u32,

    pub overlay_text: &'game str,

    /// Whether the game window should "capture" the cursor.
    /// If `true`, the cursor should be hidden and locked if possible.
    /// If `false`, the cursor should behave normally.
    pub captures_cursor: bool,
}

/// The main game state struct.
pub struct Game {
    frame_buffer: Vec<RgbaU8>,
    depth_buffer: Vec<f32>,
    frame_width: u32,
    frame_height: u32,

    key_states: [bool; 256],
    cursor_position: Option<Vec2>,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    /* === Start of Public Interface === */

    pub fn new() -> Self {
        Self {
            frame_buffer: Vec::new(),
            depth_buffer: Vec::new(),
            frame_width: 0,
            frame_height: 0,
            key_states: [false; _],
            cursor_position: None,
        }
    }

    pub fn display_buffer(&self) -> &[u8] {
        bytemuck::cast_slice(&self.frame_buffer)
    }

    pub fn display_width(&self) -> u32 {
        self.frame_width
    }

    pub fn display_height(&self) -> u32 {
        self.frame_height
    }

    pub fn frame(&mut self) -> FrameOutput<'_> {
        self.draw();
        FrameOutput {
            display_buffer: bytemuck::cast_slice(&self.frame_buffer),
            display_width: self.frame_width,
            display_height: self.frame_height,
            overlay_text: "SOFTRAS v0.0.0",
            captures_cursor: false,
        }
    }

    pub fn notify_display_resize(&mut self, width: u32, height: u32) {
        self.frame_width = width;
        self.frame_height = height;
    }

    pub fn notify_key_down(&mut self, key_code: KeyCode) {
        self.key_states[key_code as usize] = true;
    }

    pub fn notify_key_up(&mut self, key_code: KeyCode) {
        self.key_states[key_code as usize] = false;
    }

    pub fn notify_cursor_moved(&mut self, x: f32, y: f32) {
        self.cursor_position = Some(vec2(x, y));
    }

    pub fn notify_cursor_entered(&mut self) {}

    pub fn notify_cursor_left(&mut self) {
        self.cursor_position = None;
    }

    pub fn notify_cursor_scrolled_lines(&mut self, x: f32, y: f32) {
        _ = x;
        _ = y;
    }

    pub fn notify_cursor_scrolled_pixels(&mut self, x: f32, y: f32) {
        _ = x;
        _ = y;
    }

    pub fn notify_focused(&mut self) {}

    pub fn notify_unfocused(&mut self) {
        bytemuck::fill_zeroes(&mut self.key_states);
    }

    /* === End of Public Interface === */

    const CUBE_VERTICES: [Vertex; 24] = [
        // South
        Vertex::new([0., 0., 1.], [0., 1.], [0., 0., 1.]),
        Vertex::new([1., 0., 1.], [1., 1.], [0., 0., 1.]),
        Vertex::new([1., 1., 1.], [1., 0.], [0., 0., 1.]),
        Vertex::new([0., 1., 1.], [0., 0.], [0., 0., 1.]),
        // North
        Vertex::new([0., 0., 0.], [1., 1.], [0., 0., -1.]),
        Vertex::new([0., 1., 0.], [1., 0.], [0., 0., -1.]),
        Vertex::new([1., 1., 0.], [0., 0.], [0., 0., -1.]),
        Vertex::new([1., 0., 0.], [0., 1.], [0., 0., -1.]),
        // East
        Vertex::new([1., 0., 0.], [1., 1.], [1., 0., 0.]),
        Vertex::new([1., 1., 0.], [1., 0.], [1., 0., 0.]),
        Vertex::new([1., 1., 1.], [0., 0.], [1., 0., 0.]),
        Vertex::new([1., 0., 1.], [0., 1.], [1., 0., 0.]),
        // West
        Vertex::new([0., 1., 0.], [0., 0.], [-1., 0., 0.]),
        Vertex::new([0., 0., 0.], [0., 1.], [-1., 0., 0.]),
        Vertex::new([0., 0., 1.], [1., 1.], [-1., 0., 0.]),
        Vertex::new([0., 1., 1.], [1., 0.], [-1., 0., 0.]),
        // Up
        Vertex::new([1., 1., 0.], [0., 1.], [0., 1., 0.]),
        Vertex::new([0., 1., 0.], [1., 1.], [0., 1., 0.]),
        Vertex::new([0., 1., 1.], [1., 0.], [0., 1., 0.]),
        Vertex::new([1., 1., 1.], [0., 0.], [0., 1., 0.]),
        // Down
        Vertex::new([0., 0., 0.], [0., 1.], [0., -1., 0.]),
        Vertex::new([1., 0., 0.], [1., 1.], [0., -1., 0.]),
        Vertex::new([1., 0., 1.], [1., 0.], [0., -1., 0.]),
        Vertex::new([0., 0., 1.], [0., 0.], [0., -1., 0.]),
    ];

    #[rustfmt::skip]
    const CUBE_INDICIES: [[u16; 3]; 12] = [
        /* South */ [0, 1, 2], [2, 3, 0],
        /* North */ [4, 5, 6], [6, 7, 4],
        /* East  */ [8, 9, 10], [10, 11, 8],
        /* West  */ [12, 13, 14], [14, 15, 12],
        /* Up    */ [16, 17, 18], [18, 19, 16],
        /* Down  */ [20, 21, 22], [22, 23, 20],
    ];

    fn key_is_down(&self, key_code: KeyCode) -> bool {
        self.key_states[key_code as usize]
    }

    #[allow(dead_code)]
    fn super_is_down(&self) -> bool {
        self.key_is_down(KeyCode::SuperLeft) || self.key_is_down(KeyCode::SuperRight)
    }

    #[allow(dead_code)]
    fn alt_is_down(&self) -> bool {
        self.key_is_down(KeyCode::AltLeft) || self.key_is_down(KeyCode::AltRight)
    }

    #[allow(dead_code)]
    fn shift_is_down(&self) -> bool {
        self.key_is_down(KeyCode::ShiftLeft) || self.key_is_down(KeyCode::ShiftRight)
    }

    fn draw(&mut self) {
        let background_color = Rgb::from_hex(0x040404);
        let n_pixels = self.frame_width as usize * self.frame_height as usize;
        self.frame_buffer.resize(n_pixels, RgbaU8::zeroed());
        self.frame_buffer.fill(background_color.into());
        self.depth_buffer.resize(n_pixels, 0.0f32);
        self.depth_buffer.fill(f32::INFINITY);
        let t = (SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0.0f64, |duration| duration.as_secs_f64())
            % 86400.) as f32;
        let size = 0.5 * f32::min(self.frame_width as f32, self.frame_height as f32);
        let size = vec3(size, size, size);
        let model = Mat4::from_rotation_x(t)
            * Mat4::from_rotation_z(t)
            * Mat4::from_translation(-0.5 * size)
            * Mat4::from_scale(size);
        let shader = BasicShader {
            color: Rgb::from_hex(0x008080),
            light_direction: vec3(-1., -1., -1.),
            ..BasicShader::default()
        };
        let shader: &dyn Shader = &shader;
        for indices in Self::CUBE_INDICIES {
            let triangle = indices.map(|i| Self::CUBE_VERTICES[i as usize]);
            self.draw_triangle(triangle, model, shader);
        }
    }

    fn draw_triangle<S: Shader + ?Sized>(
        &mut self,
        triangle_local: [Vertex; 3],
        model: Mat4,
        shader: &S,
    ) {
        let triangle = triangle_local.map(|v| Vertex {
            position: model.transform_point3(v.position),
            ..v
        });
        for x_pixel in 0..self.frame_width {
            for y_pixel in 0..self.frame_height {
                let i_pixel = y_pixel as usize * self.frame_width as usize + x_pixel as usize;
                let coord = vec2(
                    x_pixel as f32 - 0.5 * (self.frame_width as f32), //
                    -(y_pixel as f32) + 0.5 * (self.frame_height as f32), //
                );
                let rasterize_result = rasterize(triangle.map(|v| v.position.xy()), coord);
                let Some(weights) = rasterize_result else {
                    continue;
                };
                let this_depth = triangular_interpolate(weights, triangle.map(|v| v.position.z));
                let depth = &mut self.depth_buffer[i_pixel];
                if *depth <= this_depth {
                    continue;
                }
                let fragment_input = FragmentInput {
                    position: vec3(coord.x, coord.y, this_depth),
                    uv: vec2(
                        triangular_interpolate(weights, triangle.map(|v| v.uv.x)),
                        triangular_interpolate(weights, triangle.map(|v| v.uv.y)),
                    ),
                    normal: model
                        .transform_vector3(vec3(
                            triangular_interpolate(weights, triangle.map(|v| v.normal.x)),
                            triangular_interpolate(weights, triangle.map(|v| v.normal.y)),
                            triangular_interpolate(weights, triangle.map(|v| v.normal.z)),
                        ))
                        .normalize_or(vec3(0., 0., 1.)),
                };
                let fragment_result = shader.fragment(fragment_input);
                self.frame_buffer[i_pixel] = RgbaU8::from(fragment_result);
                *depth = this_depth;
            }
        }
    }
}

fn triangular_interpolate([w0, w1, w2]: [f32; 3], [x0, x1, x2]: [f32; 3]) -> f32 {
    w0 * x0 + w1 * x1 + w2 * x2
}

/// If point `p` is inside the triangle formed by XP components of points `a`, `b`, and `c`,
/// returns the weights of `a`, `b`, and `c` for triangular-interpolation.
fn rasterize([a, b, c]: [Vec2; 3], p: Vec2) -> Option<[f32; 3]> {
    fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        0.5 * (c - a).dot((b - a).perp())
    }
    let area_bcp = signed_area(b, c, p);
    let area_cap = signed_area(c, a, p);
    let area_abp = signed_area(a, b, p);
    let sign_bcp = area_bcp >= 0.;
    let sign_cap = area_cap >= 0.;
    let sign_abp = area_abp >= 0.;
    let area_total = area_bcp + area_cap + area_abp;
    if sign_bcp && sign_cap && sign_abp && area_total > f32::EPSILON {
        // Inside.
        let inv_area_total = 1. / area_total;
        Some([
            inv_area_total * area_bcp, // weight of A
            inv_area_total * area_cap, // weight of A
            inv_area_total * area_abp, // weight of A
        ])
    } else {
        // Outside.
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    uv: Vec2,
    normal: Vec3,
}

impl Vertex {
    const fn new(
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

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
struct FragmentInput {
    position: Vec3,
    uv: Vec2,
    normal: Vec3,
}

trait Shader {
    fn fragment(&self, input: FragmentInput) -> Rgba;
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BasicShader {
    color: Rgb,
    light_direction: Vec3,
    shading_intensity: f32,
}

impl Default for BasicShader {
    fn default() -> Self {
        Self {
            color: Rgb::from_hex(0x008080),
            light_direction: vec3(1., 1., 1.).normalize(),
            shading_intensity: 0.8,
        }
    }
}

impl Shader for BasicShader {
    fn fragment(&self, input: FragmentInput) -> Rgba {
        let normal = input.normal.normalize_or(vec3(1., 0., 0.));
        let light_direction = self.light_direction.normalize_or(vec3(1., 0., 0.));
        let theta = f32::acos(normal.dot(light_direction));
        let shading = theta / (2. * std::f32::consts::PI);
        Rgb {
            r: self.color.r + self.shading_intensity * (shading - 0.5),
            g: self.color.g + self.shading_intensity * (shading - 0.5),
            b: self.color.b + self.shading_intensity * (shading - 0.5),
        }
        .into()
    }
}
