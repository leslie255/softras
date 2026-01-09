#![feature(iter_array_chunks, read_array)] // FIXME: use of unstable features

use std::{
    fmt::Write as _,
    io,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime},
};

use derive_more::{Display, Error, From};
use glam::*;

mod key_code;
mod render;
mod respack;
mod utils;

pub use key_code::*;
use obj::Obj;
use render::*;
use respack::*;
use utils::*;

/// Packs the needed resources from the directory located in `softras/res` into a one respack file.
///
/// # Arguments
///
/// * `res_dir` - the path to the `softras/res` directory
/// * `output` - the output file (e.g. `"resources.respack.bin"`)
pub fn pack_resources(res_dir: &str, output: &str) -> Result<(), PackResourceError> {
    fn display_path<'a>(x: &'a (impl AsRef<Path> + 'a + ?Sized)) -> impl std::fmt::Display + 'a {
        x.as_ref().display()
    }

    fn pack(res_packer: &mut ResourcePacker, path: &str) -> Result<(), PackResourceError> {
        log::info!("packing resource {} ...", display_path(path));
        res_packer
            .append_file(path)
            .map_err(|error| PackResourceError::MissingResource {
                path: {
                    let root: &Path = res_packer.root_path().as_ref();
                    let subpath: &Path = path.as_ref();
                    root.join(subpath)
                },
                error,
            })
    }

    let mut res_packer = ResourcePacker::new(res_dir);
    pack(&mut res_packer, "models/teapot.obj")?;
    pack(&mut res_packer, "models/suzanne.obj")?;
    pack(&mut res_packer, "textures/grass_block.png")?;

    res_packer
        .finish_into_file(output)
        .map_err(|error| PackResourceError::OutputFileError {
            path: PathBuf::from(output),
            error,
        })?;

    log::info!("finished packing",);

    Ok(())
}

#[derive(Debug, Display, Error)]
pub enum PackResourceError {
    #[display("cannot read file at {}: {error}", path.display())]
    MissingResource { path: PathBuf, error: io::Error },
    #[display("cannot write to output file at {:?}: {error}", path.display())]
    OutputFileError { path: PathBuf, error: io::Error },
}

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
    canvas: Canvas,
    overlay_text: String,
    key_states: [bool; 256],
    cursor_position: Option<Vec2>,
    fps_meter: FpsMeter,
    previous_frame_instant: Option<Instant>,
    is_paused: bool,
    camera: Camera,
    teapot: Obj<obj::Position>,
    suzanne: Obj<obj::TexturedVertex>,
    grass_block_image: image::RgbaImage,
}

#[derive(Debug, Display, Error, From)]
pub enum GameInitError {
    #[display("Malformatted respack: {_0}")]
    RespackReadError(RespackReadError),
    #[display("resource {path:?} not found")]
    MissingResourceItem { path: String },
}

impl Game {
    /* === Start of Public Interface === */

    /// Initialize a new `Game` instance.
    ///
    /// # Arguments
    ///
    /// * `respack_bytes` - the content of the packed respack file created by `pack_resources`
    pub fn new(respack_bytes: Vec<u8>) -> Result<Self, GameInitError> {
        fn get_resource<'a>(respack: &'a ResPack, path: &str) -> Result<&'a [u8], GameInitError> {
            respack
                .get(path)
                .ok_or_else(|| GameInitError::MissingResourceItem {
                    path: String::from(path),
                })
        }

        let respack = ResPack::from_vec(respack_bytes)?;

        Ok(Self {
            canvas: Canvas::new(),
            overlay_text: String::new(),
            key_states: [false; _],
            cursor_position: None,
            fps_meter: FpsMeter::new(),
            previous_frame_instant: None,
            is_paused: false,
            camera: Camera {
                // Add some small deviations so the values never look too rounded.
                position: vec3(0., 1., 10.),
                pitch_degrees: 4. * f32::EPSILON,
                yaw_degrees: 4. * f32::EPSILON,
                fov_y_degrees: 90.,
            },
            teapot: obj::load_obj(get_resource(&respack, "models/teapot.obj")?).unwrap(),
            suzanne: obj::load_obj(get_resource(&respack, "models/suzanne.obj")?).unwrap(),
            grass_block_image: {
                let bytes = get_resource(&respack, "textures/grass_block.png")?;
                image::load_from_memory(bytes).unwrap().to_rgba8()
            },
        })
    }

    pub fn display_buffer(&self) -> &[u8] {
        bytemuck::cast_slice(self.canvas.frame_buffer())
    }

    pub fn display_width(&self) -> u32 {
        self.canvas.width()
    }

    pub fn display_height(&self) -> u32 {
        self.canvas.height()
    }

    pub fn overlay_text(&self) -> &str {
        &self.overlay_text
    }

    pub fn frame(&mut self) -> FrameOutput<'_> {
        let before_frame = Instant::now();
        if let Some(previous_frame_instant) = self.previous_frame_instant {
            self.move_player(before_frame.duration_since(previous_frame_instant));
        }
        self.previous_frame_instant = Some(before_frame);

        self.draw_scene();

        let after_frame = Instant::now();
        let frame_time = (after_frame - before_frame).as_secs_f64();
        self.fps_meter.new_frame_time(frame_time);

        self.update_overlay_text();

        FrameOutput {
            display_buffer: self.display_buffer(),
            display_width: self.display_width(),
            display_height: self.display_height(),
            overlay_text: self.overlay_text(),
            captures_cursor: !self.is_paused,
        }
    }

    pub fn notify_display_resize(&mut self, width: u32, height: u32) {
        self.canvas.resize(width, height);
    }

    pub fn notify_key_down(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Escape => self.is_paused = !self.is_paused,
            KeyCode::F2 => self.screenshot(),
            _ => (),
        }
        self.key_states[key_code as usize] = true;
    }

    pub fn notify_key_up(&mut self, key_code: KeyCode) {
        self.key_states[key_code as usize] = false;
    }

    pub fn notify_cursor_down(&mut self, button: MouseButton) {
        _ = button;
    }

    pub fn notify_cursor_up(&mut self, button: MouseButton) {
        _ = button;
    }

    pub fn notify_cursor_moved(&mut self, x: f32, y: f32, dx: f32, dy: f32) {
        self.cursor_position = Some(vec2(x, y));
        self.rotation_player(vec2(dx, dy));
    }

    pub fn notify_cursor_entered(&mut self) {}

    pub fn notify_cursor_left(&mut self) {
        self.cursor_position = None;
    }

    pub fn notify_cursor_scrolled_lines(&mut self, x: f32, y: f32) {
        _ = x;
        let rate = if self.control_is_down() { 4. } else { 1. };
        let fov = &mut self.camera.fov_y_degrees;
        *fov = (*fov + rate * 0.1 * y * 14.).clamp(20., 160.);
    }

    pub fn notify_cursor_scrolled_pixels(&mut self, x: f32, y: f32) {
        _ = x;
        let rate = if self.control_is_down() { 4. } else { 1. };
        let fov = &mut self.camera.fov_y_degrees;
        *fov = (*fov + rate * 0.1 * y).clamp(20., 160.);
    }

    pub fn notify_focused(&mut self) {
        bytemuck::fill_zeroes(&mut self.key_states);
        self.cursor_position = None;
    }

    pub fn notify_unfocused(&mut self) {
        bytemuck::fill_zeroes(&mut self.key_states);
        self.cursor_position = None;
    }

    /* === End of Public Interface === */

    #[rustfmt::skip]
    const GRASS_BLOCK_VERTICES: [Vertex; 24] = {
        let bottom0: f32 = 0.;
        let bottom1: f32 = 15. / 48.;
        let side0: f32 = 16. / 48.;
        let side1: f32 = 31. / 48.;
        let top0: f32 = 32. / 48.;
        let top1: f32 = 1.;
        [
            // South
            Vertex::new([0., 0., 1.], [side0, 1.], [0., 0., 1.]),
            Vertex::new([1., 0., 1.], [side1, 1.], [0., 0., 1.]),
            Vertex::new([1., 1., 1.], [side1, 0.], [0., 0., 1.]),
            Vertex::new([0., 1., 1.], [side0, 0.], [0., 0., 1.]),
            // North
            Vertex::new([0., 0., 0.], [side1, 1.], [0., 0., -1.]),
            Vertex::new([0., 1., 0.], [side1, 0.], [0., 0., -1.]),
            Vertex::new([1., 1., 0.], [side0, 0.], [0., 0., -1.]),
            Vertex::new([1., 0., 0.], [side0, 1.], [0., 0., -1.]),
            // East
            Vertex::new([1., 0., 0.], [side1, 1.], [1., 0., 0.]),
            Vertex::new([1., 1., 0.], [side1, 0.], [1., 0., 0.]),
            Vertex::new([1., 1., 1.], [side0, 0.], [1., 0., 0.]),
            Vertex::new([1., 0., 1.], [side0, 1.], [1., 0., 0.]),
            // West
            Vertex::new([0., 1., 0.], [side0, 0.], [-1., 0., 0.]),
            Vertex::new([0., 0., 0.], [side0, 1.], [-1., 0., 0.]),
            Vertex::new([0., 0., 1.], [side1, 1.], [-1., 0., 0.]),
            Vertex::new([0., 1., 1.], [side1, 0.], [-1., 0., 0.]),
            // Up
            Vertex::new([1., 1., 0.], [top0, 1.], [0., 1., 0.]),
            Vertex::new([0., 1., 0.], [top1, 1.], [0., 1., 0.]),
            Vertex::new([0., 1., 1.], [top1, 0.], [0., 1., 0.]),
            Vertex::new([1., 1., 1.], [top0, 0.], [0., 1., 0.]),
            // Down
            Vertex::new([0., 0., 0.], [bottom0, 1.], [0., -1., 0.]),
            Vertex::new([1., 0., 0.], [bottom1, 1.], [0., -1., 0.]),
            Vertex::new([1., 0., 1.], [bottom1, 0.], [0., -1., 0.]),
            Vertex::new([0., 0., 1.], [bottom0, 0.], [0., -1., 0.]),
        ]
    };

    #[rustfmt::skip]
    const GRASS_BLOCK_INDICIES: [u16; 36] = [
        /* South */ 0, 1, 2, 2, 3, 0,
        /* North */ 4, 5, 6, 6, 7, 4,
        /* East  */ 8, 9, 10, 10, 11, 8,
        /* West  */ 12, 13, 14, 14, 15, 12,
        /* Up    */ 16, 17, 18, 18, 19, 16,
        /* Down  */ 20, 21, 22, 22, 23, 20,
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
    fn control_is_down(&self) -> bool {
        self.key_is_down(KeyCode::ControlLeft) || self.key_is_down(KeyCode::ControlRight)
    }

    #[allow(dead_code)]
    fn shift_is_down(&self) -> bool {
        self.key_is_down(KeyCode::ShiftLeft) || self.key_is_down(KeyCode::ShiftRight)
    }

    fn update_overlay_text(&mut self) {
        self.overlay_text.clear();

        // Pause prompt.
        if self.is_paused {
            _ = writeln!(&mut self.overlay_text, "[ESC] PAUSED");
        }

        // Name and version.
        let crate_version = env!("CARGO_PKG_VERSION");
        if cfg!(debug_assertions) {
            _ = writeln!(
                &mut self.overlay_text,
                "SOFTWARE RASTERIZER v{crate_version} (DEBUG BUILD)"
            );
        } else {
            _ = writeln!(
                &mut self.overlay_text,
                "SOFTWARE RASTERIZER v{crate_version}"
            );
        }

        // Resolution.
        _ = writeln!(
            &mut self.overlay_text,
            "resolution: {}x{}",
            self.canvas.width(),
            self.canvas.height(),
        );

        // FPS.
        _ = write!(&mut self.overlay_text, "FPS: ");
        match self.fps_meter.fps() {
            Some(fps) => _ = write!(&mut self.overlay_text, "{:3.0}", fps),
            None => _ = write!(&mut self.overlay_text, "***"),
        };
        _ = write!(&mut self.overlay_text, ", avg over 12: ");
        match self.fps_meter.average_fps() {
            Some(avarage_fps) => _ = writeln!(&mut self.overlay_text, "{:3.0}", avarage_fps),
            None => _ = writeln!(&mut self.overlay_text, "***"),
        };

        // Camera position/direction.
        _ = writeln!(
            &mut self.overlay_text,
            "camera x/y/z: {:.04}/{:.04}/{:.04}",
            self.camera.position.x, self.camera.position.y, self.camera.position.z,
        );

        // Camera position/direction.
        _ = writeln!(
            &mut self.overlay_text,
            "camera pitch/yaw: {:.04}/{:.04}",
            self.camera.pitch_degrees, self.camera.yaw_degrees,
        );

        // Camera position/direction.
        _ = writeln!(
            &mut self.overlay_text,
            "camera FOV: {:.04}",
            self.camera.fov_y_degrees,
        );
    }

    fn draw_scene(&mut self) {
        if self.canvas.width() == 0 || self.canvas.height() == 0 {
            return;
        }
        self.canvas.clear();

        let view = self.camera.view_matrix();
        let projection = self
            .camera
            .projection_matrix(self.canvas.width() as f32, self.canvas.height() as f32);

        let mut draw_teapot =
            |scale: f32, position: Vec3, rotation_degrees: f32, color: Rgb| -> () {
                let material = materials::Colored { fill_color: color };
                let model = Mat4::from_translation(position)
                    * Mat4::from_scale(vec3(scale, scale, scale))
                    * Mat4::from_rotation_y(rotation_degrees.to_degrees());
                unsafe {
                    draw_object_unchecked(
                        &mut self.canvas,
                        view * model,
                        projection,
                        &material,
                        &self.teapot,
                    );
                }
            };
        draw_teapot(1., vec3(0., 0., 0.), 0., Rgb::from_hex(0xC0C0C0));
        draw_teapot(0.6, vec3(3., 0., 3.), 45., Rgb::from_hex(0xE0A080));
        draw_teapot(0.7, vec3(-3., 0., 4.), 225., Rgb::from_hex(0xA080E0));

        let mut draw_suzanne =
            |scale: f32, position: Vec3, rotation_degrees: f32, color: Rgb| -> () {
                let material = materials::Colored { fill_color: color };
                let model = Mat4::from_translation(position)
                    * Mat4::from_scale(vec3(scale, scale, scale))
                    * Mat4::from_rotation_y(rotation_degrees.to_degrees());
                unsafe {
                    draw_object_unchecked(
                        &mut self.canvas,
                        view * model,
                        projection,
                        &material,
                        &self.suzanne,
                    );
                }
            };
        draw_suzanne(1., vec3(5., 1., 6.), -135., Rgb::from_hex(0xC08040));

        let mut draw_grass_block = |scale: f32, position: Vec3, rotation_degrees: f32| -> () {
            let pixels: &[u8] = self.grass_block_image.as_raw();
            let material = materials::Textured::new(
                self.grass_block_image.width(),
                self.grass_block_image.height(),
                bytemuck::cast_slice(pixels),
            );
            let model = Mat4::from_translation(position)
                * Mat4::from_scale(vec3(scale, scale, scale))
                * Mat4::from_rotation_y(rotation_degrees.to_degrees());
            draw_vertices_indexed(
                &mut self.canvas,
                view * model,
                projection,
                &material,
                &Self::GRASS_BLOCK_VERTICES,
                &Self::GRASS_BLOCK_INDICIES,
            );
        };
        draw_grass_block(1., vec3(6., 0., 2.), 30.);
        draw_grass_block(1.5, vec3(0., 0., 6.), 45.);

        // Draw ground.
        {
            #[rustfmt::skip]
            let ground_vertices = [
                Vertex::new([1., 0., 0.], [0., 1.], [0., 1., 0.]),
                Vertex::new([0., 0., 0.], [1., 1.], [0., 1., 0.]),
                Vertex::new([0., 0., 1.], [1., 0.], [0., 1., 0.]),
                Vertex::new([1., 0., 1.], [0., 0.], [0., 1., 0.]),
            ];
            let ground_indices = [0u16, 1, 2, 2, 3, 0];
            let material = materials::Colored {
                fill_color: Rgb::from_hex(0x101820),
            };
            let size = 100.0f32;
            let height = 0.1f32;
            let model = Mat4::from_translation(0.5 * vec3(-size, -height, -size))
                * Mat4::from_scale(vec3(size, height, size));
            unsafe {
                draw_vertices_indexed_unchecked(
                    &mut self.canvas,
                    view * model,
                    projection,
                    &material,
                    &ground_vertices,
                    &ground_indices,
                );
            }
        }

        let postprocessor = postprocessors::DirectionalShading {
            background_color: Rgb::from_hex(0x141414),
            light_direction: {
                let t = (SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()
                    % 86400.) as f32;
                view.transform_vector3(vec3(t.cos(), -1., t.sin()))
                    .normalize()
            },
            shading_intensity: 1.5,
            highlightness: 0.7,
        };
        // let postprocessor = postprocessors::Basic;
        postprocess(&mut self.canvas, &postprocessor);
    }

    fn screenshot(&self) {
        let path: &Path = "screenshot.png".as_ref();
        match image::save_buffer(
            path,
            bytemuck::cast_slice(self.canvas.frame_buffer()),
            self.canvas.width(),
            self.canvas.height(),
            image::ExtendedColorType::Rgba8,
        ) {
            Ok(()) => log::info!("saved screenshot to {}", path.display()),
            Err(error) => log::error!("unable to save screenshot to {}: {error}", path.display()),
        }
    }

    fn move_player(&mut self, frame_duration: Duration) {
        if self.is_paused {
            return;
        }
        let mut movement = vec3(0., 0., 0.);
        if self.key_is_down(KeyCode::KeyW) {
            movement.z += 1.;
        }
        if self.key_is_down(KeyCode::KeyS) {
            movement.z -= 1.;
        }
        if self.key_is_down(KeyCode::KeyA) {
            movement.x -= 1.;
        }
        if self.key_is_down(KeyCode::KeyD) {
            movement.x += 1.;
        }
        if self.key_is_down(KeyCode::Space) {
            movement.y += 1.;
        }
        if self.key_is_down(KeyCode::KeyR) || self.key_is_down(KeyCode::ShiftLeft) {
            movement.y -= 1.;
        }
        movement = 8. * movement.normalize();
        if movement.x.is_nan() | movement.y.is_nan() | movement.z.is_nan() {
            return;
        }
        if self.key_is_down(KeyCode::ControlLeft) && movement.z >= 0. {
            movement.z *= 2.;
        }
        if self.key_is_down(KeyCode::F3) {
            movement *= 32.;
        }
        if self.alt_is_down() {
            movement /= 8.;
        }
        movement *= frame_duration.as_secs_f32();

        self.camera.move_(movement);
    }

    fn rotation_player(&mut self, delta: Vec2) {
        if self.is_paused {
            return;
        }
        let sensitivity = 1. / 11.;
        self.camera.yaw_degrees += sensitivity * delta.x;
        self.camera.pitch_degrees -= sensitivity * delta.y;
        self.camera.pitch_degrees = self.camera.pitch_degrees.clamp(-90. + 0.0001, 90. - 0.0001);
        if self.camera.yaw_degrees <= -180. {
            self.camera.yaw_degrees += 360.;
        } else if self.camera.yaw_degrees >= 180. {
            self.camera.yaw_degrees -= 360.;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FpsMeter {
    frame_times: [f64; 12],
    cursor: usize,
    average_frame_time: Option<f64>,
}

impl Default for FpsMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl FpsMeter {
    const fn new() -> Self {
        Self {
            frame_times: [0.; _],
            cursor: 0,
            average_frame_time: None,
        }
    }

    fn new_frame_time(&mut self, frame_time: f64) {
        let i = self.cursor;
        self.cursor += 1;
        self.cursor %= self.frame_times.len();
        self.frame_times[i] = frame_time;
        let frame_time = self.frame_times.iter().sum::<f64>() / (self.frame_times.len() as f64);
        match &mut self.average_frame_time {
            Some(frame_time_) => *frame_time_ = frame_time,
            frame_time_ @ None if self.cursor == 0 => *frame_time_ = Some(frame_time),
            None => (),
        }
    }

    fn average_frame_time(&self) -> Option<f64> {
        self.average_frame_time
    }

    fn average_fps(&self) -> Option<f64> {
        self.average_frame_time().map(|f| 1. / f)
    }

    fn fps(&self) -> Option<f64> {
        if self.average_frame_time.is_some() || self.cursor >= 1 {
            let i = self
                .cursor
                .checked_sub(1)
                .unwrap_or(self.frame_times.len() - 1);
            Some(1. / self.frame_times[i])
        } else {
            None
        }
    }
}
