use std::{path::PathBuf, str::FromStr as _, sync::Arc};

use clap::Parser as _;
use muilib::{
    Canvas as _, EventLoopExt as _, RectSize, Srgb, Srgba,
    cgmath::*,
    winit::{
        application::ApplicationHandler,
        dpi::PhysicalSize,
        event::{DeviceEvent, DeviceId, MouseButton, MouseScrollDelta, WindowEvent},
        event_loop::{ActiveEventLoop, EventLoop},
        keyboard::PhysicalKey,
        window::{CursorGrabMode, Window, WindowId},
    },
};

mod key_code;

#[derive(Debug, Clone, clap::Parser)]
struct ProgramArgs {
    #[command(subcommand)]
    subcommand: ProgramSubcommand,
}

#[derive(Debug, Clone, clap::Subcommand)]
enum ProgramSubcommand {
    Run(SubcommandRunArgs),
    PackRes(SubcommandPackResArgs),
}

#[derive(Debug, Clone, clap::Parser)]
struct SubcommandRunArgs {
    /// Path of the respack file.
    #[clap(short = 'r', long = "respack", default_value_t = String::from("resources.respack.bin"))]
    respack: String,
    /// Preferred display width.
    #[clap(short = 'W', long = "display-width", default_value_t = 800)]
    display_width: u32,
    /// Preferred display height.
    #[clap(short = 'H', long = "display-height", default_value_t = 600)]
    display_height: u32,
    /// Muilib's resource directory.
    #[clap(short = 'R', long = "muilib-res", default_value_t = String::from("muilib_backend/res/"))]
    muilib_res: String,
}

#[derive(Debug, Clone, clap::Parser)]
struct SubcommandPackResArgs {
    /// Path of the resource directory.
    #[clap(short = 'd', long = "res-dir", default_value_t = String::from("res"))]
    res_dir: String,
    /// Path of the output file.
    #[clap(short = 'o', long = "output", default_value_t = String::from("resources.respack.bin"))]
    output: String,
}

fn main() {
    env_logger::init();

    let program_args = ProgramArgs::parse();

    match program_args.subcommand {
        ProgramSubcommand::PackRes(args) => subcommand_pack_res(args),
        ProgramSubcommand::Run(args) => subcommand_run(args),
    }
}

fn subcommand_pack_res(args: SubcommandPackResArgs) {
    softras::pack_resources(args.res_dir.as_ref(), args.output.as_ref());
}

fn subcommand_run(args: SubcommandRunArgs) {
    let muilib_resources = muilib::AppResources::new(PathBuf::from_str(&args.muilib_res).unwrap());
    let event_loop = EventLoop::builder().build().unwrap();
    event_loop
        .run_lazy_initialized_app::<App, _>((args, &muilib_resources))
        .unwrap();
}

struct App<'cx> {
    window: Arc<Window>,
    uicx: muilib::UiContext<'cx>,
    canvas: muilib::WindowCanvas<'cx>,
    max_width: u32,
    max_height: u32,
    game: softras::Game,
    image_view: muilib::ImageView,
    overlay_text_view: muilib::TextView<'cx>,
    cursor_captured: bool,
    cursor_position: Option<Vector2<f32>>,
}

impl<'cx> App<'cx> {
    fn draw_frame(&mut self, canvas: muilib::CanvasRef) {
        let mut render_pass = self
            .uicx
            .begin_render_pass(&canvas, Srgb::from_hex(0x000000));

        // Draw scene.
        let frame_output = self.game.frame();
        let frame_image = muilib::ImageRef {
            size: RectSize::new(frame_output.display_width, frame_output.display_height),
            format: muilib::wgpu::TextureFormat::Rgba8Unorm,
            data: frame_output.display_buffer,
        };
        if frame_image.size.width > 0 && frame_image.size.height > 0 {
            let frame_texture = self.uicx.create_texture(frame_image);
            self.image_view.set_texture(frame_texture);
            self.uicx
                .prepare_view_bounded(&canvas, canvas.bounds(), &mut self.image_view);
            self.uicx.draw_view(&mut render_pass, &self.image_view);
        }

        // Draw overlay text.
        self.overlay_text_view
            .set_text(String::from(frame_output.overlay_text));
        self.uicx
            .prepare_view(&canvas, point2(4., 4.), &mut self.overlay_text_view);
        self.uicx
            .draw_view(&mut render_pass, &self.overlay_text_view);

        // Capture cursor if requested.
        let requested_captures_cursor = frame_output.captures_cursor;
        if !self.cursor_captured && requested_captures_cursor {
            self.capture_cursor();
        } else if self.cursor_captured && !requested_captures_cursor {
            self.uncapture_cursor();
        }
        self.cursor_captured = requested_captures_cursor;
    }

    fn capture_cursor(&self) {
        match self.window.set_cursor_grab(CursorGrabMode::Locked) {
            Ok(_) => self.window.set_cursor_visible(false),
            Err(error) => log::error!("unable to lock cursor: {error}"),
        }
    }

    fn uncapture_cursor(&self) {
        match self.window.set_cursor_grab(CursorGrabMode::None) {
            Ok(_) => self.window.set_cursor_visible(true),
            Err(error) => log::error!("unable to unlock cursor: {error}"),
        }
    }
}

impl<'cx> muilib::LazyApplicationHandler<(SubcommandRunArgs, &'cx muilib::AppResources), ()>
    for App<'cx>
{
    fn new(
        (args, resources): (SubcommandRunArgs, &'cx muilib::AppResources),
        event_loop: &ActiveEventLoop,
    ) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Software Rasterizer")
                        .with_inner_size(PhysicalSize::new(
                            args.display_width,
                            args.display_height,
                        )),
                )
                .unwrap(),
        );

        let (uicx, canvas) =
            muilib::UiContext::create_for_window(resources, window.clone()).unwrap();
        Self {
            window,
            canvas,
            overlay_text_view: muilib::TextView::new(&uicx)
                .with_font_size(16.)
                .with_bg_color(Srgba::from_hex(0x80808080)),
            image_view: muilib::ImageView::new(RectSize::new(
                args.display_width as f32,
                args.display_height as f32,
            )),
            game: softras::Game::new(),
            uicx,
            max_width: args.display_width,
            max_height: args.display_height,
            cursor_captured: false,
            cursor_position: None,
        }
    }
}

impl<'cx> ApplicationHandler for App<'cx> {
    fn resumed(&mut self, _: &ActiveEventLoop) {}

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                let canvas_ref = self.canvas.create_ref().unwrap();
                self.draw_frame(canvas_ref);
                self.window.pre_present_notify();
                self.canvas.finish_drawing().unwrap();
                self.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                self.canvas.reconfigure_for_size(
                    self.uicx.wgpu_device(),
                    size,
                    self.window.scale_factor(),
                    None,
                );
                let width = size.width as f32;
                let height = size.height as f32;
                let max_width = self.max_width as f32;
                let max_height = self.max_height as f32;
                let (width, height) = match width > height {
                    true => (max_width, max_width / width * height),
                    false => (max_height / height * width, max_height),
                };
                self.game.notify_display_resize(
                    (width as u32).min(size.width),
                    (height as u32).min(size.height),
                );
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: false,
            } => {
                let key_code = match event.physical_key {
                    PhysicalKey::Code(key_code_winit) => key_code::winit_to_softras(key_code_winit),
                    PhysicalKey::Unidentified(_) => softras::KeyCode::Unidentified,
                };
                if event.state.is_pressed() {
                    self.game.notify_key_down(key_code);
                } else {
                    self.game.notify_key_up(key_code);
                }
            }
            WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
            } => {
                let button_ = match button {
                    MouseButton::Left => softras::MouseButton::Left,
                    MouseButton::Right => softras::MouseButton::Right,
                    MouseButton::Middle => softras::MouseButton::Middle,
                    _ => return,
                };
                if state.is_pressed() {
                    self.game.notify_cursor_down(button_);
                } else {
                    self.game.notify_cursor_up(button_);
                }
            }
            WindowEvent::CursorEntered { device_id: _ } => {
                self.game.notify_cursor_entered();
            }
            WindowEvent::CursorLeft { device_id: _ } => {
                self.game.notify_cursor_left();
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                let frame_size = self.image_view.size();
                let window_size = self.window.inner_size();
                self.cursor_position = Some(vec2(
                    position.x as f32 / window_size.width as f32 * frame_size.width,
                    position.y as f32 / window_size.height as f32 * frame_size.height,
                ));
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta: MouseScrollDelta::LineDelta(x, y),
                phase: _,
            } => {
                self.game.notify_cursor_scrolled_lines(x, y);
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta: MouseScrollDelta::PixelDelta(delta),
                phase: _,
            } => {
                self.game
                    .notify_cursor_scrolled_pixels(delta.x as f32, delta.y as f32);
            }
            WindowEvent::Focused(true) => {
                self.game.notify_focused();
                self.window.request_redraw();
            }
            WindowEvent::Focused(false) => {
                self.cursor_position = None;
                self.game.notify_unfocused();
                if self.cursor_captured {
                    self.capture_cursor();
                } else {
                    self.uncapture_cursor();
                }
            }
            _ => (),
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && let Some(cursor_position) = self.cursor_position
        {
            self.game.notify_cursor_moved(
                cursor_position.x,
                cursor_position.y,
                delta.0 as f32,
                delta.1 as f32,
            );
        }
    }
}
