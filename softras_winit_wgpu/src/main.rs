#![cfg_attr(windows, windows_subsystem = "windows")]

mod key_code;

use std::{fs, path::Path, process::exit, sync::Arc};

use clap::Parser as _;
use pollster::FutureExt;
use wgpu::util::DeviceExt as _;
use wgpu_text::glyph_brush::{self, ab_glyph::FontRef};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{CursorGrabMode, Window, WindowId},
};

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
    /// Preferred display width.
    #[clap(short = 'W', long = "display-width", default_value_t = 800)]
    display_width: u32,
    /// Preferred display height.
    #[clap(short = 'H', long = "display-height", default_value_t = 600)]
    display_height: u32,
    /// The game's respack file.
    #[clap(short = 'r', long = "res", default_value_t = String::from("assets.respack.bin"))]
    respack: String,
}

#[derive(Debug, Clone, clap::Parser)]
struct SubcommandPackResArgs {
    /// Path of the resource directory.
    #[clap(short = 'd', long = "res-dir", default_value_t = String::from("softras_core/res"))]
    res_dir: String,
    /// Path of the output file.
    #[clap(short = 'o', long = "output", default_value_t = String::from("assets.respack.bin"))]
    output: String,
}

fn main() {
    env_logger::init();

    match ProgramArgs::try_parse() {
        Ok(program_args) => match program_args.subcommand {
            ProgramSubcommand::PackRes(args) => subcommand_pack_res(args),
            ProgramSubcommand::Run(args) => subcommand_run(args),
        },
        Err(error) => match SubcommandRunArgs::try_parse() {
            Ok(args) => subcommand_run(args),
            Err(_) => error.print().unwrap(),
        },
    }
}

fn subcommand_pack_res(args: SubcommandPackResArgs) {
    match fs::metadata(&args.res_dir) {
        Ok(metadata) if metadata.is_dir() => (),
        Ok(_) => {
            log::error!("path {:?} exists but is not a directory", &args.res_dir);
            exit(1);
        }
        Err(_) => {
            log::error!("directory {:?} does not exist", &args.res_dir);
            exit(1);
        }
    }
    softras_core::pack_resources(args.res_dir.as_ref(), args.output.as_ref())
        .unwrap_or_else(|error| log::error!("error packing resources: {error}"));
}

fn subcommand_run(args: SubcommandRunArgs) {
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(args);
    event_loop.run_app(&mut app).unwrap();
}

static SPACE_MONO_REGULAR_TTF: &[u8] = include_bytes!("../SpaceMono-Regular.ttf");

struct WindowState {
    window: Arc<Window>,
    game: softras_core::Game,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    max_width: u32,
    max_height: u32,
    game_display_size: (u32, u32),
    cursor_captured: bool,
    text_brush: wgpu_text::TextBrush<FontRef<'static>>,
}

impl WindowState {
    fn new(event_loop: &ActiveEventLoop, args: &SubcommandRunArgs) -> WindowState {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Software Rasterizer")
                        .with_inner_size(LogicalSize::new(800, 600)),
                )
                .unwrap(),
        );
        let window_size = window.inner_size();

        let respack_path: &Path = args.respack.as_ref();
        let respack_bytes = fs::read(respack_path).unwrap_or_else(|error| {
            eprintln!("error reading file {}: {error}", respack_path.display());
            event_loop.exit();
            unreachable!()
        });
        let mut game = softras_core::Game::new(respack_bytes).unwrap_or_else(|error| {
            eprintln!("error initializing game: {error}");
            event_loop.exit();
            unreachable!()
        });
        game.notify_display_resize(window_size.width, window_size.height);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .block_on()
            .unwrap();

        let surface = instance.create_surface(window.clone()).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(surface_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let text_brush = wgpu_text::BrushBuilder::using_font_bytes(SPACE_MONO_REGULAR_TTF)
            .unwrap()
            .build(
                &device,
                window_size.width,
                window_size.width,
                surface_format,
            );

        let self_ = WindowState {
            window,
            game,
            device,
            queue,
            surface,
            surface_format,
            render_pipeline,
            bind_group_layout,
            sampler,
            game_display_size: (window_size.width, window_size.height),
            max_width: 800,
            max_height: 600,
            cursor_captured: false,
            text_brush,
        };

        self_.configure_surface();

        self_
    }

    fn configure_surface(&self) {
        let size = self.window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            view_formats: vec![self.surface_format.remove_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    fn draw_frame(&mut self) {
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("failed to acquire next swapchain texture");
        let surface_texture_view =
            surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.surface_format.add_srgb_suffix()),
                    ..Default::default()
                });

        let frame_output = self.game.frame();

        let bind_group: Option<wgpu::BindGroup> =
            match (frame_output.display_width, frame_output.display_height) {
                (0, 0) => None,
                (_, _) => {
                    let texture = self.device.create_texture_with_data(
                        &self.queue,
                        &wgpu::TextureDescriptor {
                            label: None,
                            size: wgpu::Extent3d {
                                width: frame_output.display_width,
                                height: frame_output.display_height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        },
                        wgpu::wgt::TextureDataOrder::default(),
                        frame_output.display_buffer,
                    );
                    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    }))
                }
            };

        let text_section = glyph_brush::Section::default()
            .with_screen_position((8., 8.))
            .add_text(
                glyph_brush::Text::new(frame_output.overlay_text)
                    .with_color([1., 1., 1., 1.])
                    .with_scale(17. * self.window.scale_factor() as f32),
            );

        self.text_brush
            .queue(&self.device, &self.queue, &[text_section])
            .unwrap();

        let requests_capture_cursor = frame_output.captures_cursor;
        if requests_capture_cursor != self.cursor_captured {
            self.cursor_captured = requests_capture_cursor;
            if requests_capture_cursor {
                self.capture_cursor();
            } else {
                self.uncapture_cursor();
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface_texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        if let Some(bind_group) = bind_group {
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.text_brush.draw(&mut render_pass);

        drop(render_pass);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_texture.present();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                self.draw_frame();
                self.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
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
                self.text_brush
                    .resize_view(size.width as f32, size.height as f32, &self.queue);
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
                    PhysicalKey::Unidentified(_) => softras_core::KeyCode::Unidentified,
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
                    MouseButton::Left => softras_core::MouseButton::Left,
                    MouseButton::Right => softras_core::MouseButton::Right,
                    MouseButton::Middle => softras_core::MouseButton::Middle,
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
                let window_size = self.window.inner_size();
                self.game.notify_cursor_moved_to_position(
                    position.x as f32 / window_size.width as f32 * self.game_display_size.0 as f32,
                    position.y as f32 / window_size.height as f32 * self.game_display_size.1 as f32,
                );
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
                if self.cursor_captured {
                    self.capture_cursor();
                } else {
                    self.uncapture_cursor();
                }
            }
            WindowEvent::Focused(false) => {
                self.game.notify_unfocused();
            }
            _ => (),
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.game
                .notify_cursor_moved_by_delta(delta.0 as f32, delta.1 as f32);
        }
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

struct App {
    args: SubcommandRunArgs,
    state: Option<WindowState>,
}

impl App {
    fn new(args: SubcommandRunArgs) -> Self {
        Self { args, state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let state = WindowState::new(event_loop, &self.args);
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.window_event(event_loop, window_id, event);
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.device_event(event_loop, device_id, event);
        }
    }
}
