#![feature(iter_array_chunks)]

use std::{
    fs, io::Write, path::Path, process::exit, thread::sleep, time::{Duration, Instant}
};

use clap::Parser as _;

mod terminal;

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

const DEFAULT_RES_PATH: &str = match option_env!("SOFTRAS_DEFAULT_ASSET_PATH") {
    Some(path) => path,
    None => "assets.respack.bin",
};

#[derive(Debug, Clone, clap::Parser)]
struct SubcommandRunArgs {
    /// Preferred display width.
    #[clap(short = 'W', long, default_value_t = 800)]
    display_width: u32,
    /// Preferred display height.
    #[clap(short = 'H', long, default_value_t = 600)]
    display_height: u32,
    /// The game's respack file.
    #[clap(short = 'r', long = "res", default_value_t = String::from(DEFAULT_RES_PATH))]
    respack: String,
    /// The target FPS.
    #[clap(long, default_value_t = 60.)]
    fps: f64,
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
    let res_dir: &Path = args.res_dir.as_ref();
    match fs::metadata(res_dir) {
        Ok(metadata) if metadata.is_dir() => (),
        Ok(_) => {
            log::error!("path {} exists but is not a directory", res_dir.display());
            exit(1);
        }
        Err(_) => {
            log::error!("directory {} does not exist", res_dir.display());
            exit(1);
        }
    }
    softras_core::pack_resources(args.res_dir.as_ref(), args.output.as_ref())
        .unwrap_or_else(|error| log::error!("error packing resources: {error}"));
}

fn init_game(args: &SubcommandRunArgs) -> Result<softras_core::Game, String> {
    let respack_path: &Path = args.respack.as_ref();
    log::info!("using respack path: {}", respack_path.display());
    let respack_bytes = match fs::read(respack_path) {
        Ok(respack_bytes) => respack_bytes,
        Err(error) => {
            return Err(format!(
                "error reading assets file {}: {error}",
                respack_path.display()
            ));
        }
    };
    softras_core::Game::new(respack_bytes).map_err(|error| error.to_string())
}

fn subcommand_run(args: SubcommandRunArgs) {
    let mut game = init_game(&args).unwrap_or_else(|error| {
        log::error!("{error}");
        exit(1);
    });
    game.notify_display_resize(args.display_width, args.display_height);

    ctrlc::set_handler(|| {
        // Terminal cleanup.
        terminal::disable_raw();
        terminal::show_cursor();
        terminal::delete_all_images();
        terminal::disable_advanced_keyboard_input();
        terminal::leave_alt_buffer();
        exit(0);
    })
    .unwrap();

    // Terminal setup.
    terminal::enter_alt_buffer();
    terminal::enable_raw();
    terminal::enable_advanced_keyboard_input();
    terminal::hide_cursor();

    let target_frame_time = 1. / args.fps;
    loop {
        let before = Instant::now();

        terminal::put_cursor(0, 0);
        terminal::clear_screen();

        let frame_output = game.frame();
        println!("{}", frame_output.overlay_text);
        terminal::print_image(
            frame_output.display_width,
            frame_output.display_height,
            frame_output.display_buffer,
        );
        println!();
        std::io::stdout().flush().unwrap();

        let after = Instant::now();
        let frame_seconds = after.duration_since(before).as_secs_f64();
        let wait_seconds = (target_frame_time - frame_seconds).max(0.);
        sleep(Duration::from_secs_f64(wait_seconds));
    }
}
