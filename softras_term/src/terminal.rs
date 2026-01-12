#![allow(dead_code)]

#[expect(unused_imports)]
use std::{
    io::{self, Read as _, Write as _},
    os::fd::AsRawFd as _,
};

use base64::prelude::*;
use derive_more::Debug;
use termios::*;

pub fn hide_cursor() {
    print!("\x1b[?25l");
}

pub fn show_cursor() {
    print!("\x1b[?25h");
}

pub fn enter_alt_buffer() {
    print!("\x1b[?1049h");
}

pub fn leave_alt_buffer() {
    print!("\x1b[?1049l");
}

pub fn clear_screen() {
    print!("\x1b[2J");
}

pub fn put_cursor(x: u32, y: u32) {
    print!("\x1b[{x};{y}H");
}

/// Delete all images from screen.
pub fn delete_all_images() {
    print!("\x1b_Ga=d\x1b\\");
}

pub fn print_image(width: u32, height: u32, data_rgba: &[u8]) {
    let mut stdout = io::stdout().lock();
    let base64_string = BASE64_STANDARD.encode(data_rgba);
    let n_chunks = base64_string.len().div_ceil(4096);
    let base64_chunks = base64_string.as_bytes().chunks(4096);
    for (i, chunk) in base64_chunks.enumerate() {
        match i {
            0 => write!(&mut stdout, "\x1b_Ga=T,f=32,s={width},v={height},m=1;").unwrap(),
            i if i + 1 == n_chunks => write!(&mut stdout, "\x1b_Gm=0;").unwrap(),
            _ => write!(&mut stdout, "\x1b_Gm=1;").unwrap(),
        }
        stdout.write_all(chunk).unwrap();
        write!(&mut stdout, "\x1b\\").unwrap();
    }
}

/// Enable Kitty Term's comprehensive keyboard handling features.
pub fn enable_advanced_keyboard_input() {
    let flags = 0b00011111;
    print!("\x1b[>{flags}u");
}

/// Disable Kitty Term's comprehensive keyboard handling features.
pub fn disable_advanced_keyboard_input() {
    print!("\x1b[<1u");
}

pub fn enable_raw() {
    let stdout = io::stdout().lock();
    let fd = stdout.as_raw_fd();
    let mut termios = Termios::from_fd(fd).unwrap();
    termios.c_cflag |= CREAD | CLOCAL;
    termios.c_lflag &= !(ICANON | ECHO);
    tcsetattr(fd, TCSAFLUSH, &termios).unwrap();
}

pub fn disable_raw() {
    let stdout = io::stdout().lock();
    let fd = stdout.as_raw_fd();
    let termios = Termios::from_fd(fd).unwrap();
    tcsetattr(fd, TCSAFLUSH, &termios).unwrap();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyEvent {
    pub is_press: bool,
    pub is_repeat: bool,
    pub key_code: softras_core::KeyCode,
}

pub fn wait_for_events(handler: impl FnMut(KeyEvent)) {
    _ = handler;
    todo!()
    // let mut buffer = [0u8; 4096];
    // bytemuck::fill_zeroes(&mut buffer);
    // let stdin = io::stdin();
    // let length = io::stdin().read(&mut buffer).unwrap();
    // let bytes = &buffer[0..length];
    // if let Ok(s) = str::from_utf8(bytes) {
    //     println!("{s:?}");
    // } else {
    //     println!("{bytes:?}");
    // }
}
