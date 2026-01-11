#!/bin/zsh

set -xe

SOFTRAS_DEFAULT_ASSET_PATH="../Resources/assets.respack.bin" cargo build --release --package softras_winit_wgpu --target x86_64-apple-darwin
SOFTRAS_DEFAULT_ASSET_PATH="../Resources/assets.respack.bin" cargo build --release --package softras_winit_wgpu --target aarch64-apple-darwin

mkdir -p "target/Software Rasterizer.app/Contents/"
mkdir -p "target/Software Rasterizer.app/Contents/MacOS/"
mkdir -p "target/Software Rasterizer.app/Contents/Resources/"

lipo -create -output target/Software\ Rasterizer.app/Contents/MacOS/softras_winit_wgpu \
    target/aarch64-apple-darwin/release/softras_winit_wgpu                             \
    target/x86_64-apple-darwin/release/softras_winit_wgpu

cp macos_bundle_info.plist.xml "target/Software Rasterizer.app/Contents/Info.plist"
cp assets.respack.bin "target/Software Rasterizer.app/Contents/Resources/assets.respack.bin"

