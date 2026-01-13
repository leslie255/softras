#![allow(dead_code)]

use bytemuck::{Pod, Zeroable};

use crate::render::*;

pub trait Material {
    fn fragment(&self, input: FragmentInput) -> FragmentOutput;
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct FragmentInput {
    pub position: Vec3,
    pub depth: f32,
    pub uv: Vec2,
    pub normal: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct FragmentOutput {
    pub albedo: Rgb,
    pub specular: f32,
    pub normal: Vec3,
}

pub mod materials {
    use std::{marker::PhantomData, ptr::NonNull};

    use crate::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Colored {
        /// The default value is `Rgb::from_hex(0xFF00FF)`.
        pub color: Rgb,
        pub specular_strength: f32,
    }

    impl Default for Colored {
        fn default() -> Self {
            Self {
                color: Rgb::from_hex(0xFFFFFF),
                specular_strength: 0.5,
            }
        }
    }

    impl Material for Colored {
        fn fragment(&self, input: FragmentInput) -> FragmentOutput {
            FragmentOutput {
                albedo: self.color,
                specular: self.specular_strength,
                normal: input.normal,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Textured<'a> {
        pub pixels: NonNull<RgbaU8>,
        pub width: u32,
        pub specular_strength: f32,
        pub height: u32,
        _marker: PhantomData<&'a [RgbaU8]>,
    }

    impl<'a> Textured<'a> {
        pub fn new(width: u32, height: u32, pixels: &[RgbaU8], specular_strength: f32) -> Self {
            let n_pixels = width as usize * height as usize;
            assert!(
                pixels.len() >= n_pixels,
                "Image of size {width}x{height} must have at least {n_pixels} pixels, but found {}",
                pixels.len()
            );
            Self {
                pixels: NonNull::from(&pixels[0]),
                width,
                height,
                specular_strength,
                _marker: PhantomData,
            }
        }

        pub fn with_image(image: &image::RgbaImage, specular_strength: f32) -> Self {
            Self::new(
                image.width(),
                image.height(),
                bytemuck::cast_slice(image.as_raw()),
                specular_strength,
            )
        }
    }

    impl Material for Textured<'_> {
        fn fragment(&self, input: FragmentInput) -> FragmentOutput {
            // TODO: wrap mode?
            let u = input.uv.x.clamp(0., 1.);
            let v = input.uv.y.clamp(0., 1.);
            let width_f = self.width as f32;
            let height_f = self.height as f32;
            let x = ((u * width_f) as usize).min(self.width as usize - 1);
            let y = ((v * height_f) as usize).min(self.height as usize - 1);
            let pixel = unsafe { *self.pixels.as_ptr().add(y * self.width as usize + x) };
            let color = Rgba::from(pixel);
            FragmentOutput {
                albedo: Rgb {
                    r: color.r,
                    g: color.g,
                    b: color.b,
                },
                specular: self.specular_strength,
                normal: input.normal,
            }
        }
    }
}

pub trait Postprocessor {
    fn postprocess(&self, input: PostprocessInput) -> Rgba;
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct PostprocessInput {
    pub albedo: Rgb,
    pub specular: f32,
    pub position: Vec3,
    pub normal: Vec3,
    pub depth: f32,
}

pub mod postprocessors {
    use crate::*;

    /// Directly outputs color right out.
    #[derive(Default, Debug, Clone, Copy, PartialEq)]
    pub struct Basic {
        /// Color of the background where nothing is drawn on.
        ///
        /// The default value is `Rgba::from_hex(0x000000FF)`.
        pub background_color: Rgba,
    }

    impl Postprocessor for Basic {
        fn postprocess(&self, input: PostprocessInput) -> Rgba {
            match input.depth {
                1. => self.background_color,
                _ => input.albedo.into(),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Light {
        pub position: Vec3,
        pub strength: f32,
        pub specular: bool,
    }

    impl Default for Light {
        fn default() -> Self {
            Self {
                position: vec3(0., 0., 0.),
                strength: 0.,
                specular: false,
            }
        }
    }

    /// Fill color + normal-based directional shading.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PhongLighting<const N_LIGHTS: usize> {
        pub background_color: Rgba,
        pub light_positions: [Light; N_LIGHTS],
        pub view_position: Vec3,
        pub ambient_strength: f32,
        pub diffuse_strength: f32,
        pub specular_power: i32,
    }

    impl<const N_LIGHTS: usize> Default for PhongLighting<N_LIGHTS> {
        fn default() -> Self {
            Self {
                background_color: Rgba::from_hex(0x000000FF),
                light_positions: [Light::default(); _],
                view_position: vec3(0., 0., 0.),
                ambient_strength: 0.6,
                diffuse_strength: 0.4,
                specular_power: 32i32,
            }
        }
    }

    impl<const N_LIGHTS: usize> Postprocessor for PhongLighting<N_LIGHTS> {
        fn postprocess(&self, input: PostprocessInput) -> Rgba {
            if input.depth == 1. {
                return self.background_color;
            }

            fn reflect(i: Vec3, n: Vec3) -> Vec3 {
                i - 2. * n.dot(i) * n
            }

            if input.specular > 1. {
                return input.albedo.into();
            }

            let mut result_strength = self.ambient_strength;
            for light in self.light_positions {
                let normal = input.normal.normalize_or_zero();
                let light_direction = (light.position - input.position).normalize_or_zero();
                let diffuse = self.diffuse_strength * (normal.dot(light_direction)).max(0.);

                let view_direction = (self.view_position - input.position).normalize_or_zero();
                let specular = if input.specular.abs() <= f32::EPSILON || !light.specular {
                    0.
                } else {
                    let reflection_direction = reflect(-light_direction, normal);
                    input.specular
                        * view_direction
                            .dot(reflection_direction)
                            .max(0.)
                            .powi(self.specular_power)
                };

                result_strength += light.strength * diffuse;
                result_strength += light.strength * specular;
            }

            (result_strength * input.albedo).into()
        }
    }
}
