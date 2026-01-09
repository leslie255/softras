use bytemuck::{Pod, Zeroable};
use glam::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct RgbaU8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl RgbaU8 {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn from_hex(u_hex: u32) -> Self {
        let [r, g, b, a] = u_hex.to_be_bytes();
        Self::new(r, g, b, a)
    }

    pub const fn to_array(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    pub const fn from_array([r, g, b, a]: [u8; 4]) -> Self {
        Self::new(r, g, b, a)
    }
}

impl From<RgbaU8> for [u8; 4] {
    fn from(value: RgbaU8) -> Self {
        value.to_array()
    }
}

impl From<[u8; 4]> for RgbaU8 {
    fn from(array: [u8; 4]) -> Self {
        Self::from_array(array)
    }
}

impl From<Rgb> for RgbaU8 {
    fn from(color: Rgb) -> Self {
        Self {
            r: (color.r * 255.) as u8,
            g: (color.g * 255.) as u8,
            b: (color.b * 255.) as u8,
            a: 255,
        }
    }
}

impl From<Rgba> for RgbaU8 {
    fn from(color: Rgba) -> Self {
        Self {
            r: (color.r.clamp(0., 1.) * 255.) as u8,
            g: (color.g.clamp(0., 1.) * 255.) as u8,
            b: (color.b.clamp(0., 1.) * 255.) as u8,
            a: (color.a.clamp(0., 1.) * 255.) as u8,
        }
    }
}

/// sRGB+A.
#[derive(Default, Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn from_hex(u_hex: u32) -> Self {
        let [r, g, b, a] = u_hex.to_be_bytes();
        Self {
            r: r as f32 / 255.,
            g: g as f32 / 255.,
            b: b as f32 / 255.,
            a: a as f32 / 255.,
        }
    }

    pub const fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    pub const fn from_array([r, g, b, a]: [f32; 4]) -> Self {
        Self::new(r, g, b, a)
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: f32::lerp(self.r, other.r, t),
            g: f32::lerp(self.g, other.g, t),
            b: f32::lerp(self.b, other.b, t),
            a: f32::lerp(self.a, other.a, t),
        }
    }
}

impl From<Rgba> for [f32; 4] {
    fn from(color: Rgba) -> Self {
        color.to_array()
    }
}

impl From<[f32; 4]> for Rgba {
    fn from(array: [f32; 4]) -> Self {
        Self::from_array(array)
    }
}

impl From<RgbaU8> for Rgba {
    fn from(pixel: RgbaU8) -> Self {
        Self {
            r: pixel.r as f32 / 255.,
            g: pixel.g as f32 / 255.,
            b: pixel.b as f32 / 255.,
            a: pixel.a as f32 / 255.,
        }
    }
}

impl From<Vec4> for Rgba {
    fn from(value: Vec4) -> Self {
        Self {
            r: value.x,
            g: value.y,
            b: value.z,
            a: value.w,
        }
    }
}

/// sRGB.
#[derive(Default, Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub const fn from_hex(u: u32) -> Self {
        let [zero, r, g, b] = u.to_be_bytes();
        assert!(zero == 0, "`Srgb::from_hex` called with overflowing value");
        Self {
            r: r as f32 / 255.,
            g: g as f32 / 255.,
            b: b as f32 / 255.,
        }
    }

    pub const fn to_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    pub const fn from_array([r, g, b]: [f32; 3]) -> Self {
        Self::new(r, g, b)
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: f32::lerp(self.r, other.r, t),
            g: f32::lerp(self.g, other.g, t),
            b: f32::lerp(self.b, other.b, t),
        }
    }
}

impl From<Rgb> for [f32; 3] {
    fn from(color: Rgb) -> Self {
        color.to_array()
    }
}

impl From<[f32; 3]> for Rgb {
    fn from(array: [f32; 3]) -> Self {
        Self::from_array(array)
    }
}

impl From<Rgb> for Rgba {
    fn from(color: Rgb) -> Self {
        Self::new(color.r, color.g, color.b, 1.0)
    }
}

impl From<Vec3> for Rgb {
    fn from(value: Vec3) -> Self {
        Self {
            r: value.x,
            g: value.y,
            b: value.z,
        }
    }
}
