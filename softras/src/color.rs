use bytemuck::{Pod, Zeroable};
use glam::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct RgbaPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// sRGB+A.
#[derive(Default, Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct Srgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Srgba {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn from_hex(u: u32) -> Self {
        let [r, g, b, a] = u.to_be_bytes();
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

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: f32::lerp(self.r, other.r, t),
            g: f32::lerp(self.g, other.g, t),
            b: f32::lerp(self.b, other.b, t),
            a: f32::lerp(self.a, other.a, t),
        }
    }
}

impl From<Srgba> for [f32; 4] {
    fn from(srgba: Srgba) -> Self {
        srgba.to_array()
    }
}

impl From<[f32; 4]> for Srgba {
    fn from([r, g, b, a]: [f32; 4]) -> Self {
        Self { r, g, b, a }
    }
}

impl From<Vec4> for Srgba {
    fn from(value: Vec4) -> Self {
        Self {
            r: value.x,
            g: value.y,
            b: value.z,
            a: value.w,
        }
    }
}

impl From<Srgba> for RgbaPixel {
    fn from(color: Srgba) -> Self {
        Self {
            r: (color.r * 255.) as u8,
            g: (color.g * 255.) as u8,
            b: (color.b * 255.) as u8,
            a: (color.a * 255.) as u8,
        }
    }
}

/// sRGB.
#[derive(Default, Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct Srgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Srgb {
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

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: f32::lerp(self.r, other.r, t),
            g: f32::lerp(self.g, other.g, t),
            b: f32::lerp(self.b, other.b, t),
        }
    }
}

impl From<Srgb> for [f32; 3] {
    fn from(srgba: Srgb) -> Self {
        srgba.to_array()
    }
}

impl From<[f32; 3]> for Srgb {
    fn from([r, g, b]: [f32; 3]) -> Self {
        Self { r, g, b }
    }
}

impl From<Srgb> for Srgba {
    fn from(s: Srgb) -> Self {
        Self::new(s.r, s.g, s.b, 1.0)
    }
}

impl From<Vec3> for Srgb {
    fn from(value: Vec3) -> Self {
        Self {
            r: value.x,
            g: value.y,
            b: value.z,
        }
    }
}

impl From<Srgb> for RgbaPixel {
    fn from(color: Srgb) -> Self {
        Self {
            r: (color.r * 255.) as u8,
            g: (color.g * 255.) as u8,
            b: (color.b * 255.) as u8,
            a: 255,
        }
    }
}
