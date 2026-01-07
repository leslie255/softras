//! Handles packing/reading of ResPack format.

use std::{
    collections::BTreeMap,
    fmt::Debug,
    fs,
    io::{self, Cursor, Read, Seek as _},
    mem::transmute,
    path::Path,
};

use derive_more::{Display, Error};
use typed_path::{Utf8UnixPath, Utf8UnixPathBuf};

pub(crate) const MAGIC_NUMBER: [u8; 8] = *b"RES-PACK";

/// Chunk type of an uncompressed item.
pub(crate) const ITEM: [u8; 4] = *b"ITEM";

// === ResPack Reading ===

#[derive(Debug, Clone)]
pub struct ResPack {
    /// Map from paths to content.
    ///
    /// FIXME: unsafeness from using `'static` as a hack for self-referencing?
    items: BTreeMap<&'static [u8], &'static [u8]>,
    _bytes: Box<[u8]>,
}

impl ResPack {
    pub fn from_vec(bytes: Vec<u8>) -> Result<Self, RespackReadError> {
        fn read_array<const N: usize>(
            cursor: &mut Cursor<&[u8]>,
        ) -> Result<[u8; N], RespackReadError> {
            cursor
                .read_array::<N>()
                .map_err(|_| RespackReadError::UnexpectedEof)
        }

        fn read_slice<'a>(
            bytes: &'a [u8],
            cursor: &mut Cursor<&[u8]>,
        ) -> Result<&'a [u8], RespackReadError> {
            let length_position = cursor.position();
            let length = u32::from_be_bytes(read_array::<4>(cursor)?) as usize;
            if length > i32::MAX as usize {
                return Err(RespackReadError::LengthOverflow {
                    position: length_position,
                    value: length as u64,
                });
            }
            let position = cursor.position();
            cursor
                .seek(io::SeekFrom::Current(length as i64))
                .map_err(|_| RespackReadError::UnexpectedEof)?;
            let (start, end) = (position as usize, position as usize + length);
            if end > i32::MAX as usize {
                return Err(RespackReadError::PositionOverflow {
                    position: length_position,
                    value: end as u64,
                });
            }
            bytes
                .get(start..end)
                .ok_or(RespackReadError::IndexOutofRange {
                    position,
                    start,
                    end,
                    total_length: bytes.len(),
                })
        }

        let mut cursor = Cursor::new(&bytes[..]);

        let file_magic_number = read_array(&mut cursor)?;
        if file_magic_number != MAGIC_NUMBER {
            return Err(RespackReadError::InvalidFileMagicNumber);
        }

        let mut items: BTreeMap<&[u8], &[u8]> = BTreeMap::new();
        loop {
            let position = cursor.position();
            match cursor.read_array::<4>() {
                Ok(ITEM) => (),
                Ok(_) => return Err(RespackReadError::InvalidChunkType { position }),
                Err(_) => break,
            }
            let path = read_slice(&bytes, &mut cursor)?;
            let content: &[u8] = read_slice(&bytes, &mut cursor)?;
            items.insert(path, content);
        }
        Ok(Self {
            // Refer to the FIXME in struct declaration.
            items: unsafe {
                transmute::<BTreeMap<&[u8], &[u8]>, BTreeMap<&'static [u8], &'static [u8]>>(items)
            },
            _bytes: bytes.into(),
        })
    }

    pub fn get<'a>(&'a self, path: &str) -> Option<&'a [u8]> {
        self.items.get(path.as_bytes()).copied()
    }
}

#[derive(Debug, Display, Error)]
pub enum RespackReadError {
    #[display("invalid file magic number")]
    InvalidFileMagicNumber,
    #[display("invalid chunk type at position {position}")]
    InvalidChunkType { position: u64 },
    #[display("unexpected EOF")]
    UnexpectedEof,
    #[display(
        "index out of range at position {position}: {start}..{end} (total length: {total_length})"
    )]
    IndexOutofRange {
        position: u64,
        start: usize,
        end: usize,
        total_length: usize,
    },
    #[display(
        "length overflowed at position {position}: {value} (limit is max 32-bit signed integer)"
    )]
    LengthOverflow { position: u64, value: u64 },
    #[display(
        "position overflowed at position {position}: {value} (limit is max 32-bit signed integer)"
    )]
    PositionOverflow { position: u64, value: u64 },
}

// === ResPack Packing ===

#[derive(Debug, Clone)]
pub struct ResourcePacker {
    root_path: Utf8UnixPathBuf,
    bytes: Vec<u8>,
}

impl ResourcePacker {
    pub fn new(root_path: &str) -> Self {
        Self {
            root_path: root_path.into(),
            bytes: Vec::from(MAGIC_NUMBER),
        }
    }

    pub fn root_path(&self) -> &str {
        self.root_path.as_ref()
    }

    // TODO: make generic over `io::Read`.
    pub fn append_bytes(&mut self, path: &str, content: &[u8]) {
        // Chunk type "ITEM".
        self.bytes.extend_from_slice(&ITEM);
        // Path.
        self.bytes
            .extend_from_slice(&(path.len() as u32).to_be_bytes());
        self.bytes.extend_from_slice(path.as_bytes());
        // Content.
        self.bytes
            .extend_from_slice(&(content.len() as u32).to_be_bytes());
        self.bytes.extend_from_slice(content);
    }

    pub fn append_file(&mut self, subpath: &str) -> io::Result<()> {
        let subpath = Utf8UnixPath::new(subpath);
        let path = self.root_path.join(subpath);
        let bytes: Vec<u8> = if cfg!(unix) {
            fs::read(path.as_str())?
        } else if cfg!(windows) {
            let windows_path = path.with_windows_encoding();
            let s: &str = windows_path.as_ref();
            fs::read(s)?
        } else {
            panic!("unsupported OS for resource packing (supported: Unix-like OS, Windows)");
        };
        self.append_bytes(subpath.as_str(), &bytes);
        Ok(())
    }

    pub fn finish(&mut self) -> &[u8] {
        &self.bytes
    }

    pub fn finish_into_file(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = self.finish();
        std::fs::write(path, bytes)
    }
}
