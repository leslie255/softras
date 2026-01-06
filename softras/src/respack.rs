//! Handles packing/reading of ResPack format.

// FIXME: convert paths to UNIX style path on windows.

use std::{
    collections::BTreeMap,
    fmt::Debug,
    fs,
    io::{self, Cursor, Read, Seek as _},
    mem::transmute,
    path::{Path, PathBuf},
};

use derive_more::{Display, From};

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
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, RespackReadError> {
        let bytes = fs::read(path)?;
        Self::from_vec(bytes)
    }

    pub fn from_vec(bytes: Vec<u8>) -> Result<Self, RespackReadError> {
        let mut cursor = Cursor::new(&bytes[..]);
        let file_magic_number = cursor.read_array::<8>();
        if !file_magic_number.is_ok_and(|bytes| bytes == MAGIC_NUMBER) {
            return Err(RespackReadError::InvalidFileMagicNumber);
        }
        let mut items: BTreeMap<&[u8], &[u8]> = BTreeMap::new();
        loop {
            match cursor.read_array::<4>() {
                Ok(ITEM) => (),
                Ok(_) => return Err(RespackReadError::InvalidChunkType),
                Err(_) => break,
            }
            let path: &[u8] = {
                let length = u32::from_be_bytes(cursor.read_array::<4>()?);
                let position = cursor.position() as usize;
                cursor.seek(io::SeekFrom::Current(length as i64))?;
                &bytes[position..position + length as usize]
            };
            let content: &[u8] = {
                let length = u32::from_be_bytes(cursor.read_array::<4>()?);
                let position = cursor.position() as usize;
                cursor.seek(io::SeekFrom::Current(length as i64))?;
                &bytes[position..position + length as usize]
            };
            items.insert(path, content);
        }
        Ok(Self {
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

#[derive(Debug, From, Display)]
pub enum RespackReadError {
    #[display("{_0}")]
    IoError(io::Error),
    #[display("invalid file magic number")]
    InvalidFileMagicNumber,
    #[display("invalid chunk type")]
    InvalidChunkType,
}

// === ResPack Packing ===

#[derive(Debug, Clone)]
pub struct ResourcePacker {
    root_path: PathBuf,
    bytes: Vec<u8>,
}

impl ResourcePacker {
    pub fn new(root_path: impl Into<PathBuf>) -> Self {
        Self {
            root_path: root_path.into(),
            bytes: Vec::from(MAGIC_NUMBER),
        }
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

    pub fn append_file(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let path = path.normalize_lexically().unwrap_or(path.into());
        let bytes = fs::read(self.root_path.join(&path))?;
        self.append_bytes(&path.as_os_str().to_string_lossy(), &bytes);
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
