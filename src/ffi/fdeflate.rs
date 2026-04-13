//! Implementation for `fdeflate` rust backend.
//!
//! fdeflate is a fast deflate implementation optimized for PNG images.
//! It supports decompression of arbitrary zlib streams but only provides
//! basic compression (stubbed out here).

use std::fmt;

use super::*;
use crate::mem;

pub const MZ_NO_FLUSH: isize = 0;
pub const MZ_PARTIAL_FLUSH: isize = 1;
pub const MZ_SYNC_FLUSH: isize = 2;
pub const MZ_FULL_FLUSH: isize = 3;
pub const MZ_FINISH: isize = 4;
pub const MZ_DEFAULT_WINDOW_BITS: u8 = 15;

// fdeflate doesn't provide detailed error messages.
#[derive(Clone, Default)]
pub struct ErrorMessage;

impl ErrorMessage {
    pub fn get(&self) -> Option<&str> {
        None
    }
}

/// A minimal valid zlib header (CM=8, CINFO=7, no dict, FCHECK valid).
/// 0x78 0x01 = deflate with window size 2^15, compression level 0.
const ZLIB_HEADER: [u8; 2] = [0x78, 0x01];

pub struct Inflate {
    inner: ::fdeflate::Decompressor,
    total_in: u64,
    total_out: u64,
    zlib_header: bool,
    /// When operating in raw deflate mode (`zlib_header: false`), we need to
    /// inject a synthetic zlib header before the raw data and a fake adler32
    /// checksum after it. This enum tracks the current injection phase.
    raw_state: RawDeflateState,
}

/// Tracks the synthetic header/trailer injection for raw deflate mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawDeflateState {
    /// Not in raw deflate mode (zlib header is present in input).
    NotRaw,
    /// Need to feed synthetic zlib header to the decompressor.
    NeedHeader,
    /// Header has been injected, processing raw deflate data normally.
    Data,
    /// The deflate data is done; need to feed a synthetic adler32 checksum.
    NeedTrailer,
    /// Completely done.
    Done,
}

impl fmt::Debug for Inflate {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "fdeflate inflate internal state. total_in: {}, total_out: {}",
            self.total_in, self.total_out,
        )
    }
}

impl InflateBackend for Inflate {
    fn make(zlib_header: bool, _window_bits: u8) -> Self {
        let mut inner = ::fdeflate::Decompressor::new();
        let raw_state = if zlib_header {
            RawDeflateState::NotRaw
        } else {
            // In raw deflate mode, we disable adler32 checking since the
            // checksum we feed is fake.
            inner.ignore_adler32();
            RawDeflateState::NeedHeader
        };

        Inflate {
            inner,
            total_in: 0,
            total_out: 0,
            zlib_header,
            raw_state,
        }
    }

    fn decompress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: FlushDecompress,
    ) -> Result<Status, DecompressError> {
        let end_of_input = matches!(flush, FlushDecompress::Finish);

        match self.raw_state {
            RawDeflateState::NotRaw => {
                // Zlib-wrapped stream: pass directly to fdeflate.
                self.decompress_zlib(input, output, end_of_input)
            }
            RawDeflateState::NeedHeader => {
                // Feed the synthetic zlib header first, then the actual input.
                let (consumed, produced) = self
                    .inner
                    .read(&ZLIB_HEADER, output, 0, false)
                    .map_err(|_| {
                        mem::decompress_failed::<Status>(ErrorMessage).unwrap_err()
                    })?;
                self.total_out += produced as u64;
                debug_assert_eq!(consumed, 2);
                debug_assert_eq!(produced, 0);
                self.raw_state = RawDeflateState::Data;
                // Now process the actual input data.
                self.decompress_raw(input, output, end_of_input)
            }
            RawDeflateState::Data => self.decompress_raw(input, output, end_of_input),
            RawDeflateState::NeedTrailer => {
                // Feed fake adler32 checksum (4 zero bytes, checksum is ignored).
                let fake_checksum = [0u8; 4];
                let (_consumed, produced) = self
                    .inner
                    .read(&fake_checksum, output, 0, true)
                    .map_err(|_| {
                        mem::decompress_failed::<Status>(ErrorMessage).unwrap_err()
                    })?;
                self.total_out += produced as u64;
                self.raw_state = RawDeflateState::Done;
                Ok(Status::StreamEnd)
            }
            RawDeflateState::Done => Ok(Status::StreamEnd),
        }
    }

    fn reset(&mut self, zlib_header: bool) {
        self.inner = ::fdeflate::Decompressor::new();
        self.total_in = 0;
        self.total_out = 0;
        self.zlib_header = zlib_header;
        if zlib_header {
            self.raw_state = RawDeflateState::NotRaw;
        } else {
            self.inner.ignore_adler32();
            self.raw_state = RawDeflateState::NeedHeader;
        }
    }
}

impl Inflate {
    /// Decompress a zlib-wrapped stream (zlib_header: true).
    fn decompress_zlib(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        end_of_input: bool,
    ) -> Result<Status, DecompressError> {
        match self.inner.read(input, output, 0, end_of_input) {
            Ok((consumed, produced)) => {
                self.total_in += consumed as u64;
                self.total_out += produced as u64;

                if self.inner.is_done() {
                    Ok(Status::StreamEnd)
                } else if consumed == 0 && produced == 0 {
                    Ok(Status::BufError)
                } else {
                    Ok(Status::Ok)
                }
            }
            Err(::fdeflate::DecompressionError::InsufficientInput) => {
                // Not enough input yet; this is not a fatal error in streaming mode.
                Ok(Status::BufError)
            }
            Err(_) => mem::decompress_failed(ErrorMessage),
        }
    }

    /// Decompress raw deflate data (no zlib header/trailer in input).
    fn decompress_raw(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        end_of_input: bool,
    ) -> Result<Status, DecompressError> {
        // We tell fdeflate `end_of_input: false` because the raw deflate data
        // doesn't include the adler32 trailer that fdeflate expects. We'll
        // inject that separately once the deflate blocks are done.
        match self.inner.read(input, output, 0, false) {
            Ok((consumed, produced)) => {
                self.total_in += consumed as u64;
                self.total_out += produced as u64;

                // Check if fdeflate's internal state has reached the checksum
                // phase. Since we're in raw mode, the deflate blocks are done
                // and we need to inject a fake trailer.
                if consumed == 0 && produced == 0 && !input.is_empty() {
                    // The decompressor couldn't consume any input and couldn't
                    // produce any output. If there's input available, the
                    // deflate blocks might be done and it's waiting for checksum
                    // bytes. Try injecting the trailer.
                    self.raw_state = RawDeflateState::NeedTrailer;
                    // Feed fake adler32 checksum.
                    let fake_checksum = [0u8; 4];
                    match self.inner.read(&fake_checksum, output, 0, true) {
                        Ok((_consumed_ck, produced_ck)) => {
                            self.total_out += produced_ck as u64;
                            self.raw_state = RawDeflateState::Done;
                            Ok(Status::StreamEnd)
                        }
                        Err(_) => mem::decompress_failed(ErrorMessage),
                    }
                } else if self.inner.is_done() {
                    self.raw_state = RawDeflateState::Done;
                    Ok(Status::StreamEnd)
                } else if consumed == 0 && produced == 0 {
                    Ok(Status::BufError)
                } else {
                    Ok(Status::Ok)
                }
            }
            Err(::fdeflate::DecompressionError::InsufficientInput) => {
                if end_of_input {
                    // End of raw deflate data. The decompressor is waiting for
                    // the adler32 checksum that doesn't exist in raw mode.
                    // Transition to trailer injection.
                    self.raw_state = RawDeflateState::NeedTrailer;
                    let fake_checksum = [0u8; 4];
                    match self.inner.read(&fake_checksum, output, 0, true) {
                        Ok((_consumed_ck, produced_ck)) => {
                            self.total_out += produced_ck as u64;
                            self.raw_state = RawDeflateState::Done;
                            Ok(Status::StreamEnd)
                        }
                        Err(_) => mem::decompress_failed(ErrorMessage),
                    }
                } else {
                    Ok(Status::BufError)
                }
            }
            Err(_) => mem::decompress_failed(ErrorMessage),
        }
    }
}

impl Backend for Inflate {
    #[inline]
    fn total_in(&self) -> u64 {
        self.total_in
    }

    #[inline]
    fn total_out(&self) -> u64 {
        self.total_out
    }
}

pub struct Deflate {
    total_in: u64,
    total_out: u64,
}

impl fmt::Debug for Deflate {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "fdeflate deflate internal state. total_in: {}, total_out: {}",
            self.total_in, self.total_out,
        )
    }
}

impl DeflateBackend for Deflate {
    fn make(_level: Compression, _zlib_header: bool, _window_bits: u8) -> Self {
        Deflate {
            total_in: 0,
            total_out: 0,
        }
    }

    fn compress(
        &mut self,
        _input: &[u8],
        _output: &mut [u8],
        _flush: FlushCompress,
    ) -> Result<Status, CompressError> {
        todo!("fdeflate does not yet support streaming compression")
    }

    fn reset(&mut self) {
        self.total_in = 0;
        self.total_out = 0;
    }
}

impl Backend for Deflate {
    #[inline]
    fn total_in(&self) -> u64 {
        self.total_in
    }

    #[inline]
    fn total_out(&self) -> u64 {
        self.total_out
    }
}
