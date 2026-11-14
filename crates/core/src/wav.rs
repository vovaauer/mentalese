//! Minimal WAV reading (PCM 16/24/32-bit or float, any channels, mixed to mono) and writing.

use std::path::Path;

/// (mono samples, sample rate)
pub fn read(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&data).map_err(|e| format!("{}: {e}", path.display()))
}

fn parse(d: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if d.len() < 12 || &d[0..4] != b"RIFF" || &d[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([d[i], d[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]);
    let (mut fmt, mut channels, mut rate, mut bits) = (0u16, 0usize, 0u32, 0u16);
    let mut i = 12;
    while i + 8 <= d.len() {
        let (id, size) = (&d[i..i + 4], u32_at(i + 4) as usize);
        let body = i + 8;
        let end = (body + size).min(d.len());
        if id == b"fmt " && size >= 16 {
            fmt = u16_at(body);
            channels = u16_at(body + 2) as usize;
            rate = u32_at(body + 4);
            bits = u16_at(body + 14);
            if fmt == 0xFFFE && size >= 26 {
                fmt = u16_at(body + 24); // WAVE_FORMAT_EXTENSIBLE: the real format is in the subformat GUID
            }
        } else if id == b"data" {
            if channels == 0 {
                return Err("data before format".into());
            }
            let s = &d[body..end];
            let frames: Vec<f32> = match (fmt, bits) {
                (1, 16) => s.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 / 32768.0).collect(),
                (1, 24) => s.as_chunks::<3>().0.iter().map(|b| (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0).collect(),
                (1, 32) => s.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as f32 / 2_147_483_648.0).collect(),
                (3, 32) => s.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
                _ => return Err(format!("unsupported WAV format {fmt}/{bits}-bit")),
            };
            let mono = frames.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
            return Ok((mono, rate));
        }
        i = body + size + (size & 1);
    }
    Err("no audio in the file".into())
}

/// Writes 32-bit float mono.
pub fn write(path: &Path, samples: &[f32], rate: u32) -> Result<(), String> {
    let n = (samples.len() * 4) as u32;
    let mut d = Vec::with_capacity(44 + n as usize);
    d.extend_from_slice(b"RIFF");
    d.extend_from_slice(&(36 + n).to_le_bytes());
    d.extend_from_slice(b"WAVEfmt ");
    d.extend_from_slice(&16u32.to_le_bytes());
    d.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    d.extend_from_slice(&1u16.to_le_bytes());
    d.extend_from_slice(&rate.to_le_bytes());
    d.extend_from_slice(&(rate * 4).to_le_bytes());
    d.extend_from_slice(&4u16.to_le_bytes());
    d.extend_from_slice(&32u16.to_le_bytes());
    d.extend_from_slice(b"data");
    d.extend_from_slice(&n.to_le_bytes());
    for s in samples {
        d.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, d).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        let p = std::env::temp_dir().join(format!("mentalese-wav-{}.wav", std::process::id()));
        let s: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        super::write(&p, &s, 24_000).unwrap();
        let (r, rate) = super::read(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!((r, rate), (s, 24_000));
    }
}
