//! Sample-rate conversion: polyphase windowed sinc (Kaiser) for any integer rate pair, usable
//! streaming (live audio) or all at once (files).

pub struct Resampler {
    up: usize,
    down: usize,
    /// taps on each side of the centre
    half: usize,
    /// up phases x 2*half taps
    table: Vec<f32>,
    buf: Vec<f32>,
    /// next output sample's position in the input, in 1/up input samples, relative to buf[0]
    next: usize,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Modified Bessel function I0, for the Kaiser window.
fn i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > 1e-12 * sum {
        term *= (x / (2.0 * k)).powi(2);
        sum += term;
        k += 1.0;
    }
    sum
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Resampler {
        let g = gcd(from as usize, to as usize);
        let (up, down) = (to as usize / g, from as usize / g);
        // cutoff relative to the input's Nyquist; below 1 when downsampling, with a transition band
        let ratio = (up as f64 / down as f64).min(1.0);
        let cutoff = 0.92 * ratio;
        let half = (16.0 / ratio).ceil() as usize;
        let beta = 8.6;
        let mut table = vec![0f32; up * 2 * half];
        for p in 0..up {
            let row = &mut table[p * 2 * half..(p + 1) * 2 * half];
            let mut sum = 0.0;
            for (k, t) in row.iter_mut().enumerate() {
                // distance from the output position (n + p/up) to input tap n + k - half + 1
                let d = (k as f64 - half as f64 + 1.0) - p as f64 / up as f64;
                let x = cutoff * d;
                let sinc = if x.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                let w = d / half as f64;
                let win = if w.abs() >= 1.0 { 0.0 } else { i0(beta * (1.0 - w * w).sqrt()) / i0(beta) };
                *t = (sinc * win) as f32;
                sum += *t as f64;
            }
            for t in row.iter_mut() {
                *t = (*t as f64 / sum) as f32;
            }
        }
        Resampler { up, down, half, table, buf: vec![0.0; half - 1], next: (half - 1) * up }
    }

    /// Feeds input and appends every output sample it can produce so far.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.up == self.down {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        let taps = 2 * self.half;
        loop {
            let (n, p) = (self.next / self.up, self.next % self.up);
            if n + self.half >= self.buf.len() {
                break;
            }
            let x = &self.buf[n + 1 - self.half..n + 1 + self.half];
            let h = &self.table[p * taps..(p + 1) * taps];
            out.push(x.iter().zip(h).map(|(a, b)| a * b).sum());
            self.next += self.down;
        }
        // drop input no future output needs
        let keep_from = (self.next / self.up + 1).saturating_sub(self.half);
        if keep_from > 0 {
            self.buf.drain(..keep_from);
            self.next -= keep_from * self.up;
        }
    }
}

/// Converts a whole buffer.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return input.to_vec();
    }
    let mut r = Resampler::new(from, to);
    let want = (input.len() as u64 * to as u64 / from as u64) as usize;
    let mut out = Vec::with_capacity(want + 16);
    r.process(input, &mut out);
    r.process(&vec![0.0; r.half * 2], &mut out);
    out.truncate(want);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, rate: u32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin()).collect()
    }

    #[test]
    fn keeps_a_tone_and_its_timing() {
        for (from, to) in [(24_000, 16_000), (48_000, 16_000), (16_000, 48_000), (44_100, 16_000), (24_000, 48_000)] {
            let out = resample(&tone(440.0, from, 1.0), from, to);
            assert_eq!(out.len(), to as usize);
            let want = tone(440.0, to, 1.0);
            // skip the edges, where the filter sees the zero padding
            let err = out[200..out.len() - 200].iter().zip(&want[200..]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(err < 0.01, "{from}->{to}: max error {err}");
        }
    }

    #[test]
    fn removes_what_the_new_rate_cant_hold() {
        // 10 kHz is above 16 kHz's Nyquist (8 kHz)
        let out = resample(&tone(10_000.0, 48_000, 1.0), 48_000, 16_000);
        let rms = (out[500..out.len() - 500].iter().map(|x| x * x).sum::<f32>() / (out.len() - 1000) as f32).sqrt();
        assert!(rms < 0.01, "aliased energy {rms}");
    }

    #[test]
    fn streaming_matches_whole() {
        let x = tone(300.0, 24_000, 0.5);
        let whole = resample(&x, 24_000, 16_000);
        let mut r = Resampler::new(24_000, 16_000);
        let mut out = Vec::new();
        for c in x.chunks(333) {
            r.process(c, &mut out);
        }
        assert_eq!(&out[..], &whole[..out.len()]);
    }
}
