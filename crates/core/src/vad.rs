//! Voice activity detection: a port of Silero VAD v6 (MIT, snakers4/silero-vad), 16 kHz only.
//!
//! The weights are read straight from the official `silero_vad.onnx` by a minimal protobuf
//! reader, so there is no conversion step. Per 512-sample frame the model sees 576 samples (64 of
//! context from the previous frame), takes a 4-frame STFT magnitude (256-point Hann window,
//! hop 128, right reflect padding), runs four conv1d+ReLU blocks, one LSTM step, and a linear
//! layer + sigmoid. Probabilities match onnxruntime to within a few 1e-6.

use std::{path::Path, sync::Arc};

pub const SR: usize = 16_000;
pub const WINDOW: usize = 512;

const CONTEXT: usize = 64;
const PAD: usize = 64;
const INPUT: usize = CONTEXT + WINDOW + PAD;
const NFFT: usize = 256;
const HOP: usize = 128;
const FRAMES: usize = (INPUT - NFFT) / HOP + 1;
const BINS: usize = NFFT / 2 + 1;
const HIDDEN: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Start,
    End,
}

/// The 16 kHz weights, relaid out for the forward pass. Immutable, so one `Arc<Model>` serves
/// every [`Vad`] on any thread.
pub struct Model {
    /// STFT basis with the real and imaginary rows of each bin interleaved: (2 * BINS, NFFT).
    basis: Vec<f32>,
    /// Conv weights as (out, tap, in), so a tap window over a time-major input is one dot product.
    enc0: Vec<f32>,
    enc1: Vec<f32>,
    /// Taps 1 and 2 only: tap 0 always lands on the zero padding.
    enc2: Vec<f32>,
    /// Centre tap only: the input is a single time step.
    enc3: Vec<f32>,
    bias: [Vec<f32>; 4],
    /// [W_ih | W_hh] per gate row (PyTorch order i, f, g, o), against the input [x | h].
    lstm: Vec<f32>,
    /// b_ih + b_hh.
    lstm_bias: Vec<f32>,
    out_w: Vec<f32>,
    out_b: f32,
}

impl Model {
    pub fn load(path: &Path) -> Result<Model, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Model::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn parse(bytes: &[u8]) -> Result<Model, String> {
        let mut tensors = Vec::new();
        fields(bytes, |field, v| match (field, v) {
            (7, Value::Bytes(g)) => graph(g, &mut tensors),
            _ => Ok(()),
        })?;

        // Both sample-rate branches of the top-level If carry the same weight names, each with
        // its own prefix; the 16 kHz one is recognised by the shape of its STFT basis.
        const BASIS: &str = "stft.forward_basis_buffer";
        let stft = tensors
            .iter()
            .find(|t| t.name.ends_with(BASIS) && t.dims == [2 * BINS, 1, NFFT])
            .ok_or("no 16 kHz Silero VAD branch")?;
        let prefix = &stft.name[..stft.name.len() - BASIS.len()];
        let get = |key: &str, dims: &[usize]| -> Result<&[f32], String> {
            let name = format!("{prefix}{key}");
            let t = tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("missing {name}"))?;
            if t.dims != dims || t.data.len() != dims.iter().product::<usize>() {
                return Err(format!("{name}: shape {:?}, expected {dims:?}", t.dims));
            }
            Ok(&t.data)
        };

        let b = &stft.data;
        let basis = (0..BINS).flat_map(|f| [&b[f * NFFT..][..NFFT], &b[(BINS + f) * NFFT..][..NFFT]]);
        let conv = |i: usize, out: usize, inp: usize, taps: std::ops::Range<usize>| {
            let w = get(&format!("encoder.{i}.reparam_conv.weight"), &[out, inp, 3])?;
            let b = get(&format!("encoder.{i}.reparam_conv.bias"), &[out])?;
            let w = (0..out)
                .flat_map(|o| taps.clone().flat_map(move |k| (0..inp).map(move |c| w[(o * inp + c) * 3 + k])))
                .collect();
            Ok::<_, String>((w, b.to_vec()))
        };
        let (enc0, b0) = conv(0, 128, BINS, 0..3)?;
        let (enc1, b1) = conv(1, 64, 128, 0..3)?;
        let (enc2, b2) = conv(2, 64, 64, 1..3)?;
        let (enc3, b3) = conv(3, HIDDEN, 64, 1..2)?;
        let [w_ih, w_hh] = ["weight_ih", "weight_hh"].map(|k| get(&format!("decoder.rnn.{k}"), &[4 * HIDDEN, HIDDEN]));
        let [b_ih, b_hh] = ["bias_ih", "bias_hh"].map(|k| get(&format!("decoder.rnn.{k}"), &[4 * HIDDEN]));
        let (w_ih, w_hh, b_ih, b_hh) = (w_ih?, w_hh?, b_ih?, b_hh?);

        Ok(Model {
            basis: basis.flatten().copied().collect(),
            enc0,
            enc1,
            enc2,
            enc3,
            bias: [b0, b1, b2, b3],
            lstm: w_ih.as_chunks::<HIDDEN>().0.iter().zip(w_hh.as_chunks().0).flat_map(|(a, b)| [a, b]).flatten().copied()
                .collect(),
            lstm_bias: b_ih.iter().zip(b_hh).map(|(a, b)| a + b).collect(),
            out_w: get("decoder.decoder.2.weight", &[1, HIDDEN, 1])?.to_vec(),
            out_b: get("decoder.decoder.2.bias", &[1])?[0],
        })
    }
}

/// Per-stream state: the recurrent state and context, preallocated scratch for the forward pass,
/// and the start/end iterator (silero's VADIterator).
pub struct Vad {
    model: Arc<Model>,
    threshold: f32,
    min_silence: u64,
    triggered: bool,
    temp_end: u64,
    current: u64,
    s: Scratch,
}

/// Activations are time-major; conv inputs keep a zero row before and after for the padding.
struct Scratch {
    /// [context | frame | reflect padding]
    input: [f32; INPUT],
    mag: [f32; (FRAMES + 2) * BINS],
    enc0: [f32; (FRAMES + 2) * 128],
    enc1: [f32; 2 * 64],
    enc2: [f32; 64],
    /// LSTM input then hidden state: [x | h]
    xh: [f32; 2 * HIDDEN],
    c: [f32; HIDDEN],
    gates: [f32; 4 * HIDDEN],
}

impl Vad {
    /// Speech starts at a probability of `threshold` and ends after `min_silence_ms` below
    /// `threshold - 0.15`.
    pub fn new(model: Arc<Model>, threshold: f32, min_silence_ms: u32) -> Vad {
        Vad {
            model,
            threshold,
            min_silence: SR as u64 * min_silence_ms as u64 / 1000,
            triggered: false,
            temp_end: 0,
            current: 0,
            s: Scratch {
                input: [0.; INPUT],
                mag: [0.; (FRAMES + 2) * BINS],
                enc0: [0.; (FRAMES + 2) * 128],
                enc1: [0.; 2 * 64],
                enc2: [0.; 64],
                xh: [0.; 2 * HIDDEN],
                c: [0.; HIDDEN],
                gates: [0.; 4 * HIDDEN],
            },
        }
    }

    pub fn reset(&mut self) {
        self.s.input[..CONTEXT].fill(0.);
        self.s.xh[HIDDEN..].fill(0.);
        self.s.c.fill(0.);
        (self.triggered, self.temp_end, self.current) = (false, 0, 0);
    }

    /// Speech probability of the next `WINDOW` samples at 16 kHz. Panics on any other length.
    pub fn prob(&mut self, frame: &[f32]) -> f32 {
        let s = &mut self.s;
        s.input[CONTEXT..][..WINDOW].copy_from_slice(frame);
        for i in 0..PAD {
            s.input[CONTEXT + WINDOW + i] = s.input[CONTEXT + WINDOW - 2 - i];
        }
        let p = forward(&self.model, s);
        s.input.copy_within(WINDOW..WINDOW + CONTEXT, 0);
        p
    }

    /// Like [`Vad::prob`], but reports where speech starts and ends.
    pub fn feed(&mut self, frame: &[f32]) -> Option<Event> {
        self.current += WINDOW as u64;
        let p = self.prob(frame);
        if p >= self.threshold {
            self.temp_end = 0;
            if !self.triggered {
                self.triggered = true;
                return Some(Event::Start);
            }
        } else if p < self.threshold - 0.15 && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current;
            }
            if self.current - self.temp_end >= self.min_silence {
                (self.temp_end, self.triggered) = (0, false);
                return Some(Event::End);
            }
        }
        None
    }
}

fn forward(m: &Model, s: &mut Scratch) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected (std caches the check).
        return unsafe { forward_avx2(m, s) };
    }
    // aarch64 always has FMA; elsewhere `f32::mul_add` may be a slow libm call.
    run::<{ cfg!(target_arch = "aarch64") }>(m, s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn forward_avx2(m: &Model, s: &mut Scratch) -> f32 {
    run::<true>(m, s)
}

#[inline(always)]
fn run<const FMA: bool>(m: &Model, s: &mut Scratch) -> f32 {
    matmul::<2, FRAMES, FMA>(&m.basis, NFFT, &s.input, HOP, |row, [re, im]| {
        for t in 0..FRAMES {
            s.mag[(t + 1) * BINS + row / 2] = (re[t] * re[t] + im[t] * im[t]).sqrt();
        }
    });
    matmul::<2, FRAMES, FMA>(&m.enc0, 3 * BINS, &s.mag, BINS, |row, y| {
        relu_into(&mut s.enc0[128..], 128, &m.bias[0], row, y)
    });
    matmul::<4, 2, FMA>(&m.enc1, 3 * 128, &s.enc0, 2 * 128, |row, y| relu_into(&mut s.enc1, 64, &m.bias[1], row, y));
    matmul::<8, 1, FMA>(&m.enc2, 2 * 64, &s.enc1, 0, |row, y| relu_into(&mut s.enc2, 0, &m.bias[2], row, y));
    matmul::<8, 1, FMA>(&m.enc3, 64, &s.enc2, 0, |row, y| relu_into(&mut s.xh, 0, &m.bias[3], row, y));
    matmul::<8, 1, FMA>(&m.lstm, 2 * HIDDEN, &s.xh, 0, |row, y| {
        for (r, [y]) in y.iter().enumerate() {
            s.gates[row + r] = y + m.lstm_bias[row + r];
        }
    });
    let (g, h) = (&s.gates, &mut s.xh[HIDDEN..]);
    for j in 0..HIDDEN {
        let (i, f, c, o) = (sigmoid(g[j]), sigmoid(g[HIDDEN + j]), tanh(g[2 * HIDDEN + j]), sigmoid(g[3 * HIDDEN + j]));
        s.c[j] = f * s.c[j] + i * c;
        h[j] = o * tanh(s.c[j]);
    }
    let h: [f32; HIDDEN] = std::array::from_fn(|j| h[j].max(0.));
    sigmoid(m.out_b + dot::<FMA>(&h, &m.out_w))
}

/// Bias + ReLU of a matmul block into a time-major buffer of `width` channels per step.
#[inline(always)]
fn relu_into<const R: usize, const N: usize>(out: &mut [f32], width: usize, bias: &[f32], row: usize, y: [[f32; N]; R]) {
    for (r, y) in y.iter().enumerate() {
        for (t, y) in y.iter().enumerate() {
            out[t * width + row + r] = (y + bias[row + r]).max(0.);
        }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1. / (1. + exp(-x))
}

#[inline(always)]
fn tanh(x: f32) -> f32 {
    2. * sigmoid(2. * x) - 1.
}

/// e^x within a few ulp, in plain arithmetic so that loops calling it vectorize (libm's don't).
#[inline(always)]
fn exp(x: f32) -> f32 {
    // Cephes expf: e^x = 2^n e^r with |r| <= ln2/2, n rounded by the 1.5 * 2^23 trick.
    let x = x.clamp(-87., 88.);
    let n = (x * std::f32::consts::LOG2_E + 12_582_912.) - 12_582_912.;
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = (((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 0.166_666_66;
    let p = p * r + 0.5;
    (p * r * r + r + 1.) * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

/// Calls `out(row, y)` with `y[r][n] = w[row + r] · x[n * x_step..][..k]` for each block of R
/// rows of the row-major `w`.
#[inline(always)]
fn matmul<const R: usize, const N: usize, const FMA: bool>(
    w: &[f32],
    k: usize,
    x: &[f32],
    x_step: usize,
    mut out: impl FnMut(usize, [[f32; N]; R]),
) {
    use std::array::from_fn;
    let xs: [&[f32]; N] = from_fn(|n| &x[n * x_step..][..k]);
    #[cfg(target_arch = "x86_64")]
    if FMA {
        // SAFETY: on x86_64, FMA is only set inside forward_avx2.
        return unsafe { matmul_avx2(w, xs, out) };
    }
    for (i, w) in w.chunks_exact(R * k).enumerate() {
        out(i * R, from_fn(|r| from_fn(|n| dot::<FMA>(&w[r * k..][..k], xs[n]))));
    }
}

/// Written so LLVM vectorizes it: 32 independent lanes and no reassociation needed.
#[inline(always)]
fn dot<const FMA: bool>(a: &[f32], b: &[f32]) -> f32 {
    let madd = |a: f32, b: f32, c: f32| if FMA { a.mul_add(b, c) } else { a * b + c };
    let ((ac, at), (bc, bt)) = (a.as_chunks::<32>(), b.as_chunks::<32>());
    let mut acc = [0f32; 32];
    for (a, b) in ac.iter().zip(bc) {
        for l in 0..32 {
            acc[l] = madd(a[l], b[l], acc[l]);
        }
    }
    at.iter().zip(bt).fold(acc.iter().sum(), |s, (a, b)| madd(*a, *b, s))
}

/// [`matmul`] with each block as one register tile: every weight load feeds N FMAs and every
/// input load R, which LLVM's autovectorizer won't arrange by itself.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn matmul_avx2<const R: usize, const N: usize>(w: &[f32], x: [&[f32]; N], mut out: impl FnMut(usize, [[f32; N]; R])) {
    use std::arch::x86_64::*;
    let k = x[0].len();
    assert!(x.iter().all(|x| x.len() == k));
    for (i, w) in w.chunks_exact(R * k).enumerate() {
        let mut acc = [[_mm256_setzero_ps(); N]; R];
        let mut j = 0;
        while j + 8 <= k {
            // SAFETY: w holds R rows of k, every x is k long, and j + 8 <= k.
            let xv: [__m256; N] = std::array::from_fn(|n| unsafe { _mm256_loadu_ps(x[n].as_ptr().add(j)) });
            for (r, acc) in acc.iter_mut().enumerate() {
                let wv = unsafe { _mm256_loadu_ps(w.as_ptr().add(r * k + j)) };
                for (acc, xv) in acc.iter_mut().zip(xv) {
                    *acc = _mm256_fmadd_ps(wv, xv, *acc);
                }
            }
            j += 8;
        }
        let y = std::array::from_fn(|r| {
            std::array::from_fn(|n| {
                let mut v = [0f32; 8];
                // SAFETY: v holds 8 floats.
                unsafe { _mm256_storeu_ps(v.as_mut_ptr(), acc[r][n]) };
                (j..k).fold(v.iter().sum(), |s, t| w[r * k + t].mul_add(x[n][t], s))
            })
        });
        out(i * R, y);
    }
}

// Minimal protobuf reading: just enough of ONNX's ModelProto to collect every float tensor
// (initializers and Constant node values), including those inside If subgraphs.

enum Value<'a> {
    Int(u64),
    Bytes(&'a [u8]),
}

struct Tensor {
    name: String,
    dims: Vec<usize>,
    data: Vec<f32>,
}

fn varint(buf: &mut &[u8]) -> Result<u64, String> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let (&b, rest) = buf.split_first().ok_or("truncated varint")?;
        *buf = rest;
        v |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            return Ok(v);
        }
    }
    Err("varint too long".into())
}

fn fields<'a>(mut buf: &'a [u8], mut f: impl FnMut(u64, Value<'a>) -> Result<(), String>) -> Result<(), String> {
    let take = |buf: &mut &'a [u8], n: usize| {
        let (v, rest) = buf.split_at_checked(n).ok_or("truncated protobuf")?;
        *buf = rest;
        Ok::<_, String>(Value::Bytes(v))
    };
    while !buf.is_empty() {
        let key = varint(&mut buf)?;
        let v = match key & 7 {
            0 => Value::Int(varint(&mut buf)?),
            1 => take(&mut buf, 8)?,
            2 => {
                let n = varint(&mut buf)?;
                take(&mut buf, usize::try_from(n).map_err(|_| "bad length")?)?
            }
            5 => take(&mut buf, 4)?,
            w => return Err(format!("unsupported protobuf wire type {w}")),
        };
        f(key >> 3, v)?;
    }
    Ok(())
}

fn utf8(b: &[u8]) -> Result<&str, String> {
    std::str::from_utf8(b).map_err(|e| e.to_string())
}

fn graph(buf: &[u8], out: &mut Vec<Tensor>) -> Result<(), String> {
    fields(buf, |field, v| match (field, v) {
        (1, Value::Bytes(node)) => self::node(node, out),
        (5, Value::Bytes(t)) => {
            out.extend(tensor(t, None)?);
            Ok(())
        }
        _ => Ok(()),
    })
}

fn node(buf: &[u8], out: &mut Vec<Tensor>) -> Result<(), String> {
    let (mut output, mut op, mut attrs) = (None, "", Vec::new());
    fields(buf, |field, v| {
        match (field, v) {
            (2, Value::Bytes(b)) if output.is_none() => output = Some(utf8(b)?),
            (4, Value::Bytes(b)) => op = utf8(b)?,
            (5, Value::Bytes(b)) => attrs.push(b),
            _ => {}
        }
        Ok(())
    })?;
    for attr in attrs {
        let (mut name, mut value) = ("", None);
        fields(attr, |field, v| match (field, v) {
            (1, Value::Bytes(b)) => {
                name = utf8(b)?;
                Ok(())
            }
            (5, Value::Bytes(t)) => {
                value = Some(t);
                Ok(())
            }
            (6 | 11, Value::Bytes(g)) => graph(g, out),
            _ => Ok(()),
        })?;
        if let (Some(t), "Constant", "value") = (value, op, name) {
            out.extend(tensor(t, output)?);
        }
    }
    Ok(())
}

/// A float tensor, named `name` or by its own name field; other types are skipped.
fn tensor(buf: &[u8], name: Option<&str>) -> Result<Option<Tensor>, String> {
    let (mut dims, mut dtype, mut own_name, mut data, mut external) = (Vec::new(), 0, "", Vec::new(), false);
    fields(buf, |field, v| {
        match (field, v) {
            (1, Value::Int(d)) => dims.push(d as usize),
            (1, Value::Bytes(mut b)) => {
                while !b.is_empty() {
                    dims.push(varint(&mut b)? as usize);
                }
            }
            (2, Value::Int(t)) => dtype = t,
            // float_data (packed or not) and raw_data are both little-endian f32s.
            (4 | 9, Value::Bytes(b)) => data.extend(b.as_chunks().0.iter().map(|c| f32::from_le_bytes(*c))),
            (8, Value::Bytes(b)) => own_name = utf8(b)?,
            (14, Value::Int(loc)) => external = loc == 1,
            _ => {}
        }
        Ok(())
    })?;
    if dtype != 1 {
        return Ok(None);
    }
    let name = name.unwrap_or(own_name).to_string();
    if external {
        return Err(format!("{name}: external tensor data is not supported"));
    }
    Ok(Some(Tensor { name, dims, data }))
}
