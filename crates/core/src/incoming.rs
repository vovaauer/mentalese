//! Each Discord call participant's audio, separately, from the Vesktop plugin (vencord/mentalese)
//! over local UDP.

use std::collections::HashMap;
use std::net::UdpSocket;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::audio::FRAME;

pub const PORT: u16 = 47321;
const MSG_AUDIO: u8 = 1; // + int16 LE PCM, 16 kHz mono
const MSG_OPEN: u8 = 2; // + utf8 Discord user id
const MSG_CLOSE: u8 = 3;

pub enum Msg<'a> {
    Open(u32),
    Frame(u32, &'a [f32; FRAME]),
    Close(u32),
}

/// Listens until `stop` is set, calling `on` for each stream opening, frame and closing.
pub fn listen(stop: Arc<AtomicBool>, mut on: impl FnMut(Msg) + Send + 'static) -> Result<(), String> {
    let sock = UdpSocket::bind(("127.0.0.1", PORT)).map_err(|e| {
        format!("can't listen for the Vesktop plugin on port {PORT} (is another mentalese running?): {e}")
    })?;
    sock.set_read_timeout(Some(Duration::from_millis(200))).map_err(|e| e.to_string())?;
    std::thread::Builder::new()
        .name("incoming".into())
        .spawn(move || {
            let mut pending: HashMap<u32, Vec<f32>> = HashMap::new();
            let mut packet = vec![0u8; 65536];
            while !stop.load(Ordering::Relaxed) {
                let Ok(n) = sock.recv(&mut packet) else { continue };
                if n < 5 {
                    continue;
                }
                let num = u32::from_le_bytes([packet[1], packet[2], packet[3], packet[4]]);
                match packet[0] {
                    MSG_OPEN => {
                        pending.insert(num, Vec::new());
                        on(Msg::Open(num));
                    }
                    MSG_CLOSE => {
                        if pending.remove(&num).is_some() {
                            on(Msg::Close(num));
                        }
                    }
                    MSG_AUDIO => {
                        let buf = pending.entry(num).or_insert_with(|| {
                            // mentalese started mid-call: adopt the stream
                            on(Msg::Open(num));
                            Vec::new()
                        });
                        buf.extend(packet[5..n].as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 / 32768.0));
                        let whole = buf.len() / FRAME * FRAME;
                        for frame in buf[..whole].as_chunks::<FRAME>().0 {
                            on(Msg::Frame(num, frame));
                        }
                        buf.drain(..whole);
                    }
                    _ => {}
                }
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}
