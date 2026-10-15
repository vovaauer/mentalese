/*
 * mentalese: sends each call participant's decrypted voice, separately, to the local
 * mentalese app for subtitles. Read-only: it never changes what Discord plays or sends.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import definePlugin, { PluginNative } from "@utils/types";

const Native = VencordNative.pluginHelpers.Mentalese as PluginNative<typeof import("./native")>;

// UDP message types (byte 0), followed by a uint32 LE stream number.
const MSG_AUDIO = 1; // + int16 LE PCM, 16 kHz mono
const MSG_OPEN = 2;  // + utf8 Discord user id
const MSG_CLOSE = 3;

const SAMPLE_RATE = 16000;
const CHUNK = 1024; // 64 ms per packet

interface Tap {
    num: number;
    source: MediaStreamAudioSourceNode;
    processor: ScriptProcessorNode;
    sink: HTMLAudioElement;
}

let OriginalPC: typeof RTCPeerConnection | null = null;
let ctx: AudioContext | null = null;
let nextNum = 1;
const taps = new Map<string, Tap>();

function header(type: number, num: number, payloadBytes: number) {
    const buf = new Uint8Array(5 + payloadBytes);
    buf[0] = type;
    new DataView(buf.buffer).setUint32(1, num, true);
    return buf;
}

function sendText(type: number, num: number, text = "") {
    const bytes = new TextEncoder().encode(text);
    const buf = header(type, num, bytes.length);
    buf.set(bytes, 5);
    Native.send(buf);
}

function tap(track: MediaStreamTrack, stream: MediaStream) {
    const key = `${track.id}/${stream.id}`;
    if (taps.has(key)) return;
    ctx ??= new AudioContext({ sampleRate: SAMPLE_RATE });
    const num = nextNum++;
    const own = new MediaStream([track]);

    // Chromium only feeds remote WebRTC audio into WebAudio while a media element consumes it.
    const sink = new Audio();
    sink.muted = true;
    sink.srcObject = own;

    const source = ctx.createMediaStreamSource(own);
    const processor = ctx.createScriptProcessor(CHUNK, 1, 1);
    processor.onaudioprocess = e => {
        const input = e.inputBuffer.getChannelData(0);
        const buf = header(MSG_AUDIO, num, input.length * 2);
        const view = new DataView(buf.buffer);
        for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]));
            view.setInt16(5 + i * 2, s * 32767, true);
        }
        Native.send(buf);
    };
    source.connect(processor);
    processor.connect(ctx.destination); // outputs silence; required for onaudioprocess to fire

    taps.set(key, { num, source, processor, sink });
    sendText(MSG_OPEN, num, stream.id.split("-")[0]);

    const stop = () => untap(key);
    track.addEventListener("ended", stop, { once: true });
    stream.addEventListener("removetrack", stop, { once: true });
}

function untap(key: string) {
    const t = taps.get(key);
    if (!t) return;
    taps.delete(key);
    t.processor.disconnect();
    t.source.disconnect();
    t.sink.srcObject = null;
    sendText(MSG_CLOSE, t.num);
}

export default definePlugin({
    name: "Mentalese",
    description: "Streams each voice call participant separately to the local mentalese app for live translated subtitles.",
    authors: [{ name: "mentalese", id: 0n }],

    start() {
        OriginalPC = window.RTCPeerConnection;
        window.RTCPeerConnection = new Proxy(OriginalPC, {
            construct(target, args, newTarget) {
                const pc: RTCPeerConnection = Reflect.construct(target, args, newTarget);
                pc.addEventListener("track", e => {
                    if (e.track.kind === "audio" && e.streams[0]) tap(e.track, e.streams[0]);
                });
                return pc;
            },
        });
    },

    stop() {
        if (OriginalPC) window.RTCPeerConnection = OriginalPC;
        for (const key of [...taps.keys()]) untap(key);
        ctx?.close();
        ctx = null;
        Native.close();
    },
});
