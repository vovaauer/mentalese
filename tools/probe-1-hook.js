// mentalese probe, step 1: remember every WebRTC connection Discord opens (read-only).
(() => {
  if (window.__mx) return console.log("mentalese probe already installed");
  const mx = window.__mx = { pcs: [], tracks: [] };
  window.RTCPeerConnection = new Proxy(window.RTCPeerConnection, {
    construct(target, args, newTarget) {
      const pc = Reflect.construct(target, args, newTarget);
      mx.pcs.push(pc);
      pc.addEventListener("track", e => mx.tracks.push({
        kind: e.track.kind, track: e.track.id, streams: e.streams.map(s => s.id), mid: e.transceiver?.mid,
      }));
      return pc;
    },
  });
  console.log("%cmentalese probe installed - now leave the call and join it again", "font-size:16px;color:#3ba55c");
})();
