// mentalese probe, step 2: for 10 seconds, sample the volume of every incoming audio track,
// then store a summary in window.__mxOut (read-only; nothing is sent anywhere).
(async () => {
  const mx = window.__mx;
  if (!mx) return console.log("run step 1 first, then rejoin the call");
  console.log("%csampling for 10 seconds - have people talk...", "font-size:16px;color:#faa61a");
  const out = { pcs: [], tracks: mx.tracks, audioElements: document.querySelectorAll("audio").length };
  const sample = async pc => {
    const sdp = pc.remoteDescription?.sdp || "";
    const recv = pc.getReceivers().filter(r => r.track.kind === "audio");
    const info = {
      state: pc.connectionState,
      mAudio: (sdp.match(/^m=audio/gm) || []).length,
      ssrcLines: (sdp.match(/^a=ssrc:\d+ /gm) || []).length,
      msid: (sdp.match(/^a=msid:.*$/gm) || []).slice(0, 10),
      receivers: recv.map(r => ({
        track: r.track.id, state: r.track.readyState, muted: r.track.muted, transform: !!r.transform,
        mid: pc.getTransceivers().find(t => t.receiver === r)?.mid,
      })),
      levels: [],
    };
    for (let i = 0; i < 40; i++) {
      info.levels.push(recv.map(r => +(r.getSynchronizationSources()[0]?.audioLevel ?? -1).toFixed(3)));
      await new Promise(res => setTimeout(res, 250));
    }
    return info;
  };
  out.pcs = await Promise.all(mx.pcs.map(sample));
  window.__mxOut = JSON.stringify(out);
  console.log("%cdone - now type:  copy(__mxOut)  and press Enter", "font-size:16px;color:#3ba55c", out);
})();
