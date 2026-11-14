// Mentalese's window. Talks to the Rust side through Tauri's invoke() and events.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const COLORS = ["#06b6d4", "#f59e0b", "#d946ef", "#22c55e", "#3b82f6", "#ef4444"];
const DEFAULT_STYLE = "casual and informal, like talking to a close friend; translate only what is said, add nothing";

let ov = null; // overview from the Rust side
let phase = "stopped"; // stopped | loading | live
let muted = false;

function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), 4000);
}

async function call(cmd, args) {
  try {
    return await invoke(cmd, args);
  } catch (e) {
    toast(String(e));
    throw e;
  }
}

function options(select, items, value) {
  select.replaceChildren(...items.map(([v, label]) => new Option(label, v)));
  select.value = value ?? "";
}

function langName(code) {
  return (ov.langs.find(([c]) => c === code) || [code, code])[1];
}

async function refresh() {
  ov = await invoke("overview");
  const s = ov.settings;
  $("setupModels").hidden = ov.missing_mb === 0;
  $("missingSize").textContent = ov.missing_mb >= 1024 ? `${(ov.missing_mb / 1024).toFixed(1)} GB` : `${ov.missing_mb} MB`;
  $("setupVoice").hidden = ov.missing_mb > 0 || ov.voices.length > 0;
  $("live").hidden = ov.missing_mb > 0 || ov.voices.length === 0;

  const langs = ov.langs.map(([c, n]) => [c, n]);
  options($("to"), langs, s.to);
  options($("me"), langs, s.me);
  if (!ov.voices.includes(s.voice) && ov.voices.length) s.voice = ov.voices[0];
  options($("voice"), ov.voices.map((v) => [v, v]), s.voice);

  options($("s-mic"), [["", "System default"], ...ov.inputs.map((d) => [d.id, d.name])], s.mic ?? "");
  options($("s-headphones"), [["", "System default"], ...ov.outputs.map((d) => [d.id, d.name])], s.headphones ?? "");
  fillApps(ov.apps);
  $("s-monitor").checked = s.monitor;
  $("s-practice").checked = s.practice;
  $("s-plugin").checked = s.subtitles_plugin;
  $("s-accent").checked = s.keep_accent;
  $("s-style").value = s.style;
  $("s-silence").value = s.silence_ms;
  $("s-their").value = s.their_silence_ms;
  $("s-asrgpu").checked = s.asr_gpu;
  $("dataDir").textContent = ov.data_dir;
  showRanges();
  renderVoices();
}

function fillApps(apps) {
  const cur = ov.settings.subtitles_from;
  const list = [...new Set([...(cur ? [cur] : []), ...apps])];
  options($("s-app"), [["", "None"], ...list.map((a) => [a, a])], cur ?? "");
}

function showRanges() {
  $("o-silence").textContent = `${$("s-silence").value} ms`;
  $("o-their").textContent = `${$("s-their").value} ms`;
}

function renderVoices() {
  const list = $("voiceList");
  list.replaceChildren(...ov.voices.map((name) => {
    const row = document.createElement("div");
    row.className = "voice";
    const label = document.createElement("span");
    label.textContent = name;
    const play = document.createElement("button");
    play.type = "button";
    play.className = "ghost small";
    play.textContent = "Hear it";
    play.onclick = async () => {
      play.disabled = true;
      toast(`Speaking ${langName(ov.settings.to)} as "${name}"...`);
      try { await call("preview_voice", { name, lang: ov.settings.to }); } finally { play.disabled = false; }
    };
    const del = document.createElement("button");
    del.type = "button";
    del.className = "ghost small";
    del.textContent = "Delete";
    del.onclick = async () => {
      if (!confirm(`Delete the voice "${name}"? This can't be undone.`)) return;
      await call("delete_voice", { name });
      await refresh();
    };
    row.append(label, play, del);
    return row;
  }));
}

async function save() {
  const s = ov.settings;
  s.to = $("to").value;
  s.me = $("me").value;
  s.voice = $("voice").value;
  s.mic = $("s-mic").value || null;
  s.headphones = $("s-headphones").value || null;
  s.monitor = $("s-monitor").checked;
  s.practice = $("s-practice").checked;
  s.subtitles_plugin = $("s-plugin").checked;
  s.subtitles_from = $("s-app").value || null;
  s.keep_accent = $("s-accent").checked;
  s.style = $("s-style").value.trim();
  s.silence_ms = Number($("s-silence").value);
  s.their_silence_ms = Number($("s-their").value);
  s.asr_gpu = $("s-asrgpu").checked;
  showRanges();
  await call("save_settings", { settings: s });
}

function setPhase(p, text) {
  phase = p;
  const pill = $("status");
  pill.className = "pill" + (p === "live" ? " live" : p === "loading" ? " loading" : "");
  pill.textContent = text || { stopped: "Stopped", loading: "Starting...", live: "Live" }[p];
  const start = $("startBtn");
  start.textContent = p === "stopped" ? "Start" : "Stop";
  start.classList.toggle("stop", p !== "stopped");
  $("muteBtn").disabled = p !== "live";
  $("typeInput").disabled = p !== "live";
  $("sayBtn").disabled = p !== "live";
  $("voice").disabled = p !== "stopped";
  $("me").disabled = p !== "stopped";
  if (p === "stopped") {
    $("hint").hidden = true;
    muted = false;
    $("muteBtn").setAttribute("aria-pressed", "false");
    $("muteBtn").textContent = "Mute mic";
  }
}

function feedAdd(el) {
  const feed = $("feed");
  feed.querySelector(".empty")?.remove();
  const stick = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 40;
  feed.append(el);
  while (feed.children.length > 300) feed.firstChild.remove();
  if (stick) feed.scrollTop = feed.scrollHeight;
}

function line(cls, parts) {
  const el = document.createElement("div");
  el.className = `line ${cls}`;
  for (const [k, text] of parts) {
    if (!text) continue;
    const d = document.createElement("div");
    d.className = k;
    d.textContent = text;
    el.append(d);
  }
  return el;
}

function hint(html, warn = false) {
  const h = $("hint");
  h.className = "hint" + (warn ? " warn" : "");
  h.replaceChildren(...html);
  h.hidden = false;
}

function text(t, bold = false) {
  const el = document.createElement(bold ? "b" : "span");
  el.textContent = t;
  return el;
}

function onEvent(e) {
  switch (e.kind) {
    case "status":
      setPhase("loading", `${e.text}...`);
      break;
    case "download":
      setPhase("loading", `Downloading the ${e.what}: ${Math.floor((e.done * 100) / Math.max(e.total, 1))}%`);
      break;
    case "ready": {
      setPhase("live", `Live${e.gpu ? ` on ${e.gpu}` : ""}`);
      const vm = e.virtual_mic;
      if (!vm) {
        hint([text("Practice mode: talk, and hear yourself translated in your headphones.")]);
      } else if (vm.state === "Ready") {
        hint([text("In Discord: User Settings → Voice & Video → Input Device: "), text(vm.name, true),
          text(". Turn off Noise Suppression, Echo Cancellation and Automatic Gain Control there too.")]);
      } else {
        const get = document.createElement("button");
        get.className = "link";
        get.textContent = "Get it";
        get.onclick = () => call("open_url", { url: vm.url });
        hint([text(`Others can't hear you yet: ${vm.how} `), get], true);
      }
      break;
    }
    case "said": {
      const t = e.timing;
      const how = e.typed ? "typed" : `heard ${t.asr_ms} ms`;
      feedAdd(line("mine", [
        ["heard", e.heard],
        ["said", e.said === e.heard ? "" : e.said],
        ["meta", `${langName(e.to)} · ${t.total_ms} ms (${how}, translated ${t.mt_ms} ms, voice ${t.voice_ms} ms)`],
      ]));
      break;
    }
    case "subtitle": {
      const el = line("theirs", [["said", e.text], ["heard", e.original]]);
      el.style.setProperty("--c", COLORS[e.stream % COLORS.length]);
      feedAdd(el);
      break;
    }
    case "notice":
      toast(e.text);
      break;
    case "error":
      feedAdd(line("error", [["said", e.text]]));
      break;
    case "stopped":
      setPhase("stopped");
      break;
  }
}

// recording a voice
let recName = "me";
function openRecorder(name) {
  recName = name;
  $("r-name").value = name || "";
  $("r-name").disabled = name === "me" && ov.voices.length === 0;
  $("r-text").textContent = ov.read_aloud;
  $("r-bar").style.width = "0";
  $("r-msg").textContent = "";
  $("r-go").textContent = "Record";
  $("r-go").disabled = false;
  $("r-go").dataset.state = "ready";
  $("recorder").showModal();
  if (!name) $("r-name").focus();
}

async function record() {
  const go = $("r-go");
  if (go.dataset.state === "done") {
    $("recorder").close();
    return;
  }
  const name = $("r-name").value.trim();
  if (!name) return toast("Give the voice a name");
  go.disabled = true;
  $("r-msg").textContent = "Recording... read the text out loud.";
  try {
    const transcript = await invoke("record_voice", { name });
    $("r-msg").textContent = `Saved. Heard: "${transcript}"`;
    go.textContent = "Done";
    go.dataset.state = "done";
    ov.settings.voice = name;
    await call("save_settings", { settings: ov.settings });
    await refresh();
  } catch (e) {
    $("r-msg").textContent = String(e);
    $("r-bar").style.width = "0";
    go.textContent = "Try again";
  } finally {
    go.disabled = false;
  }
}

async function main() {
  await listen("mx", (ev) => onEvent(ev.payload));
  await listen("download", (ev) => {
    const { what, done, total } = ev.payload;
    const id = `dl-${what.replace(/\W+/g, "-")}`;
    let row = document.getElementById(id);
    if (!row) {
      row = document.createElement("div");
      row.id = id;
      row.className = "dl";
      row.innerHTML = '<div class="label"></div><div class="meter"><div></div></div>';
      $("downloads").append(row);
    }
    row.querySelector(".label").textContent = `${what}: ${done >> 20} / ${total >> 20} MB`;
    row.querySelector(".meter > div").style.width = `${(done * 100) / Math.max(total, 1)}%`;
  });
  await listen("recording", (ev) => {
    const { secs, phase } = ev.payload;
    $("r-bar").style.width = `${Math.min(100, (secs * 100) / ov.record_secs)}%`;
    if (phase === "transcribing") $("r-msg").textContent = "Got it. Checking what you said...";
  });

  $("downloadBtn").onclick = async () => {
    $("downloadBtn").disabled = true;
    try {
      await call("download");
      await refresh();
    } finally {
      $("downloadBtn").disabled = false;
    }
  };
  $("startBtn").onclick = async () => {
    if (phase === "stopped") {
      setPhase("loading");
      try { await call("start"); } catch { setPhase("stopped"); }
    } else {
      $("startBtn").disabled = true;
      try { await call("stop"); } finally { $("startBtn").disabled = false; setPhase("stopped"); }
    }
  };
  $("muteBtn").onclick = async () => {
    muted = !muted;
    $("muteBtn").setAttribute("aria-pressed", String(muted));
    $("muteBtn").textContent = muted ? "Unmute mic" : "Mute mic";
    await call("set_muted", { muted });
  };
  $("typeForm").onsubmit = async (ev) => {
    ev.preventDefault();
    const input = $("typeInput");
    if (!input.value.trim()) return;
    await call("say", { text: input.value });
    input.value = "";
  };
  for (const id of ["to", "me", "voice", "s-mic", "s-headphones", "s-monitor", "s-practice", "s-plugin", "s-app",
    "s-accent", "s-style", "s-silence", "s-their", "s-asrgpu"]) {
    $(id).addEventListener("change", save);
  }
  for (const id of ["s-silence", "s-their"]) $(id).addEventListener("input", showRanges);
  $("resetStyle").onclick = () => { $("s-style").value = DEFAULT_STYLE; save(); };
  $("refreshApps").onclick = async () => fillApps(await call("apps_playing"));
  $("openSettings").onclick = async () => { await refresh(); $("settings").showModal(); };
  document.addEventListener("click", (ev) => {
    const rec = ev.target.closest("[data-record]");
    if (rec) {
      $("settings").close();
      openRecorder(rec.dataset.record);
    }
    const link = ev.target.closest("[data-url]");
    if (link) call("open_url", { url: link.dataset.url });
  });
  $("r-go").onclick = record;
  $("r-cancel").onclick = async () => {
    await invoke("cancel_recording");
    $("recorder").close();
  };

  await refresh();
  setPhase(ov.running ? "live" : "stopped");
  checkUpdate();
}

async function checkUpdate() {
  let info;
  try { info = await invoke("check_update"); } catch { return; } // offline, or no release yet
  if (!info) return;
  $("updateText").textContent = `Mentalese ${info.version} is out.`;
  $("update").hidden = false;
  $("updateBtn").onclick = async () => {
    $("updateBtn").disabled = true;
    $("updateBtn").textContent = "Updating...";
    try { await call("install_update"); } catch { $("updateBtn").disabled = false; $("updateBtn").textContent = "Update and restart"; }
  };
}

main();
