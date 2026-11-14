//! macOS: other apps' audio through Core Audio process taps (macOS 14.2+): a mono tap on the
//! app's processes, inside a private aggregate device that an IOProc reads.

use std::ffi::{CStr, c_void};
use std::mem::MaybeUninit;
use std::ptr::{NonNull, null};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::AnyClass;
use objc2_core_audio::{
    AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID, AudioDeviceIOProcID, AudioDeviceStart, AudioDeviceStop,
    AudioHardwareCreateAggregateDevice, AudioHardwareCreateProcessTap, AudioHardwareDestroyAggregateDevice,
    AudioHardwareDestroyProcessTap, AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, AudioObjectPropertySelector, CATapDescription, CATapMuteBehavior,
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceNameKey, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey, kAudioHardwarePropertyProcessObjectList,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject,
    kAudioProcessPropertyBundleID, kAudioProcessPropertyIsRunningOutput, kAudioProcessPropertyPID,
    kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey, kAudioTapPropertyFormat,
};
use objc2_core_audio_types::{
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp, kAudioFormatFlagIsFloat, kAudioFormatLinearPCM,
};
use objc2_core_foundation::{CFArray, CFBoolean, CFDictionary, CFRetained, CFString, CFType};
use objc2_foundation::{NSArray, NSNumber, NSString};

use super::cpal_io::{CAPTURE_BASE, Feed, Framer, OnFrame, Watcher};

fn address(selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// A fixed-size property: None unless it's there with exactly the size of T.
fn get<T: Copy>(object: AudioObjectID, selector: AudioObjectPropertySelector) -> Option<T> {
    let addr = address(selector);
    let mut value = MaybeUninit::<T>::uninit();
    let mut size = size_of::<T>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(object, NonNull::from(&addr), 0, null(), NonNull::from(&mut size), NonNull::from(&mut value).cast())
    };
    (status == 0 && size as usize == size_of::<T>()).then(|| unsafe { value.assume_init() })
}

/// A CFString property; Core Audio hands it over retained, so it's released here.
fn get_string(object: AudioObjectID, selector: AudioObjectPropertySelector) -> Option<String> {
    let s = NonNull::new(get::<*mut CFString>(object, selector)?)?;
    Some(unsafe { CFRetained::from_raw(s) }.to_string())
}

/// Every process that is a Core Audio client.
fn process_objects() -> Vec<AudioObjectID> {
    let addr = address(kAudioHardwarePropertyProcessObjectList);
    let system = kAudioObjectSystemObject as AudioObjectID;
    let mut size = 0u32;
    if unsafe { AudioObjectGetPropertyDataSize(system, NonNull::from(&addr), 0, null(), NonNull::from(&mut size)) } != 0 {
        return Vec::new();
    }
    let mut ids = vec![0 as AudioObjectID; size as usize / size_of::<AudioObjectID>()];
    let mut size = size_of_val(&ids[..]) as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(system, NonNull::from(&addr), 0, null(), NonNull::from(&mut size), NonNull::from(&mut ids[..]).cast())
    };
    if status != 0 {
        return Vec::new();
    }
    // the list can shrink between the two calls
    ids.truncate(size as usize / size_of::<AudioObjectID>());
    ids
}

struct Proc {
    object: AudioObjectID,
    exe: String,
    bundle: String,
}

fn exe_name(pid: i32) -> String {
    let mut buf = [0u8; 256];
    let n = unsafe { libc::proc_name(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
    String::from_utf8_lossy(&buf[..n.max(0) as usize]).into_owned()
}

/// Core Audio's client processes, but not mentalese.
fn processes() -> Vec<Proc> {
    let me = std::process::id() as i32;
    process_objects()
        .into_iter()
        .filter_map(|object| {
            let pid = get::<i32>(object, kAudioProcessPropertyPID).filter(|&pid| pid != me)?;
            let bundle = get_string(object, kAudioProcessPropertyBundleID).unwrap_or_default();
            Some(Proc { object, exe: exe_name(pid), bundle })
        })
        .collect()
}

pub(super) fn apps_playing() -> Vec<String> {
    let mut apps: Vec<String> = processes()
        .into_iter()
        .filter(|p| get::<u32>(p.object, kAudioProcessPropertyIsRunningOutput).is_some_and(|r| r != 0))
        .map(|p| {
            let name = if p.exe.is_empty() { p.bundle } else { p.exe };
            // Chromium and Electron apps play from "<App> Helper": name the app
            match name.find(" Helper") {
                Some(i) => name[..i].to_string(),
                None => name,
            }
        })
        .filter(|a| !a.is_empty() && !a.to_lowercase().contains("mentalese"))
        .collect();
    apps.sort();
    apps.dedup();
    apps
}

fn check(status: i32, what: &str) -> Result<(), String> {
    if status == 0 { Ok(()) } else { Err(format!("can't {what} (Core Audio error {status})")) }
}

fn key(k: &CStr) -> CFRetained<CFString> {
    CFString::from_str(&k.to_string_lossy())
}

/// A private aggregate device holding just the tap, which starts with it.
fn aggregate(tap_uid: &str, uid: &str) -> CFRetained<CFDictionary<CFString, CFType>> {
    let yes: &CFType = CFBoolean::new(true);
    let tap = CFDictionary::<CFString, CFType>::from_slices(
        &[&key(kAudioSubTapUIDKey), &key(kAudioSubTapDriftCompensationKey)],
        &[&CFString::from_str(tap_uid), yes],
    );
    let taps = CFArray::from_retained_objects(&[tap]);
    CFDictionary::<CFString, CFType>::from_slices(
        &[
            &key(kAudioAggregateDeviceNameKey),
            &key(kAudioAggregateDeviceUIDKey),
            &key(kAudioAggregateDeviceIsPrivateKey),
            &key(kAudioAggregateDeviceTapAutoStartKey),
            &key(kAudioAggregateDeviceTapListKey),
        ],
        &[&CFString::from_str("mentalese-capture"), &CFString::from_str(uid), yes, yes, taps.as_opaque()],
    )
}

unsafe fn samples(b: &AudioBuffer) -> &[f32] {
    if b.mData.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(b.mData.cast::<f32>(), b.mDataByteSize as usize / 4) } }
}

/// On Core Audio's IO thread: mixes to mono into the feed, nothing more.
unsafe extern "C-unwind" fn io_proc(
    _: AudioObjectID,
    _: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    _: NonNull<AudioTimeStamp>,
    _: NonNull<AudioBufferList>,
    _: NonNull<AudioTimeStamp>,
    feed: *mut c_void,
) -> i32 {
    let feed = unsafe { &*feed.cast::<Feed>() };
    let list = input.as_ptr();
    // mBuffers is declared [AudioBuffer; 1] but holds mNumberBuffers of them
    let bufs = unsafe {
        std::slice::from_raw_parts((&raw const (*list).mBuffers).cast::<AudioBuffer>(), (*list).mNumberBuffers as usize)
    };
    match bufs {
        [] => {}
        [b] => {
            let ch = b.mNumberChannels.max(1) as usize;
            feed.push(unsafe { samples(b) }.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
        }
        // non-interleaved: one buffer per channel
        _ => {
            let len = bufs.iter().map(|b| unsafe { samples(b) }.len()).min().unwrap_or(0);
            let n = bufs.len() as f32;
            feed.push((0..len).map(|i| bufs.iter().map(|b| unsafe { samples(b) }[i]).sum::<f32>() / n));
        }
    }
    0
}

/// A tap on some processes and the device reading it; dropping it tears both down.
struct Tap {
    procs: Vec<AudioObjectID>,
    tap: AudioObjectID,
    device: AudioObjectID,
    io: AudioDeviceIOProcID,
    feed: Arc<Feed>,
    reader: Option<JoinHandle<()>>,
}

impl Tap {
    fn new(procs: Vec<AudioObjectID>, serial: u32, on_frame: OnFrame) -> Result<Tap, String> {
        // 2 s at up to 96 kHz, should the reader stall
        let mut t = Tap { procs, tap: 0, device: 0, io: None, feed: Feed::new(192_000), reader: None };
        let uid = autoreleasepool(|_| -> Result<String, String> {
            let ids: Vec<Retained<NSNumber>> = t.procs.iter().map(|&p| NSNumber::new_u32(p)).collect();
            let desc = unsafe {
                let desc = CATapDescription::initMonoMixdownOfProcesses(CATapDescription::alloc(), &NSArray::from_retained_slice(&ids));
                desc.setName(&NSString::from_str("mentalese"));
                desc.setPrivate(true);
                desc.setMuteBehavior(CATapMuteBehavior::Unmuted);
                desc
            };
            // the first tap asks the user for permission (NSAudioCaptureUsageDescription)
            check(unsafe { AudioHardwareCreateProcessTap(Some(&desc), &mut t.tap) }, "listen to the app")?;
            Ok(unsafe { desc.UUID().UUIDString() }.to_string())
        })?;
        // from here on, dropping `t` undoes whatever was done
        let format: AudioStreamBasicDescription = get(t.tap, kAudioTapPropertyFormat).ok_or("can't read the app's audio format")?;
        let float = format.mFormatID == kAudioFormatLinearPCM && format.mFormatFlags & kAudioFormatFlagIsFloat != 0;
        if !float || format.mBitsPerChannel != 32 || format.mSampleRate < 8_000.0 {
            return Err("the app's audio format isn't one mentalese reads".into());
        }
        let desc = aggregate(&uid, &format!("mentalese.capture.{}.{serial}", std::process::id()));
        check(unsafe { AudioHardwareCreateAggregateDevice(desc.as_opaque(), NonNull::from(&mut t.device)) }, "create the capture device")?;
        // the feed outlives the IOProc: Drop destroys the IOProc before the fields go
        let feed = Arc::as_ptr(&t.feed).cast_mut().cast();
        check(unsafe { AudioDeviceCreateIOProcID(t.device, Some(io_proc), feed, NonNull::from(&mut t.io)) }, "read the app's audio")?;
        let (feed, mut framer) = (t.feed.clone(), Framer::new(format.mSampleRate as u32, CAPTURE_BASE, on_frame));
        let reader = std::thread::Builder::new().name("app-capture-read".into()).spawn(move || {
            let mut raw = Vec::new();
            while feed.take(&mut raw).is_ok() {
                framer.push(&raw);
            }
        });
        t.reader = Some(reader.map_err(|e| e.to_string())?);
        check(unsafe { AudioDeviceStart(t.device, t.io) }, "start reading the app's audio")?;
        Ok(t)
    }
}

impl Drop for Tap {
    fn drop(&mut self) {
        unsafe {
            if self.io.is_some() {
                // Assumed (untested here): once these return off the IO thread, the IOProc isn't
                // running and won't run again, so the feed it points at may go.
                AudioDeviceStop(self.device, self.io);
                AudioDeviceDestroyIOProcID(self.device, self.io);
            }
            if self.device != 0 {
                AudioHardwareDestroyAggregateDevice(self.device);
            }
            if self.tap != 0 {
                AudioHardwareDestroyProcessTap(self.tap);
            }
        }
        self.feed.end(String::new());
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
    }
}

pub(super) fn capture_app(app: &str, on_frame: super::OnFrame) -> Result<Box<dyn Send>, String> {
    if AnyClass::get(c"CATapDescription").is_none() {
        return Err("hearing other apps needs macOS 14.2 or later".into());
    }
    let app = app.to_lowercase();
    let on_frame: OnFrame = Arc::new(Mutex::new(on_frame));
    let mut tap: Option<Tap> = None;
    let mut serial = 0;
    let watcher = Watcher::spawn(move || {
        let mut procs: Vec<AudioObjectID> = processes()
            .into_iter()
            .filter(|p| p.exe.to_lowercase().contains(&app) || p.bundle.to_lowercase().contains(&app))
            .map(|p| p.object)
            .collect();
        procs.sort_unstable();
        if tap.as_ref().map(|t| &t.procs) == Some(&procs) {
            return;
        }
        // The app started, quit, or gained a process (Chromium starts its audio helper late): tap
        // anew. One stream throughout, so a sentence in progress isn't lost.
        tap = None;
        if !procs.is_empty() {
            serial += 1;
            tap = Tap::new(procs, serial, on_frame.clone()).ok();
        }
    })?;
    Ok(Box::new(watcher))
}
