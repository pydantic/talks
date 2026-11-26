//! A very small band: drum kit, synth pad, guitar and bass, all synthesised from scratch.
//!
//! Parts are written into the current bar as 16-step patterns, then [`Band::bar`] sends the bar
//! to the speakers and blocks until the speakers are nearly out of music. Whoever is calling
//! (a Python script in a sandbox, in this talk) is therefore always composing one bar ahead of
//! what the room is hearing, which hides the latency of each call.
//!
//! Set `BAND_WAV=out.wav` to write a file instead of playing, which needs no sound card.

use std::{
    env,
    f32::consts::TAU,
    fs,
    num::NonZero,
    time::{Duration, Instant},
};

use rodio::{DeviceSinkBuilder, MixerDeviceSink, Player, buffer::SamplesBuffer};

const RATE: usize = 44_100;
const STEPS: usize = 16;

/// Where finished bars go.
enum Output {
    Speakers { _sink: MixerDeviceSink, player: Player },
    Wav { path: String, samples: Vec<f32> },
}

pub struct Band {
    /// Samples in one 16th note.
    step: usize,
    /// The bar being written, plus whatever is still ringing from earlier bars.
    mix: Vec<f32>,
    output: Output,
    noise: Noise,
    /// When the music already sent to the speakers runs out.
    queued_until: Option<Instant>,
}

impl Band {
    /// Opens the speakers (or the file named by `BAND_WAV`) at `bpm` beats per minute.
    pub fn new(bpm: f32) -> Self {
        let output = match env::var("BAND_WAV") {
            Ok(path) => Output::Wav { path, samples: Vec::new() },
            Err(_) => {
                let mut sink = DeviceSinkBuilder::open_default_sink().expect("no speakers found");
                sink.log_on_drop(false);
                let player = Player::connect_new(sink.mixer());
                Output::Speakers { _sink: sink, player }
            }
        };
        Self {
            step: (RATE as f32 * 60.0 / bpm / 4.0) as usize,
            mix: Vec::new(),
            output,
            noise: Noise(0x2545_f491),
            queued_until: None,
        }
    }

    /// Writes a drum pattern into the current bar.
    ///
    /// `name` is `kick`, `snare`, `hat`, `tom` or `crash`. Each of the 16 characters is one 16th
    /// note: `.` is a rest, `x` a hit and `X` a loud one. The big gated snare is `X`. Toms use
    /// digits instead, `1` (high) to `5` (floor). Spaces are ignored.
    pub fn drum(&mut self, name: &str, pattern: &str) {
        for (i, c) in steps(pattern) {
            let loud = c.is_ascii_uppercase();
            let (sound, gain) = match (name, c) {
                (_, '.') => continue,
                ("kick", _) => (kick(), if loud { 0.9 } else { 0.8 }),
                ("snare", _) => (snare(&mut self.noise, loud), if loud { 0.6 } else { 0.35 }),
                ("hat", _) => (hat(&mut self.noise), if loud { 0.5 } else { 0.3 }),
                ("crash", _) => (crash(&mut self.noise), 0.6),
                ("tom", '1'..='5') => (tom(&mut self.noise, c as usize - '1' as usize), 0.5),
                _ => continue,
            };
            self.add(i * self.step, &sound, gain);
        }
    }

    /// Holds a synth chord (`Dm`, `C`, `Bb`...) for the whole bar, fading into the next one.
    pub fn pad(&mut self, chord: &str) {
        let (root, third) = parse_chord(chord);
        let bar = self.step * STEPS;
        let release = RATE * 8 / 10;
        let mut sound = vec![0.0; bar + release];
        for semitone in [root - 12, root, root + third, root + 7, root + 12] {
            // two saws a few cents apart, which is most of what makes a pad sound like a pad
            for detune in [0.996, 1.004] {
                let freq = pitch(semitone) * detune;
                for (i, s) in sound.iter_mut().enumerate() {
                    *s += 2.0 * (i as f32 * freq / RATE as f32).fract() - 1.0;
                }
            }
        }
        lowpass(&mut sound, 900.0);
        lowpass(&mut sound, 900.0);
        let attack = RATE * 35 / 100;
        for (i, s) in sound.iter_mut().enumerate() {
            let fade_in = (i as f32 / attack as f32).min(1.0);
            let fade_out = ((bar + release - i) as f32 / release as f32).min(1.0);
            *s *= fade_in * fade_out;
        }
        self.add(0, &sound, 0.045);
    }

    /// Plays a power chord on the root of `chord`: `x` lets it ring, `o` is a short muted chug.
    pub fn guitar(&mut self, chord: &str, pattern: &str) {
        let (root, _) = parse_chord(chord);
        for (i, c) in steps(pattern) {
            let (seconds, damping) = match c {
                'x' | 'X' => (2.2, 0.997),
                'o' => (0.16, 0.96),
                _ => continue,
            };
            let mut sound = vec![0.0; (seconds * RATE as f32) as usize];
            // root, fifth and octave, strummed a few milliseconds apart
            for (string, semitone) in [root - 24, root - 17, root - 12].into_iter().enumerate() {
                let note = pluck(&mut self.noise, pitch(semitone), sound.len(), damping);
                let offset = string * RATE / 150;
                for (s, n) in sound[offset..].iter_mut().zip(note) {
                    *s += n;
                }
            }
            // an overdriven amp, then a speaker cabinet that loses the fizz
            for s in &mut sound {
                *s = (*s * 6.0).tanh();
            }
            lowpass(&mut sound, 3200.0);
            fade_out(&mut sound, RATE / 20);
            self.add(i * self.step, &sound, 0.13);
        }
    }

    /// Plays the root of `chord` on the bass wherever `pattern` has an `x`.
    pub fn bass(&mut self, chord: &str, pattern: &str) {
        let (root, _) = parse_chord(chord);
        let freq = pitch(root - 24);
        for (i, c) in steps(pattern) {
            if c == '.' {
                continue;
            }
            let mut sound: Vec<f32> = (0..RATE * 45 / 100)
                .map(|n| {
                    let t = n as f32 / RATE as f32;
                    let saw = 2.0 * (t * freq).fract() - 1.0;
                    (saw + (TAU * freq * t).sin()) * (-t / 0.3).exp()
                })
                .collect();
            lowpass(&mut sound, 500.0);
            fade_out(&mut sound, RATE / 50);
            self.add(i * self.step, &sound, 0.3);
        }
    }

    /// Sends the bar to the speakers, then waits until it is about to start playing.
    pub async fn bar(&mut self) {
        let bar = self.step * STEPS;
        self.send(bar);
        // keep one bar queued: enough to hide a slow call, not so much that the music runs ahead
        self.wait(Duration::from_secs_f32(bar as f32 / RATE as f32)).await;
    }

    /// Lets the last notes ring out, and waits for the speakers to finish.
    pub async fn finish(mut self) {
        self.send(self.mix.len());
        self.wait(Duration::ZERO).await;
        match self.output {
            Output::Speakers { player, .. } => player.sleep_until_end(),
            Output::Wav { path, samples } => write_wav(&path, &samples),
        }
    }

    fn add(&mut self, at: usize, sound: &[f32], gain: f32) {
        if self.mix.len() < at + sound.len() {
            self.mix.resize(at + sound.len(), 0.0);
        }
        for (m, s) in self.mix[at..].iter_mut().zip(sound) {
            *m += s * gain;
        }
    }

    fn send(&mut self, len: usize) {
        self.mix.resize(self.mix.len().max(len), 0.0);
        // tanh is a soft limiter, for the odd peak: the mix itself should stay below full scale
        let samples: Vec<f32> = self.mix.drain(..len).map(|s| (s * 0.8).tanh()).collect();
        match &mut self.output {
            Output::Speakers { player, .. } => {
                let len = Duration::from_secs_f32(samples.len() as f32 / RATE as f32);
                self.queued_until = Some(queue_end(self.queued_until, Instant::now(), len));
                let (mono, rate) = (NonZero::new(1).unwrap(), NonZero::new(RATE as u32).unwrap());
                player.append(SamplesBuffer::new(mono, rate, samples));
            }
            Output::Wav { samples: all, .. } => all.extend(samples),
        }
    }

    /// Sleeps until only `keep` of music is left in the queue.
    async fn wait(&self, keep: Duration) {
        if let Some(sleep) = self.queued_until.and_then(|q| q.checked_duration_since(Instant::now() + keep)) {
            tokio::time::sleep(sleep).await;
        }
    }
}

/// When the queue runs out after `len` more music is appended to it at `now`.
///
/// If the queue already ran dry (a slow call), the new music starts now, not where the old ran out.
fn queue_end(queued_until: Option<Instant>, now: Instant, len: Duration) -> Instant {
    queued_until.filter(|q| *q > now).unwrap_or(now) + len
}

/// The 16 steps of a pattern, with their positions.
fn steps(pattern: &str) -> impl Iterator<Item = (usize, char)> + '_ {
    pattern.chars().filter(|c| !c.is_whitespace()).take(STEPS).enumerate()
}

/// `Dm` -> (semitones of the root above C4, semitones from root to third).
fn parse_chord(chord: &str) -> (i32, i32) {
    let mut chars = chord.chars().peekable();
    let mut root = match chars.next() {
        Some('C') => 0,
        Some('D') => 2,
        Some('E') => 4,
        Some('F') => 5,
        Some('G') => 7,
        Some('A') => 9,
        Some('B') => 11,
        _ => 0,
    };
    match chars.peek() {
        Some('#') => root += 1,
        Some('b') => root -= 1,
        _ => {}
    }
    // keep everything within a few semitones of C, so Bb sits below D rather than above it
    if root > 6 {
        root -= 12;
    }
    (root, if chord.ends_with('m') { 3 } else { 4 })
}

/// Frequency of the note `semitone` semitones above middle C.
fn pitch(semitone: i32) -> f32 {
    261.63 * 2f32.powf(semitone as f32 / 12.0)
}

/// A sine that drops from a thump to a rumble, with a click on the front so small speakers hear it.
fn kick() -> Vec<f32> {
    let mut phase = 0.0f32;
    (0..RATE * 4 / 10)
        .map(|n| {
            let t = n as f32 / RATE as f32;
            phase += TAU * (48.0 + 160.0 * (-t / 0.035).exp()) / RATE as f32;
            phase.sin() * (-t / 0.16).exp() + 0.4 * (TAU * 1800.0 * t).sin() * (-t / 0.004).exp()
        })
        .collect()
}

/// The drum-machine snare of the intro: a short tone with a little noise for the wires.
/// The `big` one is the 1981 trick: a drum in a huge room, cut off dead by a gate.
fn snare(noise: &mut Noise, big: bool) -> Vec<f32> {
    let (len, gate) = if big { (0.4, 0.26) } else { (0.16, 0.14) };
    let len = (len * RATE as f32) as usize;
    // the room is noise with the top taken off: a boom, where unfiltered noise would be a hiss
    let mut room: Vec<f32> = (0..len).map(|_| noise.next()).collect();
    let mut wires = room.clone();
    lowpass(&mut room, 1500.0);
    lowpass(&mut room, 1500.0);
    highpass(&mut room, 250.0);
    lowpass(&mut wires, if big { 4500.0 } else { 7000.0 });
    let mut sound: Vec<f32> = (0..len)
        .map(|n| {
            let t = n as f32 / RATE as f32;
            let skin = (TAU * 185.0 * t).sin() + 0.6 * (TAU * 330.0 * t).sin();
            let hit = skin * (-t / 0.07).exp() + wires[n] * (-t / 0.05).exp();
            if big { hit + room[n] * 2.2 * (0.6 * (-t / 0.12).exp() + 0.4) } else { hit }
        })
        .collect();
    gated(&mut sound, gate);
    sound
}

/// Concert toms, high (0) to floor (4), with the same gated room as the big snare.
fn tom(noise: &mut Noise, which: usize) -> Vec<f32> {
    let freq = [196.0, 155.0, 123.0, 98.0, 82.0][which];
    let mut phase = 0.0f32;
    let mut room: Vec<f32> = (0..RATE * 45 / 100).map(|_| noise.next()).collect();
    lowpass(&mut room, 1400.0);
    let mut sound: Vec<f32> = room
        .iter()
        .enumerate()
        .map(|(n, room)| {
            let t = n as f32 / RATE as f32;
            phase += TAU * freq * (1.0 + 0.7 * (-t / 0.03).exp()) / RATE as f32;
            let body = phase.sin() + 0.35 * (2.0 * phase).sin();
            body * (0.55 * (-t / 0.22).exp() + 0.45) + room * 1.0 * (0.5 * (-t / 0.1).exp() + 0.5)
        })
        .collect();
    gated(&mut sound, 0.34);
    sound
}

/// A closed hi-hat: a short tick of [`metal`], with only a trace of noise.
fn hat(noise: &mut Noise) -> Vec<f32> {
    let mut sound = metal(noise, RATE / 14, 0.15);
    highpass(&mut sound, 7000.0);
    for (n, s) in sound.iter_mut().enumerate() {
        *s *= (-(n as f32) / RATE as f32 / 0.022).exp();
    }
    sound
}

/// A crash cymbal: the same metal, lower, noisier and left to ring.
fn crash(noise: &mut Noise) -> Vec<f32> {
    let mut sound = metal(noise, RATE * 2, 0.3);
    highpass(&mut sound, 3500.0);
    for (n, s) in sound.iter_mut().enumerate() {
        let t = n as f32 / RATE as f32;
        *s *= 0.7 * (-t / 0.45).exp() + 0.3 * (-t / 0.05).exp();
    }
    fade_out(&mut sound, RATE / 10);
    sound
}

/// What a cymbal is made of: six square waves at frequencies that share no harmonics, so they
/// clang rather than sing. This is how the drum machines of the time did it. Plain white noise
/// has no pitch at all and just sounds like hiss.
fn metal(noise: &mut Noise, len: usize, noisiness: f32) -> Vec<f32> {
    const PARTIALS: [f32; 6] = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0];
    (0..len)
        .map(|n| {
            let t = n as f32 / RATE as f32;
            let squares: f32 = PARTIALS.iter().map(|f| if (t * f * 8.0).fract() < 0.5 { 1.0 } else { -1.0 }).sum();
            squares / 6.0 * (1.0 - noisiness) + noise.next() * noisiness
        })
        .collect()
}

/// A plucked string by Karplus-Strong: a burst of noise going round a delay line one period long,
/// averaged a little on each lap so the high harmonics die first, as they do on a real string.
fn pluck(noise: &mut Noise, freq: f32, len: usize, damping: f32) -> Vec<f32> {
    let mut line: Vec<f32> = (0..(RATE as f32 / freq) as usize).map(|_| noise.next()).collect();
    lowpass(&mut line, 2500.0);
    let period = line.len();
    (0..len)
        .map(|i| {
            let out = line[i % period];
            line[i % period] = damping * 0.5 * (out + line[(i + 1) % period]);
            out
        })
        .collect()
}

/// One-pole low-pass filter, in place.
fn lowpass(sound: &mut [f32], cutoff: f32) {
    let k = 1.0 - (-TAU * cutoff / RATE as f32).exp();
    let mut y = 0.0;
    for s in sound {
        y += k * (*s - y);
        *s = y;
    }
}

/// One-pole high-pass filter, in place: whatever the low-pass leaves out.
fn highpass(sound: &mut [f32], cutoff: f32) {
    let mut low = sound.to_vec();
    lowpass(&mut low, cutoff);
    for (s, l) in sound.iter_mut().zip(low) {
        *s -= l;
    }
}

/// Fades out the last `len` samples.
fn fade_out(sound: &mut [f32], len: usize) {
    let start = sound.len().saturating_sub(len);
    for (i, s) in sound[start..].iter_mut().enumerate() {
        *s *= 1.0 - i as f32 / len as f32;
    }
}

/// A noise gate: full volume until `seconds`, then silence 20ms later.
fn gated(sound: &mut Vec<f32>, seconds: f32) {
    sound.truncate((seconds * RATE as f32) as usize + RATE / 50);
    fade_out(sound, RATE / 50);
}

/// White noise from a xorshift generator: the same every run, and no dependency.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

/// Writes mono 16-bit PCM.
fn write_wav(path: &str, samples: &[f32]) {
    let data_len = samples.len() as u32 * 2;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend(b"RIFF");
    bytes.extend((36 + data_len).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes()); // PCM
    bytes.extend(1u16.to_le_bytes()); // mono
    bytes.extend((RATE as u32).to_le_bytes());
    bytes.extend((RATE as u32 * 2).to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(data_len.to_le_bytes());
    for s in samples {
        bytes.extend(((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    fs::write(path, bytes).expect("could not write the wav file");
}

/// The Logfire API key, from the environment or from a `.env` file next to `Cargo.toml`.
pub fn logfire_api_key() -> String {
    env::var("LOGFIRE_API_KEY")
        .ok()
        .or_else(|| {
            let dotenv = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/.env")).ok()?;
            let line = dotenv.lines().find_map(|l| l.trim_start_matches("export ").strip_prefix("LOGFIRE_API_KEY="))?;
            Some(line.trim_matches(['"', '\'']).to_owned())
        })
        .expect("set LOGFIRE_API_KEY, or put it in .env")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_end_appends_to_a_busy_queue() {
        let now = Instant::now();
        let bar = Duration::from_secs(2);
        assert_eq!(queue_end(Some(now + bar), now, bar), now + bar * 2);
    }

    #[test]
    fn queue_end_restarts_a_drained_queue() {
        let now = Instant::now() + Duration::from_secs(60);
        let bar = Duration::from_secs(2);
        // the queue ran out 30s ago: the next bar plays now, it doesn't count the silence as music
        assert_eq!(queue_end(Some(now - Duration::from_secs(30)), now, bar), now + bar);
        assert_eq!(queue_end(None, now, bar), now + bar);
    }
}
