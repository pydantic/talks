import asyncio

import logfire
import numpy as np
import sounddevice as sd

# this is the code that runs in the sandbox: In the Hall of the Mountain King, getting faster
code = """
# each '-' makes a note one beat longer
melody = (
    'B3 C#4 D4 E4 F#4 D4 F#4- F4 C#4 F4- E4 C4 E4- '
    'B3 C#4 D4 E4 F#4 D4 F#4 B4 A4 F#4 D4 F#4 A4---'
)
beat = 0.18
for octave in 0, 0, 1:
    for token in melody.split():
        note = token.rstrip('-')
        play(note[:-1] + str(int(note[-1]) + octave), beat * (1 + token.count('-')))
        # await asyncio.sleep(beat * (1 + token.count('-')))
    beat *= 0.75
"""

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
RATE = 44100
# (relative volume, how fast it fades) for each harmonic, which is what makes it sound like a piano not a beep
HARMONICS = [(1.0, 3.0), (0.5, 4.5), (0.25, 6.0), (0.12, 7.5), (0.06, 9.0)]
speakers = sd.OutputStream(samplerate=RATE, channels=1, dtype='float32')


def play(note: str, seconds: float) -> None:
    """Play a note like 'F#4' through the speakers, and return when it has finished."""
    print(note, end=' ', flush=True)
    name, octave = note[:-1], int(note[-1])
    freq = 440 * 2 ** ((NAMES.index(name) - 9 + 12 * (octave - 4)) / 12)
    t = np.arange(int(RATE * seconds)) / RATE
    sound = sum(
        vol * np.exp(-fade * t) * np.sin(2 * np.pi * freq * (h + 1) * t) for h, (vol, fade) in enumerate(HARMONICS)
    )
    # ramp up over the first 5ms and down over the last 30ms, so there are no clicks
    sound *= np.minimum(1, np.minimum(t / 0.005, (seconds - t) / 0.03))
    # write() returns once the sound has been handed to the speakers, which takes as long as the note
    speakers.write((0.4 * sound).astype('float32'))


async def main() -> None:
    async with logfire.agent_sandboxes() as pool:
        async with pool.checkout() as session:
            with speakers:
                await session.feed_run(code, external_lookup={'play': play})


if __name__ == '__main__':
    asyncio.run(main())
