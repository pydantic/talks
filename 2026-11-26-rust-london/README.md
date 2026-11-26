



# Monty

## Zed x Rust London - November 2026

A Python interpreter, written in Rust, for code written by AI.

Samuel Colvin - Pydantic

Slides at <https://github.com/pydantic/talks>














## whoami

**Pydantic** - Python people, who mostly write Rust.

* **pydantic Validation** - the validation in Pydantic, PyO3
* **jiter** - JSON parser
* **Pydantic Logfire** - observability, the database is Rust on DataFusion
* **Pydantic Monty** - this talk
















## Why

LLMs are better at writing code than at calling tools one at a time.

So every agent needs somewhere to run code it just made up.

That means a new sandbox on every step of every agent.


| Sandbox              | New sandbox |
|----------------------|-------------|
| Pyodide in Deno      | 2700 ms     |
| a sandbox service    | 1500 ms     |
| `docker exec`        | 195 ms      |
| CPython in wasmtime  | 16 ms       |
| **Monty**            | **0.8 ms**  |


![The sandbox continuum](sandbox-continuum.png)















## No way out

Every other sandbox runs a real language, then tries to block the exits.

Monty is a **language-level sandbox**:

* no syscalls
* no filesystem
* no network
* no FFI, no C dependencies

Not "blocked". **The capability does not exist in the VM.**

The only thing sandboxed code can do is stop, and ask the host to do something.


**Tradeoff:** Monty will never run your existing Python application.

**That's fine with me**














## Hello world

```bash
uv add pydantic-monty
```

```python title="hello.py"
from pydantic_monty import Monty


def get_weather(city: str) -> str:
    return f'raining in {city}'


code = """
weather = get_weather('London')
f'Hello Rust London, it is {weather}'
"""

with Monty() as monty:
    with monty.checkout() as session:
        output = session.feed_run(code, external_lookup={'get_weather': get_weather})
        print(output)
```

The sandbox can call `get_weather`, and nothing else.














## Hello world, in Rust

```bash
cargo add monty monty-types
```

```rust title="hello.rs"
use monty::{MontyRun, RunProgress};
use monty_types::{CompileOptions, MontyObject, PrintWriter, ResourceTracker};

const CODE: &str = "
weather = get_weather('London')
f'Hello Rust London, it is {weather}'
";

fn main() {
    let inputs = vec!["get_weather".to_owned()];
    let run = MontyRun::new(CODE.to_owned(), "main.py", inputs, CompileOptions::default());
    let get_weather = MontyObject::function("get_weather", None);
    let mut progress = run
        .unwrap()
        .start(vec![get_weather], ResourceTracker::default(), PrintWriter::Stdout)
        .unwrap();

    loop {
        progress = match progress {
            // the interpreter has stopped, and is asking us to do something
            RunProgress::FunctionCall(call) => {
                println!("sandbox wants: {}({})", call.function_name, call.args.arg(0).unwrap());
                call.resume(MontyObject::string("raining"), PrintWriter::Stdout).unwrap()
            }
            RunProgress::Complete(result) => break println!("{result}"),
            other => panic!("unexpected {other:?}"),
        }
    }
}
```


`resume(self)` takes the suspension by value. You can't answer the same call twice, the compiler won't let you.














## What about files?

```rust title="no_way_out.rs"
use monty::{MontyRun, RunProgress};
use monty_types::{CompileOptions, MontyObject, PrintWriter, ResourceTracker};

const CODE: &str = "
from pathlib import Path

Path('/etc/passwd').read_text()
";

fn main() {
    let run = MontyRun::new(CODE.to_owned(), "main.py", vec![], CompileOptions::default()).unwrap();
    let progress = run.start(vec![], ResourceTracker::default(), PrintWriter::Stdout).unwrap();

    // Monty has no code that opens a file. All it can do is stop and ask.
    let RunProgress::OsCall(call) = progress else { panic!("unexpected {progress:?}") };
    println!("sandbox wants: {:?}", call.function_call);

    let progress = call.resume(MontyObject::string("nice try"), PrintWriter::Stdout).unwrap();
    println!("sandbox got:   {}", progress.into_complete().unwrap());
}
```

That's the whole interface:

```rust
// crates/monty/src/run_progress.rs
pub enum RunProgress {
    FunctionCall(FunctionCall),
    OsCall(OsCall),
    ResolveFutures(ResolveFutures),
    NameLookup(NameLookup),
    Complete(MontyObject),
}
```














## How it works

```
Python source
   │
   ▼  ruff_python_parser      the only bit we didn't write
   ▼  prepare                 scopes resolved, every name interned to an integer
   ▼  bytecode compiler       one-byte opcodes, same idea as CPython's .pyc
   ▼  VM                      a `match` in a loop
   ▼  heap                    paged arena, manual refcounts, cycle collector
```

* 150k lines of Rust in the interpreter, 20 crates
* 800+ Python test scripts, each run on Monty **and** CPython - output must match exactly
* `Value` is 16 bytes, with a `const` assert to keep it that way
* no generators, no metaclasses, no `numpy` - a small and growing bit of the standard library
















## serde on everything

353 `#[derive(Serialize, Deserialize)]`s. Including this one:

```rust
// crates/monty/src/run_progress.rs
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Snapshot {
    /// The executor containing compiled code and interns.
    pub(crate) executor: Executor,
    /// The VM state containing stack, frames, globals, and exception state.
    pub(crate) vm_state: VMSnapshot,
    /// The heap containing all allocated objects.
    pub(crate) heap: Heap,
}
```

So the interpreter can be paused half way through a function, and written to a database.

```rust title="snapshot.rs"
use monty::{Dump, MontyRun, RunProgress, Session, SessionRef, dump};
use monty_types::{CompileOptions, MontyObject, PrintWriter, ResourceTracker};

const CODE: &str = "
orders = [{'id': n, 'total': n * 9.99} for n in range(1, 6)]
approved = ask_a_human(f'refund {len(orders)} orders?')
sum(o['total'] for o in orders) if approved else 0
";

fn main() {
    let inputs = vec!["ask_a_human".to_owned()];
    let run = MontyRun::new(CODE.to_owned(), "main.py", inputs, CompileOptions::default());
    let ask = MontyObject::function("ask_a_human", None);
    let progress = run.unwrap().start(vec![ask], ResourceTracker::default(), PrintWriter::Stdout);
    let progress = progress.unwrap();

    // paused inside the Python, waiting for a human: the whole interpreter is one serde value
    let bytes = dump("main.py", None, &[], SessionRef::Running(&progress)).unwrap();
    drop(progress);
    println!("the interpreter is now {} bytes", bytes.len());

    // ...days later, on another machine
    let Session::Running(progress) = Dump::load(&bytes).unwrap().state else { panic!() };
    let RunProgress::FunctionCall(call) = *progress else { panic!() };
    println!("sandbox wants: {}({})", call.function_name, call.args.arg(0).unwrap());

    let progress = call.resume(MontyObject::bool(true), PrintWriter::Stdout).unwrap();
    println!("result: {}", progress.into_complete().unwrap());
}
```

* CBOR, with a magic number and a version
* `#[serde(rename = "I")] Int(i64)` - single letter variant names, the dump is mostly `Value`s
* `#[serde(skip)]` on the count of live borrows, it must be 0 when you dump
* the GC's colour bits are saved on purpose, drop them and you leak a cycle

**Durable execution**, with no replay and no workflow framework. A microVM can snapshot too, at thousands of times the size.














## A custom allocator

A memory limit is only a limit if you count every byte.

We tried estimating the size of each Python object. People found the ones we forgot (`list.append`, `json.dumps(indent=...)`).

So now we count in the one place nothing can avoid:

```rust
// crates/monty-alloc/src/lib.rs - 141 lines
unsafe impl GlobalAlloc for LimitedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        charge(layout.size());
        // SAFETY: `layout` comes from the caller and is forwarded unchanged.
        let ptr = unsafe { System.alloc(layout) };
        ...
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        refund(layout.size());
        // SAFETY: `ptr` came from our `alloc`/`realloc` with this same `layout`.
        unsafe { System.dealloc(ptr, layout) };
    }
}

fn charge(size: usize) {
    let live = LIVE_MEMORY.fetch_add(size, Ordering::Relaxed).saturating_add(size);
    if live > HARD_LIMIT.load(Ordering::Relaxed) {
        out_of_memory(...);
    }
}
```

* **soft limit** - the VM reads the same counter and raises a Python `MemoryError`
* **hard limit** - 4 MiB later, `process::exit(65)`. Not `panic!`: panicking allocates.

Which means the process running Python has to be allowed to die...














## The wire protocol

You can't make a process crash-proof. Stack overflows abort. Allocators abort.

So the interpreter runs in a pool of worker subprocesses, and a dead worker gets replaced.

```rust
// crates/monty-proto/src/frame.rs
pub fn write_frame(writer: &mut impl Write, msg: &impl Message) -> Result<(), FrameError> {
    let body = encode_to_capped_vec(msg)?;
    let len = u32::try_from(body.len()).unwrap_or(u32::MAX);
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}
```

* 4 byte length, then protobuf, over stdin and stdout
* strict alternation: one request, one reply - a client is a blocking read loop, in any language
* the Python and JS packages don't link the interpreter any more, they speak this to a worker
* the same messages over gRPC, and the worker can be on someone else's computer














## Everywhere

Because it's Rust:

* **Rust** - `monty`, in your process, or `monty-pool`
* **Python** - PyO3, `pip install pydantic-monty`
* **JavaScript** - napi-rs, `npm i @pydantic/monty`
* **Browser** - a WASM component
* **Go** - `fugue-labs/monty-go`, WASM in wazero, no cgo
* **Dart & Flutter** - `dart_monty`
* **anything else** - MCP

**Agent Sandboxes** in Pydantic Logfire: Monty, hosted, over that gRPC stream.

| Time to run `1 + 1` in a new sandbox | median  |
|--------------------------------------|---------|
| Logfire Agent Sandboxes, pre-warmed  | 24 ms   |
| Logfire Agent Sandboxes              | 85 ms   |
| E2B                                  | 459 ms  |
| Modal                                | 464 ms  |
| Daytona                              | 506 ms  |
| Vercel                               | 1070 ms |

(from a VM in GCP us-east4, this week)














## Hack Monty

hackmonty.com - break out of the sandbox, read the secret, get paid.

**Round 1**, April, $5,000:

* 1.5 million requests
* 650 IP addresses
* 65 submissions
* **1** escape, in under 48 hours

```rust
// crates/monty/src/heap.rs - v0.0.15
impl<'a, T: ?Sized> HeapRead<'a, T> {
    /// Accesses the value contained in this reference.
    pub fn get<'r, RT: ResourceTracker>(&self, _: &'r HeapReader<'a, RT>) -> &'r T {
        // SAFETY: (DH)
        //  - The HeapReader has an invariant lifetime 'a which guarantees that this HeapRead
        //    came from the heap borrowed by this HeapReader.
        //  - The address of the `HeapValue` never changes because entries are stored in
        //    paged storage (`HeapEntries`) where each page is never reallocated or moved.
        //  - The HeapRead holds a strong reader reference (via the `readers` counter in
        //    `HeapValue`) which guarantees the entry will never be freed by `dec_ref`
        //    while this `HeapRead` exists.
        //  - The type of the `HeapValue` can never change once allocated. This is
        //    guaranteed by never exposing `&mut HeapData` outside of this module.
        //  - The borrow on `HeapReader` guarantees that there are no mutable borrows on any heap
        //    data while the return value of this function is alive.
        unsafe { self.value.as_ref() }
    }
}
```

Five careful invariants. All true. Miri was happy.

"never be freed by `dec_ref`" - but there were two ways to free an object.














## The exploit

```python
my_list.sort(key=evil)
```

* `sort` holds a `HeapRead` on the list
* `evil()` allocates until the garbage collector runs
* the GC walks from its roots... and the list being sorted wasn't one of them
* the list is freed, while Rust still has a `&` to it
* `evil()` allocates a `bytes` in the same slot
* Rust thinks it's reading a list, it's reading bytes the attacker wrote
* out comes the `SECRET` environment variable

Owen Kwan, Veria Labs. Patched within a day.


> without the `unsafe` code, this flaw in the GC would likely have led Monty to panic

One `unsafe` turned a bug into an exploit.














## What we changed

* the mark-and-sweep GC is gone - cycles are found by trial deletion, which needs no list of roots to get wrong
* `unsafe_code = "deny"` for the workspace, every exception is an `#[expect(unsafe_code, reason = "...")]`
* all the interpreter's `unsafe` is in one directory, `heap/`
* ~25 tests that pass only if misusing `HeapReader` **fails to compile**
* Miri in CI, `cargo careful`, fuzzing

**Round 2**, $10,000 - nobody got out.

**Round 3**, $20,000.














## Demo

Python, in a sandbox, in a data centre. Rust, on this laptop. One of them has speakers.

```rust title="play_tune.rs"
use std::time::Duration;

use monty_pool::{CheckoutOptions, Pool, PoolConfig, ReplConfig, ResumeValue, TurnEvent, on_print_sync};
use monty_types::MontyObject;
use rust_london::{Band, logfire_api_key};

// this is the code that runs in the sandbox
const CODE: &str = "
# one chord per bar: the intro, the fill, the drums, and a last chord to finish
chords = 'Dm C Bb C Dm C  Dm C Bb C  Dm'.split()

# every call below is a function on the laptop: the sandbox stops, and waits for it to return
# a pattern is one bar of 16th notes: '.' is a rest, 'x' a hit, 'X' a loud one
for n, chord in enumerate(chords):
    print(f'bar {n + 1:>2}  {chord}')
    # the synth holds the chord for the whole bar
    pad(chord)

    if n < 5:
        # a lonely drum machine
        kick('x... .... ..x. ....')
        snare('.... x... .... x...')
        hat('x.x. x.x. x.x. x.xx')
        if n >= 2:
            # the guitar joins in, one power chord left to ring
            guitar(chord, 'x... .... .... ....')
    elif n == 5:
        # you know the one: toms are numbered, 1 is the highest and 5 the floor tom
        tom('11.2 2.33 .44. 5.5.')
        kick('.... .... .... x.x.')
    elif n < 10:
        # the gorilla
        kick('X... ..X. X.X. ....')
        snare('.... X... .... X...')
        hat('X.x. X.x. X.x. X.x.')
        bass(chord, 'x... ..x. x.x. ....')
        guitar(chord, 'x... .... x... ....')
        if n % 2 == 0:
            crash('x')
        elif n == 7:
            # a short fill into the next bar
            tom('.... .... .... 3.45')
        else:
            # and a long one down the toms, into the last chord
            tom('.... .... 11.2 2.34')
    else:
        # everything at once, then let it ring
        crash('x')
        kick('X')
        guitar('Dm', 'x')
        bass('Dm', 'x')

    # send the bar to the speakers, and wait until it is about to play
    bar()
";

// the only things in the sandbox that can reach the outside world
const HOST_FUNCTIONS: [&str; 9] = ["kick", "snare", "hat", "tom", "crash", "pad", "guitar", "bass", "bar"];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = logfire_api_key();
    let region = key.split('_').nth(2).unwrap_or("us");
    let mut config = PoolConfig::remote(format!("https://logfire-{region}.pydantic.dev"));
    config.request_timeout = Some(Duration::from_secs(10));
    let pool = Pool::new(config).await?;
    let auth = vec![("authorization".to_owned(), format!("Bearer {key}"))];
    let options = CheckoutOptions::default().with_metadata(auth);
    let mut session = pool.checkout_with(&ReplConfig::default(), options).await?;

    let mut band = Band::new(95.0);
    let mut on_print = on_print_sync(|_, text| print!("{text}"));
    let functions: Vec<_> = HOST_FUNCTIONS
        .iter()
        .map(|name| (name.to_string(), MontyObject::function(*name, None)))
        .collect();

    let mut event = session.feed(CODE, functions, vec![], false, &mut on_print).await?;
    loop {
        event = match event {
            TurnEvent::FunctionCall { function_name, args, .. } => {
                let text = |i| args.arg(i).and_then(|arg| arg.as_str()).unwrap_or_default();
                match function_name.as_str() {
                    "pad" => band.pad(text(0)),
                    "guitar" => band.guitar(text(0), text(1)),
                    "bass" => band.bass(text(0), text(1)),
                    // returns when the speakers are nearly out of music
                    "bar" => band.bar().await,
                    drum => band.drum(drum, text(0)),
                }
                session.resume(ResumeValue::Return(MontyObject::none()), &mut on_print).await?
            }
            TurnEvent::Complete(_) => break,
            other => panic!("unexpected {other:?}"),
        }
    }
    session.finish().await?;
    band.finish().await;
    Ok(())
}
```














## Learn more

* github.com/pydantic/monty
* hackmonty.com
* pydantic.dev/articles/hack-monty-postmortem
* pydantic.dev/docs/logfire/agent-sandboxes
* x.com/pydantic
