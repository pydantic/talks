use monty::{MontyRun, RunProgress};
use monty_fs::{MountCallOutcome, MountMode, MountTable};
use monty_types::{CompileOptions, PrintWriter, ResourceTracker};

const CODE: &str = "
from pathlib import Path

Path('/talk/Cargo.toml').read_text()
";

fn main() {
    // the sandbox sees this directory at /talk, and nothing else
    let mut mounts = MountTable::new();
    mounts.mount("/talk", env!("CARGO_MANIFEST_DIR"), MountMode::ReadOnly, None).unwrap();

    let run = MontyRun::new(CODE.to_owned(), "main.py", vec![], CompileOptions::default()).unwrap();
    let mut progress = run.start(vec![], ResourceTracker::default(), PrintWriter::Stdout).unwrap();

    loop {
        progress = match progress {
            // still just a request to stop and ask, the mount table answers it
            RunProgress::OsCall(call) => {
                println!("sandbox wants: {:?}", call.function_call);
                call.resume_with(PrintWriter::Stdout, |call| match mounts.handle_os_call(call) {
                    MountCallOutcome::Handled(Ok(value)) => value.into(),
                    MountCallOutcome::Handled(Err(e)) => e.into_exception().into(),
                    MountCallOutcome::NotHandled(call) => call.on_no_handler().into(),
                })
                .unwrap()
            }
            RunProgress::Complete(result) => break println!("sandbox got:   {result}"),
            other => panic!("unexpected {other:?}"),
        }
    }
}
