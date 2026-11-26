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
