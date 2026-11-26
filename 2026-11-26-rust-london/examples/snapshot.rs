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
