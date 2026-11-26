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
