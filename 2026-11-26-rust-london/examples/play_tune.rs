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
