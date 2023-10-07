#![feature(iter_next_chunk, buf_read_has_data_left)]

extern crate bincode; // en/decoding to file
extern crate clap;
extern crate itertools; // some iterator helper tools
extern crate log; // logging, (duh)
extern crate paste; // helper to generate test cases
extern crate rand; // used to generate random inputs
extern crate rand_distr; // more random input gen
extern crate slice_deque; // <- better version of Deque that actually turns into a slice

use bincode::{config, Decode, Encode};
use itertools::Itertools;
use log::{debug, error, info};
use slice_deque::SliceDeque;
use std::{
    error::Error,
    fmt::Display,
    io::{BufReader, BufWriter, Write},
};

#[derive(Debug, Copy, Clone, Encode, Decode, PartialEq)]
struct CompressionTuple {
    pos: usize,
    len: usize,
    next: char,
}
impl CompressionTuple {
    fn new(pos: usize, len: usize, next: char) -> Self {
        CompressionTuple { pos, len, next }
    }
    fn from_tuple((pos, len, next): (usize, usize, char)) -> Self {
        CompressionTuple { pos, len, next }
    }
}

impl Display for CompressionTuple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.pos, self.len, self.next)
    }
}

//const WINDOW_SIZE: usize = 30;
//const DICTIONARY_SIZE: usize = 26;

#[derive(Debug)]
enum ErrorTypes {
    InitialUndersized,
    DictionaryUndersized(String),
}

impl Display for ErrorTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InitialUndersized => write!(f, "Not enough bytes to compress! (at least 4 needed)"),
            Self::DictionaryUndersized(_)=> write!(f, "Out of bounds on the dictionary index, possibly bad tuple given / dictionary and/or window size was altered!")
        }
    }
}

impl<T> From<arrayvec::CapacityError<T>> for ErrorTypes {
    fn from(value: arrayvec::CapacityError<T>) -> Self {
        ErrorTypes::DictionaryUndersized(value.to_string())
    }
}

impl Error for ErrorTypes {}

#[derive(Clone)]
struct SlidingWindowArray<T> {
    back_cap: usize,
    front_cap: usize,
    pub back: SliceDeque<T>,
    pub front: SliceDeque<T>,
}

impl<T: Clone + Copy> SlidingWindowArray<T> {
    fn new(back_cap: usize, front_cap: usize) -> Self {
        SlidingWindowArray {
            back_cap,
            front_cap,
            front: SliceDeque::with_capacity(front_cap),
            back: SliceDeque::with_capacity(back_cap),
        }
    }

    pub fn front_remaining_capacity(&self) -> usize {
        self.front_cap - self.front.len()
    }
    pub fn back_remaining_capacity(&self) -> usize {
        self.back_cap - self.back.len()
    }

    fn push_to_back(&mut self, value: T) -> Option<T> {
        let removed = if self.back_remaining_capacity() == 0 {
            self.back.pop_front()
        } else {
            None
        };

        self.back.push_back(value);

        removed
    }

    fn move_front(&mut self) -> Option<T> {
        let v = self
            .front
            .pop_front()
            .expect("Front capacity is full, but nothing is inside?");
        self.push_to_back(v)
    }
    /**
     * Pushes a new element to the front array.
     * If the front array is full, then remove the first element in the front array,
     * and push it to the back of back-array (dictionary).
     * returns the removed element from the __back__ array, if any
     */
    pub fn push_new(&mut self, value: T) -> Option<T> {
        let removed = if self.front_remaining_capacity() == 0 {
            self.move_front()
        } else {
            None
        };
        self.front.push_back(value);

        removed
    }
    /**
     * the above but accepts a vec, returns any values that have been removed
     */
    pub fn push_many(&mut self, values: Vec<T>) -> Vec<Option<T>> {
        let mut out = Vec::new();
        for val in values {
            out.push(self.push_new(val));
        }
        out
    }

    /**
     * helper code, only pushes if optional exists
     */
    pub fn push_optional_new(&mut self, value: Option<T>) -> Option<T> {
        match value {
            Some(c) => self.push_new(c),
            None => None,
        }
    }

    /**
     * __NOTE__ this function differs as it will always move the left buffer's front value
     * even if the given value is None.
     */
    pub fn push_shift_optional_new(&mut self, value: Option<T>) -> Option<T> {
        let removed = if !self.front.is_empty() {
            self.move_front()
        } else {
            None
        };

        match value {
            Some(c) => self.push_new(c),
            None => removed,
        }
    }

    pub fn prepare_advancement(&mut self, n: usize) {
        if n >= self.back.capacity() {
            self.back.clear();
        } else {
            let _ = self.back.drain(0..n).collect_vec();
        }
    }
}

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
struct LZ77Program {
    _window_size: usize,
    _dictionary_size: usize,
    pub window: Rc<RefCell<SlidingWindowArray<char>>>,
}

impl LZ77Program {
    fn new(win_size: usize, dict_size: usize) -> Self {
        LZ77Program {
            _window_size: win_size,
            _dictionary_size: dict_size,
            window: Rc::new(RefCell::new(SlidingWindowArray::new(dict_size, win_size))),
        }
    }
    fn find_match(
        &self,
        window: &SlidingWindowArray<char>,
        search: &[char],
        dict: &[char],
    ) -> Option<(usize, usize)> {
        let window_size = self._window_size;
        let maximum_factor = window_size.div_ceil(window.front.len().max(1));
        /**
         * 1st. repeat the dictionary so we can represent wrap-around
         * 2nd. use the "windows" iterator so we can cycle through fixed length
         *      iterations of the dict. Unlike chunks it will overlap the substring.
         */
        fn find_match_inner(
            search: &[char],
            dict: &[char],
            maximum_factor: usize,
        ) -> Option<(usize, usize)> {
            let len = search.len();
            let repeated = dict.repeat(maximum_factor);
            let mut views = repeated.windows(len);

            for pos in 0..views.len() {
                if let Some(view) = views.next() {
                    if view == search {
                        return Some((pos, len));
                    }
                } else {
                    continue;
                }
            }

            None
        }

        let mut longest_match = None;
        debug!("Initiating search with = {:?}", search);
        // range is non inclusive
        for i in 1..search.len() + 1 {
            let result = find_match_inner(&search[0..i], dict, maximum_factor);

            match result {
                Some(v) => {
                    debug!("{:?} -> Found result = {:?}", &window.front[0..i], result);
                    longest_match = Some(v);
                }
                None => {
                    return longest_match;
                }
            }
        }

        longest_match
    }

    fn compress(&mut self, input: &str) -> Result<Vec<CompressionTuple>, ErrorTypes> {
        let mut output = Vec::new();
        //let mut dictionary = ArrayVec::<char, DICTIONARY_SIZE>::new();
        //let mut buf: ArrayVec<char, WINDOW_SIZE> = ArrayVec::new();
        let mut stream = input.chars();
        let mut window = self.window.borrow_mut();

        for _ in 0..self._window_size {
            window.push_optional_new(stream.next());
            //if let Some(c) = stream.next() {
            //    buf.push(c);
            //}
        }

        let mut cursor = 0;
        loop {
            debug!("### ITER AT CURSOR {} ###", cursor);
            // check for matches
            // we have to also handle the default cases in which no dictionary match was found.
            let search_space = if window.front.len() < self._window_size && window.front.len() > 0 {
                &window.front[0..window.front.len() - 1]
            } else {
                &window.front[..]
            };
            let (pos, len) = self
                .find_match(&window, search_space, &window.back)
                .unwrap_or((0, 0));

            debug!(
                "det pos, cursor = {}, dict.len = {}, pos = {}",
                cursor,
                window.back.len(),
                pos
            );

            // emitted position is relative to the cursor, e.g. the position is more so a negative
            // index from the end of the dictionary.
            // we calculate this before we change any of the buffers/dictionary, as the dictionary
            // might not be filled yet.
            let tuple_pos = window.back.len() - (pos);

            debug!(
                "total read (len+1): {}, dictionary.len = {}",
                len + 1,
                window.back.len()
            );

            let adv = len + 1;
            cursor += adv;
            // might be redundant, but if we are reading more than our dict can handle, we should
            // just clear the dictionary
            //window.prepare_advancement(adv);

            debug!(
                "changing buffer with [adv = {}] prestate: {:?}, left = {:?}",
                adv,
                window.front,
                stream.clone().collect_vec()
            );

            for _ in 0..adv {
                window.push_shift_optional_new(stream.next());
            }

            // emit a tuple, it is a bit easier if we do this after moving the buffers around,
            // especially for when our len is complex and we are performing wraparound.
            // If we generate the tuple after this and read the last from the dict, we should
            // have the correct next character.

            let tuple = CompressionTuple {
                pos: if len == 0 { 0 } else { tuple_pos },
                len,
                next: *window.back.last().unwrap(),
            };

            debug!(
                "{} [{:?}] | [{:?}] , stream left: {:?}",
                tuple, window.back, window.front, stream
            );

            output.push(tuple);
            // no more work to do.
            if window.front.is_empty() {
                break;
            }
        }

        window.front.clear();
        window.back.clear();

        Ok(output)
    }

    fn decompress(&self, input: Vec<CompressionTuple>) -> Result<String, ErrorTypes> {
        let mut out = String::new();
        let mut window = SlidingWindowArray::new(self._dictionary_size, self._window_size);

        debug!("### decompression loop! ###");
        let mut cursor = 0; // unused, debug info only
        for CompressionTuple { pos, len, next } in input {
            // relative position for the dictionary, will give the index that is at "-len"
            let relpos = window.back.len().saturating_sub(pos);

            debug!(
                "[cursor #{}] overall = {} ({}), with = {:?}, on ({}, {}, {}), rel pos = {}",
                cursor,
                out,
                out.len(),
                window.back,
                pos,
                len,
                next,
                relpos
            );
            if len == 0 {
                debug!(", simple op");
            } else if len + relpos <= window.back.len() {
                debug!(", reading dict simple");
                let window_copy = window.clone();
                let read = &window_copy.back[relpos..relpos + len];

                out.push_str(&String::from_iter(read));
                for r in read {
                    window.push_to_back(*r);
                }
            } else {
                debug!(", reading dict complex");
                let wrapped = window
                    .back
                    .iter()
                    .cycle()
                    .skip(relpos)
                    .take(len)
                    .collect::<String>();

                out.push_str(&wrapped);

                let rem_chars: Vec<char> = wrapped.chars().collect_vec();
                for r in rem_chars {
                    window.push_to_back(r);
                }
            }

            debug!("[cursor {}] out state {} <- {}", cursor, out, next);
            window.push_to_back(next);
            out.push(next);
            cursor += len + 1;
        }
        debug!("[FINAL] [Decompression] Output is {}", out);

        window.front.clear();
        window.back.clear();
        Ok(out)
    }
}

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    decompress: bool,

    #[arg(short, long, default_value_t = 30)]
    window_size: usize,

    #[arg(short = 'n', long, default_value_t = 26)]
    dict_size: usize,
}

fn main() -> Result<(), ErrorTypes> {
    let coder_cfg = config::standard()
        .with_little_endian()
        .with_variable_int_encoding();

    let args = Args::parse();
    let mut handle = BufWriter::new(std::io::stdout());
    let mut handle_in = BufReader::new(std::io::stdin());

    let mut program = LZ77Program::new(args.window_size, args.dict_size);

    if args.decompress {
        let result_in: Vec<CompressionTuple> =
            bincode::decode_from_std_read(&mut handle_in, coder_cfg)
                .expect("failure during decoding input!");
        let decomp = program.decompress(result_in)?;
        write!(&mut handle, "{}", decomp).expect("couldn't write to stdout!");
    } else {
        let target = std::io::read_to_string(handle_in).unwrap();
        let result = program.compress(&target)?;

        bincode::encode_into_std_write(result, &mut handle, coder_cfg)
            .expect("Failure during encoding!");
    }

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! build_tuples {
        [$(($p:literal, $l:literal, $c:literal)),*]=> {
             vec![$(CompressionTuple::new($p,$l,$c)),*]
        };
    }

    macro_rules! assert_output {
        ($e:expr, $t:expr) => {{
            let tmp = $t.iter().zip($e.iter());
            for (k, v) in tmp {
                assert_eq!(k, v);
            }
        }};
    }

    #[derive(Clone)]
    struct Setup {
        target_phrase: String,
        correct_tuple: Vec<CompressionTuple>,
        p: LZ77Program,
    }
    impl Setup {
        fn preinit() -> LZ77Program {
            let _ = env_logger::builder()
                .is_test(true)
                .filter_level(log::LevelFilter::Debug)
                .try_init();

            LZ77Program::new(30, 26)
        }
        fn new() -> Self {
            let p = Setup::preinit();
            Setup {
                target_phrase: String::from("abracadabrad"),
                correct_tuple: build_tuples![
                    (0, 0, 'a'),
                    (0, 0, 'b'),
                    (0, 0, 'r'),
                    (3, 1, 'c'),
                    (5, 1, 'd'),
                    (7, 4, 'd')
                ],
                p,
            }
        }

        fn new_alt() -> Self {
            let p = Setup::preinit();
            Setup {
                target_phrase: String::from("aacaacabcabaaac"),
                correct_tuple: build_tuples![
                    (0, 0, 'a'),
                    (1, 1, 'c'),
                    (3, 4, 'b'),
                    (3, 5, 'a'),
                    (0, 0, 'c')
                ],
                p,
            }
        }

        fn new_custom(phrase: &str) -> Self {
            let mut p = Setup::preinit();
            Self {
                target_phrase: String::from(phrase),
                // hopeful, this is more to test randomisation
                correct_tuple: p
                    .compress(phrase)
                    .expect("Trying to encode custom phrase failed"),
                p,
            }
        }
    }

    fn compress_fw(mut setup: Setup) -> Vec<CompressionTuple> {
        let result = setup
            .p
            .compress(&setup.target_phrase)
            .expect("Error while compressing");

        assert_output!(result, setup.correct_tuple);
        result
    }

    fn decompress_fw(setup: Setup) -> String {
        let result = setup
            .p
            .decompress(setup.correct_tuple)
            .expect("Could not decompress given tuple!");

        assert_eq!(result, setup.target_phrase);

        result
    }

    #[test]
    fn it_can_compress() {
        let setup = Setup::new();
        compress_fw(setup);
    }

    #[test]
    fn it_can_compress_alt() {
        let setup = Setup::new_alt();
        compress_fw(setup);
    }
    #[test]
    fn it_can_decompress() {
        let setup = Setup::new();
        decompress_fw(setup);
    }

    #[test]
    fn it_can_compress_and_decompress() {
        let setup = Setup::new();
        let result = compress_fw(setup.clone());
        let output = setup
            .p
            .decompress(result)
            .expect("Error attempting to decompress prior compressed input");

        assert_eq!(output, setup.target_phrase);
    }

    // from rust cookbook, meant for password generation
    // but will work for our input fuzzing.
    use rand::distributions::Alphanumeric;
    use rand::prelude::Distribution;
    use rand::{thread_rng, Rng};

    fn compress_n_fw<D>(mut p: LZ77Program, n: usize, chars_gen: &mut D, do_decomp: bool)
    where
        D: Iterator<Item = char>,
    {
        for _ in 0..n {
            let phrase_buf = chars_gen
                .next_chunk::<40>()
                .expect("Error with random number generation!");

            let phrase = phrase_buf.into_iter().collect::<String>();

            let comp_result = p.compress(&phrase);
            assert!(
                comp_result.is_ok(),
                "Compression of phrase {} failed!",
                phrase
            );
            let comp_result = comp_result.unwrap();
            if do_decomp {
                let decomp_result = p.decompress(comp_result);
                assert!(
                    decomp_result.is_ok(),
                    "Attempting to decompress phrase {} failed",
                    phrase
                );
                assert_eq!(phrase, decomp_result.unwrap());
            }
        }
    }

    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                        abcdefghijklmnopqrstuvwxyz\
                        0123456789)(*&^%$#@!~";

    macro_rules! random_n {
        ($n:tt) => {
            paste::paste! {
            #[test]
            fn [<it_can_compress_random_n_ $n >] () {
            let mut gen = thread_rng();
            let setup = Setup::new();


            let mut chars_gen = std::iter::repeat(())
                .map(|()| gen.sample(Alphanumeric))
                .map(char::from);

                compress_n_fw(setup.p, $n, &mut chars_gen, false);
            }

            #[test]
            fn [<it_can_decompress_n_ $n>]() {
                let setup = Setup::new();
                let mut gen = thread_rng();
                let mut chars_gen = std::iter::repeat(())
                    .map(|()| gen.sample(Alphanumeric))
                    .map(char::from);

                compress_n_fw(setup.p, $n, &mut chars_gen, true)
            }
            }
        };
    }

    random_n!(10);
    random_n!(30);
    random_n!(100);
    random_n!(500);

    use rand::distributions as dist;
    macro_rules! nonuniform_n {
        ($n:tt) => {
            paste::paste! {
                #[test]
                fn [<it_can_compress_nonuniform_n_ $n >] () {
                    let mut gen = thread_rng();
                    let setup = Setup::new();
                    let range_of: Vec<f32> = gen.clone().sample_iter(dist::Standard).take(12).collect_vec();
                    let biased_dist = dist::WeightedIndex::new(&range_of).unwrap();

                    let mut iter = std::iter::repeat(())
                        .map(|()| CHARSET[biased_dist.sample(&mut gen)+6])
                        .map(char::from);

                    compress_n_fw(setup.p,$n, &mut iter, false)
                }

                #[test]
                fn [<it_can_decompress_nonuniform_n_ $n >] () {
                    let mut gen = thread_rng();
                    let setup = Setup::new();
                    let range_of: Vec<f32> = gen.clone().sample_iter(dist::Standard).take(8).collect_vec();
                    let biased_dist = dist::WeightedIndex::new(&range_of).unwrap();

                    let mut iter = std::iter::repeat(())
                        .map(|()| CHARSET[biased_dist.sample(&mut gen)])
                        .map(char::from);

                    compress_n_fw(setup.p, $n, &mut iter, true)
                }
            }
        };
    }

    nonuniform_n!(10);
    nonuniform_n!(20);
    nonuniform_n!(30);
    nonuniform_n!(50);
    nonuniform_n!(100);

    #[test]
    fn it_can_encode_to_file() {}
}
