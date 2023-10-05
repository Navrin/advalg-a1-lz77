#![feature(iter_next_chunk, buf_read_has_data_left)]

extern crate arrayvec;
extern crate itertools;

use arrayvec::ArrayVec;
use itertools::Itertools;
use std::{
    borrow::BorrowMut,
    error::Error,
    fmt::Display,
    io::{stdout, BufRead, BufReader, BufWriter, Read, Write},
    str::from_utf8,
};

macro_rules! ut {
    () => {
        u8
    };
    (x) => {
        1
    };
}

#[derive(Debug, Copy, Clone)]
struct CompressionTuple {
    pos: ut!(),
    len: ut!(),
    next: char,
}

impl Display for CompressionTuple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.pos, self.len, self.next)
    }
}

macro_rules! println {
    ($($rest:tt)*) => {
        if std::env::var("DEBUG").is_ok() {
            std::println!($($rest)*);
        }
    }
}

const WINDOW_SIZE: ut!() = 30;
const WINDOW_SIZE_HELPER: usize = WINDOW_SIZE as usize;
const DICTIONARY_SIZE: ut!() = 10;
const DICTIONARY_SIZE_HELPER: usize = DICTIONARY_SIZE as usize;

#[derive(Debug)]
enum ErrorTypes {
    Undersized,
    BufferOverread,
    InitialUndersized,
    DictionaryUndersized(String),
}
impl Display for ErrorTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str("");
    }
}

trait Coder {
    fn encode<W: Write>(self, out: &mut BufWriter<W>);
    fn decode<R: Read>(arr: &mut BufReader<R>) -> Self;
    const DELIM: [u8; 1] = [b'\x1f'];
    const END: [u8; 1] = [b'\x1e'];
}

impl Coder for CompressionTuple {
    fn encode<W: Write>(self, out: &mut BufWriter<W>) {
        out.write_all(&self.pos.to_ne_bytes()).unwrap();
        //out.write_all(&Self::DELIM);
        out.write_all(&self.len.to_ne_bytes()).unwrap();
        //out.write_all(&Self::DELIM);

        let mut tmp: [u8; 4] = [0; 4];
        self.next.encode_utf8(&mut tmp);
        out.write_all(&tmp);
        //out.write_all(&Self::END);
    }

    fn decode<R: Read>(arr: &mut BufReader<R>) -> Self {
        let mut buf: Vec<u8> = Vec::new();
        let sized = std::mem::size_of::<ut!()>();

        let mut num_buf: [u8; ut!(x)] = [0; ut!(x)];
        arr.read_exact(&mut num_buf);
        let pos = <ut!()>::from_ne_bytes(num_buf);

        arr.read_exact(&mut num_buf);
        let len = <ut!()>::from_ne_bytes(num_buf);

        let mut char_buf: [u8; 4] = [0; 4];
        arr.read_exact(&mut char_buf);
        let next = from_utf8(&char_buf).unwrap().chars().next().unwrap();

        CompressionTuple { pos, len, next }
    }
}

impl Coder for Vec<CompressionTuple> {
    fn encode<W: Write>(self, out: &mut BufWriter<W>) {
        for tpl in self {
            tpl.encode(out);
        }
    }

    fn decode<R: Read>(arr: &mut BufReader<R>) -> Self {
        let mut out: Self = Vec::new();
        while arr.has_data_left().unwrap() {
            let res = CompressionTuple::decode(arr);
            out.push(res);
        }
        out
    }
}

impl<T> From<arrayvec::CapacityError<T>> for ErrorTypes {
    fn from(value: arrayvec::CapacityError<T>) -> Self {
        return ErrorTypes::DictionaryUndersized(String::from(value.to_string()));
    }
}

impl Error for ErrorTypes {}

fn shift_push<T, const CAP: usize>(array: &mut ArrayVec<T, CAP>, value: T) {
    if (array.remaining_capacity() == 0) {
        array.pop_at(0);
    }
    array.push(value);
}
fn shift_push_many<T: Copy, const CAP: usize>(array: &mut ArrayVec<T, CAP>, values: &[T]) {
    for val in values {
        shift_push(array, *val)
    }
}

fn find_match(search: &[char], dict: &[char]) -> Option<(ut!(), ut!())> {
    fn find_match_inner(search: &[char], dict: &[char]) -> Option<(ut!(), ut!())> {
        let len = search.len();
        //        let extras = &dict.get(0..len);
        //
        //        let dict_wraparound = extras
        //            .map(|ext| [dict.to_vec(), ext.repeat(len).to_vec()].concat())
        //            .unwrap_or(dict.to_vec());
        let repeated = dict.repeat(len);
        let mut views = repeated.windows(len);

        //println!(
        //    "wraparound dict windows = {:?}",
        //    views
        //        .clone()
        //        .map(|d| d.to_vec())
        //        .collect::<Vec<Vec<char>>>()
        //);

        for pos in 0..views.len() {
            if let Some(view) = views.next() {
                if view == search {
                    return Some((pos as ut!(), len as ut!()));
                }
            } else {
                continue;
            }
        }

        None
    }

    let mut longest_match = None;
    println!("Initiating search with = {:?}", search);
    for i in 1..search.len() + 1 {
        let result = find_match_inner(&search[0..i], dict);
        match result {
            Some(v) => {
                println!("{:?} -> Found result = {:?}", &search[0..i], result);
                longest_match = Some(v);
            }
            None => {
                return longest_match;
            }
        }
    }

    longest_match
}

fn compress(input: &str) -> Result<Vec<CompressionTuple>, ErrorTypes> {
    let total_len = input.len();
    let mut output = Vec::new();
    let mut dictionary = ArrayVec::<char, DICTIONARY_SIZE_HELPER>::new();
    let annotated_chars: Vec<(usize, char)> = input
        .char_indices()
        .map(|(idx, c)| (total_len - idx - 1, c))
        .collect_vec();
    let mut stream = annotated_chars.iter();

    let mut buf: ArrayVec<char, WINDOW_SIZE_HELPER> = stream
        .borrow_mut()
        .map(|(idx, char)| *char)
        .next_chunk::<WINDOW_SIZE_HELPER>()
        .map_err(|_| ErrorTypes::InitialUndersized)?
        .into();

    //let mut last = CompressionTuple {
    //    pos: 0,
    //    len: 0,
    //    next: buf[0],
    //};

    //output.push(last);
    //dictionary.push(*buf.first().ok_or(ErrorTypes::UndersizedError)?);
    let mut cursor = 0;
    loop {
        println!("### ITER AT CURSOR {} ###", cursor);

        // check for matches
        let (pos, len) = find_match(&buf, &dictionary).unwrap_or((0, 0));
        println!(
            "det pos, cursor = {}, dict.len = {}, pos = {}",
            cursor,
            dictionary.len(),
            pos
        );
        let tuple_pos = dictionary.len() as ut!() - (pos);

        println!(
            "total read (len+1): {}, dictionary.len = {}",
            len + 1,
            dictionary.len()
        );

        let adv = len as usize + 1;
        cursor += adv;

        if (adv > DICTIONARY_SIZE_HELPER) {
            dictionary.clear();
        } else if (dictionary.remaining_capacity() < adv) {
            let dr = dictionary
                .drain(..adv - dictionary.remaining_capacity())
                .collect::<String>();
            println!("drain result = {}", dr);
        }

        println!(
            "changing buffer with [adv = {}] prestate: {:?}, left = {:?}",
            adv,
            buf,
            stream.clone().collect_vec()
        );
        for _ in 0..adv {
            let rem = buf.pop_at(0);
            if let Some(r) = rem {
                dictionary.push(r);
            }

            let next = stream.next();
            if let Some((_, c)) = next {
                buf.push(*c);
            }
        }

        // dictionary.try_extend_from_slice(&buf[0..std::cmp::min(adv, WINDOW_SIZE)])?;

        let tuple = CompressionTuple {
            pos: tuple_pos,
            len,
            next: *dictionary.last().unwrap(),
        };

        output.push(tuple);
        println!(
            "{} [{:?}] | [{:?}] , stream left: {:?}",
            tuple, dictionary, buf, stream
        );

        if buf.is_empty() {
            break;
        }
    }

    return Ok(output);
}

fn decompress(input: Vec<CompressionTuple>) -> Result<String, ErrorTypes> {
    let mut out = String::new();
    let mut dictionary: ArrayVec<char, DICTIONARY_SIZE_HELPER> = ArrayVec::new();

    println!("### decompression loop! ###");
    let mut cursor: ut!() = 0;
    for CompressionTuple { pos, len, next } in input {
        let relpos = dictionary.len().checked_sub(pos as usize).unwrap_or(0);

        print!(
            "[cursor #{}] overall = {} ({}), with = {:?}, on ({}, {}, {}), rel pos = {}",
            cursor,
            out,
            out.len(),
            dictionary,
            pos,
            len,
            next,
            relpos
        );
        if len == 0 {
            print!(", simple op");
        } else if (len as usize + relpos <= dictionary.len()) {
            print!(", reading dict simple");
            let dict_temp = dictionary.clone();
            let read = dict_temp
                .get(relpos..relpos + len as usize)
                .ok_or(ErrorTypes::DictionaryUndersized(String::from("Failure")))?;

            out.push_str(&String::from_iter(read));
            shift_push_many(&mut dictionary, read);
        } else {
            print!(", reading dict complex");
            let wrapped = dictionary
                .clone()
                .iter()
                .cycle()
                .skip(relpos)
                .take(len as usize)
                .collect::<String>();
            //let rem = dict_temp
            //    .get(relpos..)
            //    .ok_or(ErrorTypes::DictionaryUndersized(String::from("Failure")))?;
            //let wrapped = rem.iter().cycle().take(len).collect::<String>();
            out.push_str(&wrapped);

            let rem_chars: Vec<char> = wrapped.chars().collect_vec();
            shift_push_many(&mut dictionary, &rem_chars);
        }
        print!("\n");
        shift_push(&mut dictionary, next);
        out.push(next);
        cursor += len + 1;
    }

    Ok(out)
}

fn main() -> Result<(), ErrorTypes> {
    //let target = "aacaacabcabaaac";
    //let target = "deffeghhifeddefghiifffgiggh";
    let target = std::io::read_to_string(std::io::stdin()).unwrap();
    let result = compress(&target)?;
    // for c in result.clone() {
    //   print!("{} ", c);
    //}
    let mut handle = BufWriter::new(std::io::stdout());
    result.encode(&mut handle);

    //println!("\nattempting to decompress {}", target);
    //let mut handle_in = BufReader::new(std::io::stdin());
    //let result_in: Vec<CompressionTuple> = Coder::decode(&mut handle_in);

    //let decomp = decompress(result_in)?;
    //println!("decompressed: {}, valid?={}", decomp, decomp == target);
    //write!(&mut handle, "{}", decomp);
    Ok(())
}
