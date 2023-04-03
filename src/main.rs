use color_eyre::{Result};
use polars::prelude::*;
use reqwest::blocking::Client;
use std::io::Cursor;
//use std::io::BufReader
use std::io::{self, BufRead};
use std::fs::File;
use std::env;
use rand::Rng;
use regex::{Regex, RegexSet};

fn main() -> Result<()> { 
    let data: Vec<u8> = Client::new()
        .get("https://j.mp/iriscsv")
        .send()?
        .text()?
        .bytes()
        .collect();

    let df = CsvReader::new(Cursor::new(data))
        .has_header(true)
        .finish()?
        .lazy()
        .filter(col("sepal_length").gt(5))
        .groupby([col("species")])
        .agg([col("*").sum()])
        .collect()?;

    println!("{:?}", df);

    for arg in env::args() {
	println!( "{arg}" )
	    //numbers.push(u64::from_str(&arg)
	//	     .expect("error parsing argument"));
    }

    
    // =================================================

    //let csv_path = "/Users/pberck/Development/MoCap/Neural/eaf_NewAnno.csv";
    let csv_path = "street_adapt_1.tsv"; //"./eaf_NewAnno.csv";

    /*
    Let df = CsvReader::from_path(csv_path)
	.unwrap()
	.finish()
	.unwrap();
    println!("{:?}", df);

    let df = LazyCsvReader::new(&csv_path).finish().unwrap();
     */

    //panic!("crash and burn");
    
    // https://docs.rs/polars/0.0.1/polars/frame/csv/struct.CsvReader.html

    // =====================================================================
    // Stuff
    // =====================================================================
    let re_set = RegexSet::new(&[
	r"NO_OF_FRAMES\t(.*?)",
	r"NO_OF_CAMERAS\t(.*?)",
	r"NO_OF_MARKERS\t(.*?)",
    ]).expect("Error compiling RegexSet");
    // .case_insensitive(true)
    eprintln!("{:?}", re_set);
    let mut regex_hits = vec![0, 0, 0]; // Count which ones we match
    //let zero_vec = vec![0; len];
    
    let file = File::open(csv_path).expect("could not open file");
    let fileiter = std::io::BufReader::new(file).lines();
    println!("Reading file");
    let mut line_no: usize = 0;
    for line in fileiter {
        if let Ok(l) = line {
            //println!("{}", l);

	    if line_no < 12 { 
		test_re(&l);
	    }
	    
	    let bits: Vec<&str> = l.split("\t")
		.collect();
	    //println!("{:?}", bits);

	    let matches: Vec<_> = re_set.matches(&l) // &clean was &l
		.into_iter()
		.collect();
	    if matches.contains(&2) {
		eprintln!("{}", &l);
		
	    }
	    if matches.len() > 0 {
		println!("{:?}", matches);
		// Count
		matches.iter().for_each(|&index| {
		    *regex_hits.get_mut(index).unwrap() += 1;
		});
		eprintln!("{} {:?}", line_no, regex_hits);
	    }
	    line_no += 1;
        }
    }
    println!("read file");
    
    let file1 = File::open(csv_path).expect("could not open file");
    let foo = CsvReader::new(file1)
        .infer_schema(None)
        .has_header(true)
	.with_delimiter(9) // ascii 9 is TAB
        .finish()?;
	//.expect("reading error"); // was .unwrap()
    println!( "{:?}", foo.schema() );

    //foo.as_single_chunk_par();
    
    let mut iters = foo.columns(["x_LHandOut_dX", "x_LHandOut_dY", "x_LHandOut_dZ"])? //.expect("no column")
	.iter()
	.map( |s| s.iter() )
	.collect::<Vec<_>>();
    
    for row in 0..3 { //foo.height() {
	println!( "ROW {}", row );
	for iter in &mut iters {
            let value = iter
		.next()
		.expect("should have as many iterations as rows");
            // process value
	    println!( "{:.8}", value );
	}
    }

    let bar = make_rnd(28, 42);
    println!( "{:?}", bar );

    let q = "Petrus";
    println!( "{:?}", q );
    let s = vec!["udon".to_string(), "ramen".to_string(),
		 "soba".to_string()];
    println!( "{:?}", s );
	
    Ok(())
}

/*
(VENV) (base) pberck@Peters-MacBook-Pro mocap % head -n12 street_adapt_1.tsv
NO_OF_FRAMES	16722
NO_OF_CAMERAS	20
NO_OF_MARKERS	64
FREQUENCY	200
NO_OF_ANALOG	0
ANALOG_FREQUENCY	0
DESCRIPTION	--
TIME_STAMP	2022-06-03, 10:47:36.627	94247.45402301
DATA_INCLUDED	3D
MARKER_NAMES	x_HeadL	x_HeadTop	x_HeadR	x_HeadFront	x_LShoulderTop	
TRAJECTORY_TYPES	Measured	Measured
23.2 34.34 ... DATA
*/

#[derive(Debug)]
struct MoCapFile {
    name: String,
    no_of_frames: u32,
    no_of_cameras: u16,
    no_of_markers: u16,
    frequency: u16,
    no_of_analog: u8,
    description: String,
    time_stamp: String,
    data_included: String,
    marker_names: Vec<String>,
}

fn test_re(line: &str) {
    let re_FRAMES = Regex::new(r"NO_OF_FRAMES\t(\d+)").unwrap();
    let re_CAMERAS = Regex::new(r"NO_OF_CAMERAS\t(.*?)").unwrap();
    let re_MARKERS = Regex::new(r"NO_OF_MARKERS\t(.*?)").unwrap();

    match re_FRAMES.captures(line) {
	Some(caps) => {
            let cap = caps.get(1).unwrap().as_str();
	    let cap_int = cap.parse::<u32>().unwrap(); 
            println!("cap '{}'", cap_int);
	}
	None => {
            // The regex did not match. Deal with it here!
	}
    }

    /*
    match re_FRAMES.captures(line) {
	Some(caps) => {
	    let no_frames = caps.get(1).unwrap();
	    let foo = no_frames.as_str().parse::<i32>().unwrap(); 
	},
	None => ()
}
    */
    //let no_frames = caps.get(1).map_or("0", |m| m.as_str().trim() ); //parse::<i32>());
    //println!( "{:?}", no_frames );

    /*re_FRAMES.captures(line).and_then(|cap| {
        cap.get(1).map(|var| var.as_str().trim().parse::<i32>())
    });*/
/*
    if let Some(captures) = re_FRAMES.captures(&l) {
	let a_title = captures.get(1).unwrap().as_str();
	//eprintln!("{}", a_title);
    }
*/
}

////
fn parse(line: &str) -> Option<(i32, i32, i32, i32)> {
    let re_str = concat!(
        r"^\s+(?P<qrw1>\d+)\|(?P<qrw2>\d+)",//qrw 0|0
        r"\s+(?P<arw1>\d+)\|(?P<arw2>\d+)",//arw 34|118
    );
    let re = Regex::new(re_str).unwrap();
    match re.captures(line) {
        Some(caps) => {
            let internal_parse = |key| {
                caps.name(key).unwrap().as_str().parse::<i32>().unwrap()
            };
            let qrw1 = internal_parse("qrw1");
            let qrw2 = internal_parse("qrw2");
            let arw1 = internal_parse("arw1");
            let arw2 = internal_parse("arw2");
            Some((qrw1, qrw2, arw1, arw2))
        }
        None => None,
    }
}
////

fn make_rnd(n: u64, m: u64) -> (u64, u64) {
    
    let mut rng = rand::thread_rng();
    let n0: u64 = rng.gen();
    let n1 = rng.gen_range(n.. m);
    println!("Random u64: {}", n0);
    (n0, n1)
}

#[test]
fn test_make_rnd() {
    for i in 0..10 {
	let (foo, bar) = make_rnd(28, 42);
	assert!( bar >= 28 && bar <= 42 );
    }
}
