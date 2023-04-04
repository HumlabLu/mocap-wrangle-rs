use color_eyre::{Result};
use polars::prelude::*;
use reqwest::blocking::Client;
use std::io::Cursor;
//use std::io::BufReader
use std::io::{BufRead};
use std::fs::File;
use std::env;
use rand::Rng;
use regex::{Regex, RegexSet};
use clap::Parser;

#[macro_use] extern crate log;
extern crate simplelog;
use simplelog::*;

#[derive(Parser, Debug)]
#[command(author="Peter Berck <peter.berck@humlab.lu.se>",
	  version="0.1.0",
	  about="Scans TSv MoCap data.",
	  name="mocap",
	  long_about = None)]

/*
Two command line arguments:
  file: scan this XML file.
  number: don't scan more than this number.
*/
struct Args {
    /// Filename
    #[arg(short, long, default_value_t = String::from("street_adapt_1tsv"))]
    file: String, // Path thingy?
    
    /// Number of pages to scan
    #[arg(short, long, default_value_t = 0, help = "Max pages to scan (0 for all)")]
    number: usize,
}

fn main() -> Result<()> {
    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Warn, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Info, Config::default(), File::create("my_rust_binary.log").unwrap()),
        ]
    ).unwrap();
    error!("Bright red error");
    info!("This only appears in the log file");
    debug!("This level is currently not enabled for any logger");
    
    let args = Args::parse();
    eprintln!("{:?}", args); // or: dbg!(&args);
    
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

    
    // =================================================

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

    let re_frames = Regex::new(r"NO_OF_FRAMES\t(\d+)").unwrap();
    let re_cameras = Regex::new(r"NO_OF_CAMERAS\t(\d+)").unwrap();
    let re_markers = Regex::new(r"NO_OF_MARKERS\t(\d+)").unwrap();
    let re_marker_names = Regex::new(r"MARKER_NAMES\t(.+)").unwrap();
    //TIME_STAMP	2022-06-03, 10:47:36.627	94247.45402301
    //TIME_STAMP	2022-11-22, 22:00:35
    let re_time_stamp = Regex::new(r"TIME_STAMP\t(.+?)(\t(.+)|\z)").unwrap();

    let filename = args.file;
    let file = File::open(csv_path).expect("could not open file");
    let fileiter = std::io::BufReader::new(file).lines();
    println!("Reading file");
    let mut line_no: usize = 0;

    let mut myfile = MoCapFile {
	name: csv_path.to_string(),
	no_of_frames: 0,
	no_of_cameras: 0,
	no_of_markers: 0,
	frequency: 200,
	no_of_analog: 0,
	description: String::new(),
	time_stamp: String::new(),
	data_included: String::new(),
	marker_names: vec![],
    };

    for line in fileiter {
        if let Ok(l) = line {
            //println!("{}", l);

	    if line_no < 12 { 
		test_re(&l);

		// Some matching for testing.
		match re_frames.captures(&l) {
		    Some(caps) => {
			let cap = caps.get(1).unwrap().as_str();
			let cap_int = cap.parse::<u32>().unwrap(); 
			println!("cap '{}'", cap_int);
			myfile.no_of_frames = cap_int;
		    }
		    None => {
			// The regex did not match.
		    }
		}		
		match re_cameras.captures(&l) {
		    Some(caps) => {
			let cap = caps.get(1).unwrap().as_str();
			let cap_int = cap.parse::<u16>().unwrap(); 
			println!("cap '{}'", cap_int);
			myfile.no_of_cameras = cap_int;
		    }
		    None => {
			// The regex did not match.
		    }
		}
		match re_markers.captures(&l) {
		    Some(caps) => {
			let cap = caps.get(1).unwrap().as_str();
			let cap_int = cap.parse::<u16>().unwrap(); 
			println!("cap '{}'", cap_int);
			myfile.no_of_markers = cap_int;
		    }
		    None => {
			// The regex did not match.
		    }
		}
		match re_marker_names.captures(&l) {
		    Some(caps) => {
			let cap = caps.get(1).unwrap().as_str();
			//println!("cap '{}'", cap);
			let seperator = Regex::new(r"(\t)").expect("Invalid regex");
			// Split, convert to String, iterate and collect.
			let splits: Vec<_> = seperator.split(cap).map(|s| s.to_string()).into_iter().collect();
			//println!( "{:?}", splits );
			myfile.marker_names = splits; // Move it here.
		    }
		    None => {
			// No match.
		    }
		}
		match re_time_stamp.captures(&l) {
		    Some(caps) => {
			println!("caps {:?}", caps);
			let cap = caps.get(1).unwrap().as_str();
		    }
		    None => {
			// No match.
		    }
		}

	    }
	    
	    let bits: Vec<&str> = l.split("\t")
		.collect();
	    //println!("{:?}", bits);

	    line_no += 1;
        }
    }
    println!("read file");
    println!("{:?}", myfile);
    
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
TIME_STAMP	2022-11-22, 22:00:35
DATA_INCLUDED	3D
MARKER_NAMES	x_HeadL	x_HeadTop	x_HeadR	x_HeadFront	x_LShoulderTop	
TRAJECTORY_TYPES	Measured	Measured
23.2 34.34 ... DATA
*/

#[derive(Debug, Clone)]
struct MoCapFile {
    pub name: String,
    pub no_of_frames: u32,
    pub no_of_cameras: u16,
    pub no_of_markers: u16,
    pub frequency: u16,
    pub no_of_analog: u8,
    pub description: String,
    pub time_stamp: String,
    pub data_included: String,
    pub marker_names: Vec<String>,
}

fn test_re(line: &str) {
    let re_frames = Regex::new(r"NO_OF_FRAMES\t(\d+)").unwrap();
    let re_cameras = Regex::new(r"NO_OF_CAMERAS\t(.+)").unwrap();
    let re_markers = Regex::new(r"NO_OF_MARKERS\t(.+)").unwrap();
    let re_time_stamp = Regex::new(r"TIME_STAMP\t(.+)").unwrap();

    match re_frames.captures(line) {
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
