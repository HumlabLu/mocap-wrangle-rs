use color_eyre::{Result};
use polars::prelude::*;
use reqwest::blocking::Client;
use std::io::Cursor;
//use std::io::BufReader
use std::io::{BufRead};
use std::io::{BufWriter, Write};
use std::fs::OpenOptions;
use std::fs::File;
use std::env;
use rand::Rng;
use regex::{Regex, RegexSet};
use clap::Parser;

#[macro_use] extern crate log;
extern crate simplelog;
use simplelog::*;

#[derive(Parser, Debug)]
/*
Two command line arguments:
  file: scan this XML file.
  number: don't scan more than this number.
*/
struct Args {
    /// Filename
    #[arg(short, long, default_value_t = String::from("street_adapt_1.tsv"))]
    file: String, // Path thingy?
    
    /// Number of pages to scan
    #[arg(short, long, default_value_t = 0, help = "Max pages to scan (0 for all)")]
    number: usize,
}

fn main() -> Result<()> {
    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Info,
			    Config::default(),
			    TerminalMode::Mixed,
			    ColorChoice::Auto
	    ),
            WriteLogger::new(LevelFilter::Info,
			     Config::default(),
			     OpenOptions::new()
			     .create(true) // to allow creating the file, if it doesn't exist
			     .append(true) // to not truncate the file, but instead add to it
			     .open("mocap.log")
			     .unwrap()
	    )
        ]
    ).unwrap();
    
    let args = Args::parse();
    info!("{:?}", args);
        
    // =================================================
    
    /*
    let csv_path = "street_adapt_1.tsv"; //"./eaf_NewAnno.csv";
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
    // Read file line by line and caclulate d3D
    // =====================================================================

    let re_frames = Regex::new(r"NO_OF_FRAMES\t(\d+)").unwrap();
    let re_cameras = Regex::new(r"NO_OF_CAMERAS\t(\d+)").unwrap();
    let re_markers = Regex::new(r"NO_OF_MARKERS\t(\d+)").unwrap();
    let re_marker_names = Regex::new(r"MARKER_NAMES\t(.+)").unwrap();
    //TIME_STAMP	2022-06-03, 10:47:36.627	94247.45402301
    //TIME_STAMP	2022-11-22, 22:00:35
    let re_time_stamp = Regex::new(r"TIME_STAMP\t(.+?)(\t(.+)|\z)").unwrap();

    let filename = args.file;
    let file = File::open(filename.clone()).expect("could not open file");
    let fileiter = std::io::BufReader::new(file).lines();

    let path = "results.txt";
    let mut file_out = File::create(path)?;
    let mut buffer_out = BufWriter::new(file_out);
	
    println!("// =====================================================================");
    println!("// Reading file");
    println!("// =====================================================================");
    
    let mut line_no: usize = 0;
    let mut data_no: usize = 0;
    
    let mut myfile = MoCapFile {
	name: filename,
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

    let mut prev_bits: Option<Vec<f32>> = None;
    let mut prev_slice: &[f32] = &[0.0, 0.0, 0.0];
    
    for line in fileiter {
        if let Ok(l) = line {
            //println!("{}", l);

	    if line_no < 12 { 
		//test_re(&l);

		// Some matching for testing.
		match re_frames.captures(&l) {
		    Some(caps) => {
			let cap = caps.get(1).unwrap().as_str();
			let cap_int = cap.parse::<u32>().unwrap(); 
			//println!("cap '{}'", cap_int);
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
			//println!("cap '{}'", cap_int);
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
			//println!("cap '{}'", cap_int);
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
			//println!("caps {:?}", caps);
			let cap = caps.get(1).unwrap().as_str();
		    }
		    None => {
			// No match.
		    }
		}

	    } // if line_no < 12
	    else {
		//let bits: Vec<&str> = l.split("\t").collect();
		let bits = l.split("\t").
		    filter_map(
			|s| s.parse::<f32>().ok() // We assume f32 for now.
		    ).collect::<Vec<_>>();
		let num_bits = bits.len(); // Should be 3 * marker_names.len()
		let expected_num_bits = (myfile.no_of_markers * 3) as usize;
		if num_bits > expected_num_bits {
		    info!("Got {} extra fields in line {}, skip!", num_bits - expected_num_bits, line_no);
		} else if num_bits < expected_num_bits {
		    info!("Got {} missing fields in line {}, skip!", expected_num_bits - num_bits, line_no);
		} else {
		    let mut output_bits = Vec::new(); // Collect and save values at the end.
		    for triplet in (0..num_bits).step_by(3) { // Process per triple.
			let slice = &bits[triplet..triplet+3];
			if prev_bits.is_some() { // Do we have a saved "previous line/triplet"?
			    let x = prev_bits.clone().unwrap();
			    prev_slice = &x[triplet..triplet+3];
			    let dist = dist_3d(slice, prev_slice);
			    //println!("{} {:?} {:?} {}", data_no, slice, prev_slice, dist);
			    //write!(file_out, "{}\t{}\t{}\t{}", slice[0], slice[1], slice[2], dist);
			    output_bits.extend_from_slice(&slice);
			    output_bits.push(dist);
			} else {
			    // No previous bits, the dist is 0 (our starting value).
			    prev_slice = &slice;
			    let dist = 0.0;
			    //println!("{} {:?} {:?} {}", data_no, slice, prev_slice, dist);
			    //write!(file_out, "{}\t{}\t{}\t{}", slice[0], slice[1], slice[2], dist);
			    output_bits.extend_from_slice(&slice);
			    output_bits.push(dist);
			}
		    }
		    prev_bits = Some(bits);
		    data_no += 1;
		    //write!(file_out, "{:?}\n", output_bits);
		    for (i, value) in output_bits.iter().enumerate() {
			if i > 0 {
			    buffer_out.write(b"\t")?;
			}
			buffer_out.write_fmt(format_args!("{:.2}", value))?;
		    }
		    buffer_out.write(b"\n")?;
		    output_bits.clear();
		}
	    } // if line_no >= 12
	    line_no += 1;
        }
    }
    info!("read file, lines:{} data:{}", line_no, data_no);
    println!("{:?}", myfile.name);
    println!("{:?}", myfile.num_markers());
	
    Ok(())
}

fn dist_3d(coords0: &[f32], coords1: &[f32]) -> f32 {
    assert!(coords0.len() == 3);
    assert!(coords1.len() == 3);
    /*
    let x_diff = coords0[0] - coords1[0];
    let y_diff = coords0[1] - coords1[1];
    let z_diff = coords0[2] - coords1[2];
    let distance_squared = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
    distance_squared.sqrt()
    */
    let squared_sum = coords0.iter()
        .zip(coords1.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .fold(0.0, |acc, x| acc + x);
    squared_sum.sqrt()
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

impl MoCapFile {
    fn num_markers(&self) -> usize {
        self.marker_names.len()
    }
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
	assert!(bar >= 28 && bar <= 42);
    }
}

#[test]
fn test_dist0() {
    let dist = dist_3d(&[0.0,0.0,0.0], &[0.0,0.0,0.0]);
    assert!(dist==0.0f32);
}

#[test]
fn test_dist1() {
    let dist = dist_3d(&[1.0,0.0,0.0], &[0.0,0.0,0.0]);
    assert!(dist==1.0f32);
}
