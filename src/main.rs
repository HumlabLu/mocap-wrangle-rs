#![allow(unused)] // Remove me for release build
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
use std::path::Path;
use std::time::{Duration, Instant};

#[macro_use] extern crate log;
extern crate simplelog;
use simplelog::*;

// =====================================================================
// Command line arguments.
// =====================================================================

#[derive(Parser, Debug)]
struct Args {
    /// Filename
    #[arg(short, long, default_value_t = String::from("street_adapt_1.tsv"))]
    file: String, // Path thingy?

    #[arg(short = 'o', long, help="Output filename (auto-generated if unspecified).")]
    fileout: Option<String>,

    /// Number of lines to scan
    #[arg(short, long, default_value_t = 0, help = "Max pages to scan (0 for all).")]
    number: usize,

    // Extra output
    #[clap(long, short, action, help = "Produce superfluous output.")]
    verbose: bool,

    // Force overwrite of output
    #[clap(long, action, help = "Overwrite output if it exists.")]
    force: bool,
}

// =====================================================================
// Main.
// =====================================================================

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
			     .create(true) // To allow creating the file, if it doesn't exist,
			     .append(true) // do not truncate the file, but instead add to it.
			     .open("mocap.log")
			     .unwrap()
	    )
        ]
    ).unwrap();
    
    let args = Args::parse();
    info!("{:?}", args);
        
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
    let re_frequency = Regex::new(r"^FREQUENCY\t(\d+)").unwrap();
    
    let filename = args.file;
    let file_size = std::fs::metadata(&filename)?.len();
    if file_size < 28 { // Arbitrary size... but to prevent creation of 0-byte files.
	error!("Error: Inputfile is too small!");
	std::process::exit(2);
    }
    let file = File::open(filename.clone()).expect("could not open file");
    let fileiter = std::io::BufReader::new(file).lines();

    let outfilename = if args.fileout.is_some() {
	args.fileout.unwrap() // unwrap() to get the value, which we know exists.
    } else {
	create_outputfilename(&filename)
    };
    if !args.force && Path::new(&outfilename).exists() == true {
	error!("Error: {} exists!", outfilename);
	std::process::exit(1);
	// Or use Ulid to generate a filename? (https://github.com/huxi/rusty_ulid)
    }
    let mut file_out = File::create(&outfilename)?;
    let mut buffer_out = BufWriter::new(file_out);
	
    info!("Reading file {}", filename);
    info!("Writing file {}", outfilename);
    
    let mut line_no: usize = 0;
    let mut data_no: usize = 0;
    
    let mut myfile = MoCapFile {
	name: filename,
	no_of_frames: 0,
	no_of_cameras: 0,
	no_of_markers: 0,
	frequency: 0,
	no_of_analog: 0,
	description: String::new(),
	time_stamp: String::new(),
	data_included: String::new(),
	marker_names: vec![],
    };

    let mut prev_bits: Option<Vec<f32>> = None;
    let mut prev_slice: &[f32] = &[0.0, 0.0, 0.0];

    let time_start = Instant::now();
    
    for line in fileiter {
        if let Ok(l) = line {
            //println!("{}", l);

	    let ch = &l.chars().take(1).last().unwrap(); // Surely, this could be simplified?!
	    if ch.is_ascii_uppercase() {
		//if line_no < 12 { // Arbitrary, fix me, maybe count fields? (loop while !is_valid()? numbits < 3?)
		if args.verbose {
		    info!("{}", l);
		}
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
		match re_frequency.captures(&l) {
		    Some(caps) => {
			let cap = caps.get(1).unwrap().as_str();
			let cap_int = cap.parse::<u16>().unwrap(); 
			//println!("cap '{}'", cap_int);
			myfile.frequency = cap_int;
		    }
		    None => {
			// The regex did not match.
		    }
		}

	    } // if line_no < 12 (but we need a better check)
	    else {
		if !myfile.is_valid() {
		    error!("The file does not seem to contain a header!");
		    break;
		}
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
		    info!("Got {} ({}) missing fields in line {}, skip!",
			  expected_num_bits - num_bits, expected_num_bits, line_no);
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
			buffer_out.write_fmt(format_args!("{:.3}", value))?; // Note the ".3"!
		    }
		    buffer_out.write(b"\n")?;
		    output_bits.clear();
		}
	    } // if line_no >= 12
	    line_no += 1;
        }
    }
    let time_duration = time_start.elapsed().as_millis();
    let lps = data_no as u128 * 1000 / time_duration; // as usize;
    info!("read file, lines:{} data:{}", line_no, data_no);
    info!("{} -> {:?}, {} l/s", myfile.name, outfilename, lps);
	
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

fn create_outputfilename(filename: &str) -> String {
    let len = filename.len();
    if len > 4 { 
	let suffix = &filename[len-4..len]; // Also test for ".tsv" suffix.
	if suffix == ".tsv" {
	    format!("{}{}", &filename[0..len-4], "_d3D.tsv")
	} else {
	    format!("{}{}", &filename, "_d3D.tsv")
	}
    } else {
	"output_d3D.tsv".to_string()
    }
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

impl MoCapFile {
    fn is_valid(&self) -> bool {
        if self.frequency > 0 {
	    true
	} else {
	    false
	}
    }
}

// =====================================================================
// Tests.
// =====================================================================

#[test]
fn test_zero_dist() {
    let dist = dist_3d(&[0.0,0.0,0.0], &[0.0,0.0,0.0]);
    assert!(dist==0.0f32);
}

#[test]
fn test_normal_dist() {
    let dist = dist_3d(&[1.0,0.0,0.0], &[0.0,0.0,0.0]);
    assert!(dist==1.0f32);
}

#[test]
#[should_panic]
fn test_wrong_params_lhs() {
    let dist = dist_3d(&[1.0,0.0,0.0,4.0], &[0.0,0.0,0.0]);
}

#[test]
fn test_dist_wrong_params_rhs() {
    let result = std::panic::catch_unwind(|| dist_3d(&[1.0,0.0,0.0], &[0.0,0.0,0.0,4.0]));
    assert!(result.is_err()); 
}

#[test]
fn filename_normal() {
    let result = create_outputfilename("filename.tsv");
    assert!(result=="filename_d3D.tsv");
}

#[test]
fn filename_short() {
    let result = create_outputfilename("");
    assert!(result=="output_d3D.tsv");
}

#[test]
fn filename_four_chars() {
    let result = create_outputfilename("abcd");
    assert!(result=="output_d3D.tsv");
}

#[test]
fn filename_five_chars() {
    let result = create_outputfilename("a.tsv");
    assert!(result=="a_d3D.tsv");
}

#[test]
fn filename_no_tsv() {
    let result = create_outputfilename("abcde");
    assert!(result=="abcde_d3D.tsv");
}

#[test]
fn struct_is_valid() {
    let mut myfile = MoCapFile {
	name: String::new(),
	no_of_frames: 0,
	no_of_cameras: 0,
	no_of_markers: 0,
	frequency: 0,
	no_of_analog: 0,
	description: String::new(),
	time_stamp: String::new(),
	data_included: String::new(),
	marker_names: vec![],
    };
    assert!(myfile.is_valid()==false);
}
