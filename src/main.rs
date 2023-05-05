#![allow(unused)] // Remove me for release build
use color_eyre::{Result};
use reqwest::blocking::Client;
use std::io::{Cursor, BufRead, BufWriter, Write, Seek, SeekFrom};
use std::fs::{OpenOptions, File, remove_file};
use std::env;
use rand::Rng;
use regex::{Regex, RegexSet};
use clap::Parser;
use std::path::Path;
use std::time::{Duration, Instant};

use mocap::{dist_3d, MoCapFile};
use mocap::SensorFloat;

#[macro_use] extern crate log;
extern crate simplelog;
use simplelog::*;

// =====================================================================
// Command line arguments.
// =====================================================================

#[derive(Parser, Debug)]
struct Args {
    // Filename
    #[arg(short, long, default_value_t = String::from("street_adapt_1.tsv"))]
    file: String, // Path thingy?

    #[arg(short = 'o', long, help="Output filename (auto-generated if unspecified).")]
    fileout: Option<String>,

    // Extra output
    #[clap(long, short, action, help = "Produce superfluous output.")]
    verbose: bool,

    // Extra output
    #[clap(long, short, action, default_value_t = 0, help = "Skip first n columns in sensor data.")]
    skip: usize,

    // Header output
    #[clap(long, action, help = "Output header row.")]
    header: bool,

    // Force overwrite of output
    #[clap(long, action, help = "Overwrite output if it exists.")]
    force: bool,
}

// =====================================================================
// Main.
// =====================================================================

/// Reads the MoCap file with sensor data.
///
/// Expects a header, followed by numeric data. Values should be tab-separated.
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
    let re_time_stamp = Regex::new(r"TIME_STAMP\t(.+?)(\t(.+)|\z)").unwrap();
    let re_description = Regex::new(r"DESCRIPTION\t(.+)").unwrap();
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
	error!("Error: {} exists! Use --force to overwrite.", outfilename);
	std::process::exit(1);
	// Or use Ulid to generate a filename? (https://github.com/huxi/rusty_ulid)
    }
    let mut file_out = File::create(&outfilename)?;
    let mut buffer_out = BufWriter::new(file_out);
	
    info!("Reading file {}", filename);
    info!("Writing file {}", outfilename);

    // Metadata structure.
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
    
    let mut line_no: usize = 0; // Counts line in the file.
    let mut frames_no: usize = 0; // Counts the lines with sensor data.
    let mut prev_bits: Option<Vec<SensorFloat>> = None; // Previous line used in calculations.
    let mut prev_slice: &[SensorFloat] = &[0.0, 0.0, 0.0]; // Previous X, Y and Z coordinates.
    let mut wrote_header = !args.header; // If we specify --header, wrote_header becomes false.
    let mut output_bits = Vec::<SensorFloat>::new(); // Sensor values as SensorFloat.
    
    let time_start = Instant::now();
    
    for line in fileiter {
        if let Ok(l) = line {
            //println!("{}", l);
	    if l.len() < 1 {
		continue;
	    }
	    let ch = &l.chars().take(1).last().unwrap(); // Surely, this could be simplified?!
	    if ch.is_ascii_uppercase() { // Assume we are still parsing the header.
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
		// These are to be used again if we request the header in the output.
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
			myfile.time_stamp = cap.to_string();
		    }
		    None => {
			// No match.
		    }
		}
		match re_description.captures(&l) {
		    Some(caps) => {
			//println!("caps {:?}", caps);
			let cap = caps.get(1).unwrap().as_str();
			myfile.description = cap.to_string();
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

	    } 
	    else { // Assume we are in the data part.
		if !myfile.is_valid() {
		    error!("The file does not seem to contain a header!");
		    break; // This creates a zero byte file.
		}
		// If we requested a header, print it first (at this point we have not
		// written to the output file yet.
		//
		if !wrote_header {
		    let mut output_strs = Vec::new();
		    for marker_name in &myfile.marker_names {
			for marker_type in vec!["_X", "_Y", "_X", "_d3D"] {
			    //print!("{}{}", marker_name, marker_type );
			    output_strs.push(format!("{}{}", marker_name, marker_type));
			}
		    }
		    for (i, value) in output_strs.iter().enumerate() {
			if i > 0 {
			    buffer_out.write(b"\t")?; // If not at start, write a tab.
			}
			buffer_out.write_fmt(format_args!("{}", value))?;
		    }
		    buffer_out.write(b"\n")?;
		    wrote_header = true;
		}
		    
		//let bits: Vec<&str> = l.split("\t").collect();
		let mut bits = l.split("\t").
		    filter_map(
			|s| s.parse::<SensorFloat>().ok() // We assume all SensorFloat values for now.
		    ).collect::<Vec<_>>();
		if args.skip > 0 {
		    //let u: Vec<_> = bits.drain(0..args.skip).collect(); // Keep them, output them?
		    bits.drain(0..args.skip);
		}
		let num_bits = bits.len(); // Should be 3 * marker_names.len()
		let expected_num_bits = (myfile.no_of_markers * 3) as usize;
		if num_bits > expected_num_bits {
		    // Two extra fields could mean a frame number and frame time!
		    let num_extra = num_bits - expected_num_bits;
		    info!("Got {} extra fields in line {}, skipping (or use -s{})!",
			  num_extra, line_no, num_extra);
		} else if num_bits < expected_num_bits {
		    info!("Got {} ({}) missing fields in line {}, skip!",
			  expected_num_bits - num_bits, expected_num_bits, line_no);
		} else {
		    //let mut output_bits = Vec::new(); // Collect and save values at the end.
		    for triplet in (0..num_bits).step_by(3) { // Process per triple.
			let slice = &bits[triplet..triplet+3];
			if prev_bits.is_some() { // Do we have a saved "previous line/triplet"?
			    let x = prev_bits.clone().unwrap();
			    prev_slice = &x[triplet..triplet+3];
			    let dist = dist_3d(slice, prev_slice);
			    //println!("{} {:?} {:?} {}", frames_no, slice, prev_slice, dist);
			    //write!(file_out, "{}\t{}\t{}\t{}", slice[0], slice[1], slice[2], dist);
			    output_bits.extend_from_slice(&slice);
			    output_bits.push(dist);
			} else {
			    // No previous bits, the dist is 0 (our starting value).
			    prev_slice = &slice;
			    let dist = 0.0;
			    //println!("{} {:?} {:?} {}", frames_no, slice, prev_slice, dist);
			    //write!(file_out, "{}\t{}\t{}\t{}", slice[0], slice[1], slice[2], dist);
			    output_bits.extend_from_slice(&slice);
			    output_bits.push(dist);
			}
		    }
		    prev_bits = Some(bits);
		    frames_no += 1;
		    for (i, value) in output_bits.iter().enumerate() {
			if i > 0 {
			    buffer_out.write(b"\t")?;
			}
			buffer_out.write_fmt(format_args!("{:.3}", value))?; // Note the ".3"!
		    }
		    buffer_out.write(b"\n")?;
		    output_bits.clear();
		}
	    } // If sensor data.
	    line_no += 1;
        }
    }
    let time_duration = time_start.elapsed().as_millis() + 1; // Add one to avoid division by zero.
    let lps = frames_no as u128 * 1000 / time_duration; 
    info!("Ready, lines:{} data:{} (in {} ms)", line_no, frames_no, time_duration);
    
    let bytes_written = buffer_out.into_inner()?.seek(SeekFrom::Current(0))?;
    if bytes_written == 0 {
	error!("Created file is 0 bytes, removing it!");
	remove_file(&outfilename)?;
    } else {
	info!("{} -> {}, {} l/s", myfile.name, outfilename, lps);
    }
    
    if myfile.no_of_frames as usize != frames_no {
	error!("Error, did not read the specified number ({}) of frames.", myfile.no_of_frames);
    }

    // The metadata.
    //myfile.no_of_frames = frames_no as u32; // Maybe not.
    println!("{}", myfile);
    
    Ok(())
}

// Should be moved back, does not belong in lib.rs
/// Create a new output filename, tries to append "_d3D" to
/// the filename.
///
/// Short input filenames will return `output_d3D.tsv`.
///
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

// =====================================================================
// Tests.
// =====================================================================

#[test]
fn test_zero_dist() {
    let dist = dist_3d(&[0.0,0.0,0.0], &[0.0,0.0,0.0]);
    assert!(dist==0.0);
}

#[test]
fn test_normal_dist() {
    let dist = dist_3d(&[1.0,0.0,0.0], &[0.0,0.0,0.0]);
    assert!(dist==1.0);
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
	frequency: 200,
	no_of_analog: 0,
	description: String::new(),
	time_stamp: String::new(),
	data_included: String::new(),
	marker_names: vec!["X".to_string()],
    };
    assert!(myfile.is_valid()==true);
}

#[test]
fn struct_is_invalid() {
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

