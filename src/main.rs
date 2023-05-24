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
use mocap::{SensorFloat, SensorInt};

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
/// The file is not read into memory, it reads one line, processes it and then
/// writes it to the new file.
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
    // Read file line by line and calculate d3D
    //
    // Space requirements...
    // Let's say we have 128 sensors per line, so 384 values.
    // Each value is a f32 (SensorFloat in lib.rs), 4 bytes.
    // 384 * 4 = 1136 bytes per line.
    // 1 minute = 60 seconds x 200 frames, this takes
    // 1136 * 200 * 60 = 13,632,000 = 13 MB per minute
    // 30 minutes = 390 MB in memory (plus overhead).
    // About 1 GB per hour of video.
    // =====================================================================
    
    let filename = args.file;
    let file_size = std::fs::metadata(&filename)?.len();
    if file_size < 28 { // Arbitrary size... but to prevent creation of 0-byte files.
	error!("Error: Inputfile is too small!");
	std::process::exit(2);
    }
    
    info!("Reading header file {}", filename);
    let mocap_file = parse_header(&filename).unwrap();
    info!("Header contains {} lines, {} matched.", mocap_file.num_header_lines, mocap_file.num_matches);
    
    let file = File::open(&filename).expect("could not open file");
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
    let mut myfile = MoCapFile { name: filename, ..Default::default() };
    
    let mut line_no: usize = 0; // Counts line in the file.
    let mut num_matches:usize = 0; // Counts the number of matches regexs.
    let mut frames_no: usize = 0; // Counts the lines with sensor data.
    let mut prev_bits: Option<Vec<SensorFloat>> = None; // Previous line used in calculations.
    let mut prev_slice: &[SensorFloat] = &[0.0, 0.0, 0.0]; // Previous X, Y and Z coordinates.
    let mut wrote_header = !args.header; // If we specify --header, wrote_header becomes false.
    let mut output_bits = Vec::<SensorFloat>::new(); // Sensor values as SensorFloat.
    
    let time_start = Instant::now();
    
    for line in fileiter {
        if let Ok(l) = line {
            //Println!("{}", l);
	    if l.len() < 1 {
		continue;
	    }
	    let ch = &l.chars().take(1).last().unwrap(); // Surely, this could be simplified?!
	    if ch.is_ascii_uppercase() { // Assume we are still parsing the header.
		if args.verbose {
		    info!("{}", l);
		}
		// Some matching for testing.
		if let Some(x) = mocap::extract_no_of_frames(&l) {
		    myfile.no_of_frames = x;
		    num_matches += 1;
		}
		if let Some(x) = mocap::extract_no_of_cameras(&l) {
		    myfile.no_of_cameras = x;
		    num_matches += 1;
		}
		if let Some(x) = mocap::extract_no_of_markers(&l) {
		    myfile.no_of_markers = x;
		    num_matches += 1;
		}		
		if let Some(x) = mocap::extract_frequency(&l) {
		    myfile.frequency = x;
		    num_matches += 1;
		}
		if let Some(x) = mocap::extract_marker_names(&l) {
		    myfile.marker_names = x;
		    num_matches += 1;
		}		
		if let Some(x) = mocap::extract_time_stamp(&l) {
		    myfile.time_stamp = x;
		    num_matches += 1;
		}		
		if let Some(x) = mocap::extract_description(&l) {
		    myfile.description = x;
		    num_matches += 1;
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

// What we want is a parse_header() function returning a MoCap structure, and then a
// separate read_data function.

fn parse_header(filename: &String) -> Option<MoCapFile> {
    let file = File::open(&filename).expect("could not open file");
    let fileiter = std::io::BufReader::new(&file).lines();
    let mut mocap_file = MoCapFile { name: filename.clone(), ..Default::default() };
    let mut line_no: usize = 0; // Counts lines in the file.
    let mut num_matches: usize = 0; // Counts regex matches.
    let mut bytes_read: u64 = 0;
    let time_start = Instant::now();
    
    for line in fileiter {
        if let Ok(l) = line {
            //println!("{}", l);
	    if l.len() < 1 {
		continue;
	    }
	    let ch = &l.chars().take(1).last().unwrap(); // Surely, this could be simplified?!
	    if ch.is_ascii_uppercase() { // Assume we are parsing the header.
		// Some matching for testing.
		if let Some(x) = mocap::extract_no_of_frames(&l) {
		    mocap_file.no_of_frames = x;
		    num_matches += 1;
		}
		if let Some(x) = mocap::extract_no_of_cameras(&l) {
		    mocap_file.no_of_cameras = x;
		    num_matches += 1;
		}
		if let Some(x) = mocap::extract_no_of_markers(&l) {
		    mocap_file.no_of_markers = x;
		    num_matches += 1;
		}		
		if let Some(x) = mocap::extract_frequency(&l) {
		    mocap_file.frequency = x;
		    num_matches += 1;
		}
		if let Some(x) = mocap::extract_marker_names(&l) {
		    mocap_file.marker_names = x;
		    num_matches += 1;
		}		
		if let Some(x) = mocap::extract_time_stamp(&l) {
		    mocap_file.time_stamp = x;
		    num_matches += 1;
		}		
		if let Some(x) = mocap::extract_description(&l) {
		    mocap_file.description = x;
		    num_matches += 1;
		}
		line_no += 1;
	    } else {
		break;
	    }
	}
    }
    mocap_file.num_header_lines = line_no;
    mocap_file.num_matches = num_matches;
    
    Some(mocap_file)
}

fn read_and_write(args: Args) {
    let filename = args.file;
    let file_size = std::fs::metadata(&filename).unwrap().len();
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
    let mut file_out = File::create(&outfilename).unwrap();
    let mut buffer_out = BufWriter::new(file_out);
    
    info!("Reading file {}", filename);
    info!("Writing file {}", outfilename);

    // Metadata structure.
    let mut myfile = MoCapFile { name: filename, ..Default::default() };
    
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
		if let Some(x) = mocap::extract_no_of_frames(&l) {
		    myfile.no_of_frames = x;
		}
		if let Some(x) = mocap::extract_no_of_cameras(&l) {
		    myfile.no_of_cameras = x;
		}
		if let Some(x) = mocap::extract_no_of_markers(&l) {
		    myfile.no_of_markers = x;
		}		
		if let Some(x) = mocap::extract_frequency(&l) {
		    myfile.frequency = x;
		}
		if let Some(x) = mocap::extract_marker_names(&l) {
		    myfile.marker_names = x;
		}		
		if let Some(x) = mocap::extract_time_stamp(&l) {
		    myfile.time_stamp = x;
		}		
		if let Some(x) = mocap::extract_description(&l) {
		    myfile.description = x;
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
			    buffer_out.write(b"\t").unwrap(); // If not at start, write a tab.
			}
			buffer_out.write_fmt(format_args!("{}", value)).unwrap();
		    }
		    buffer_out.write(b"\n").unwrap();
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
			    buffer_out.write(b"\t").unwrap();
			}
			buffer_out.write_fmt(format_args!("{:.3}", value)).unwrap(); // Note the ".3"!
		    }
		    buffer_out.write(b"\n").unwrap();
		    output_bits.clear();
		}
	    } // If sensor data.
	    line_no += 1;
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    
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
}
