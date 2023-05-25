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

use mocap::{dist_3d, dist_3d_t, MoCapFile};
use mocap::{SensorFloat, SensorInt, SensorData, Triplet, Frame, Frames, Distances, Velocities, Accelerations};

#[macro_use] extern crate log;
extern crate simplelog;
use simplelog::*;

// =====================================================================
// Command line arguments.
// =====================================================================

#[derive(Parser, Debug, Clone)]
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
    
    let filename = &args.file;
    let file_size = std::fs::metadata(&filename)?.len();
    if file_size < 28 { // Arbitrary size... but to prevent creation of 0-byte files.
	error!("Error: Inputfile is too small!");
	std::process::exit(2);
    }

    let out_filename = if args.fileout.is_some() {
	args.clone().fileout.unwrap() // unwrap() to get the value, which we know exists.
    } else {
	create_outputfilename(&filename)
    };
    if !&args.force && Path::new(&out_filename).exists() == true {
	error!("Error: {} exists! Use --force to overwrite.", out_filename);
	std::process::exit(1); // Or use Ulid to generate a filename? (https://github.com/huxi/rusty_ulid)
    }

    // We have the filenames, create the structure.
    let mut mocap_file = MoCapFile{ filename: filename.clone(), out_filename: out_filename, ..Default::default() };
    
    info!("Reading header file {}", filename);
    parse_header(&mut mocap_file);
    
    info!("Header contains {} lines, {} matched.", mocap_file.num_header_lines, mocap_file.num_matches);
    info!("Expecting {} frames.", mocap_file.no_of_frames );
    
    let time_start = Instant::now();
    parse_data(&mut mocap_file, &args);
    let time_duration = time_start.elapsed().as_millis() + 1; // Add one to avoid division by zero.
    let lps = mocap_file.num_frames as u128 * 1000 / time_duration;
    
    info!("Ready, frames: {} (in {} ms)", mocap_file.num_frames, time_duration);
    info!("{} -> {}, {} l/s", mocap_file.filename, mocap_file.out_filename, lps);

    let time_start = Instant::now();
    let frames: Frames = read_frames(&mut mocap_file, &args);

    let distances: Distances = calculate_distances(&mocap_file, &frames);
    //println!("{:?}", distances);
    
    let time_duration = time_start.elapsed().as_millis() + 1; // Add one to avoid division by zero.
    let lps = mocap_file.num_frames as u128 * 1000 / time_duration;

    info!("Ready, frames: {} (in {} ms)", mocap_file.num_frames, time_duration);
    info!("{} -> {}, {} l/s", mocap_file.filename, mocap_file.out_filename, lps);

    for d in &distances {
	println!("{}", d.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
	println!("{}", d.iter().fold(-f32::INFINITY, |a, &b| a.max(b)));
    }
   
    let velocities: Velocities = calculate_velocities(&mocap_file, &distances);
    //println!("{:?}", velocities);

    for v in &velocities {
	println!("{}", v.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
	println!("{}", v.iter().fold(-f32::INFINITY, |a, &b| a.max(b)));
    }

    let accelerations: Accelerations = calculate_accelerations(&mocap_file, &velocities);
    //println!("{:?}", accelerations);

    for a in &accelerations {
	println!("{}", a.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
	println!("{}", a.iter().fold(-f32::INFINITY, |a, &b| a.max(b)));
    }

    let it = mocap_file.marker_names[0..3.min(mocap_file.marker_names.len())].iter();
    for (i, marker_name) in it.enumerate() {
	println!("{}", marker_name);
	let mut it_d = distances[i].iter();
	let mut it_v = velocities[i].iter();
	let mut it_a = accelerations[i].iter();
	for frame in &frames[0..4.min(frames.len())] {	    
	    let curr_triplet: &Triplet = &frame[i];
	    let curr_d = &it_d.next();
	    let curr_v = &it_v.next();
	    let curr_a = &it_a.next();
            println!("{:?} -> {:.3} {:.3} {:.3}", curr_triplet, curr_d.unwrap(), curr_v.unwrap(), curr_a.unwrap());
	}
    }

    if args.verbose {
	println!("{:?}", mocap_file);
    }
    
    Ok(())
}

/// Parse the header, fill in the fields in the struct.
fn parse_header(mocap_file: &mut MoCapFile) -> Result<()> {
    let filename = mocap_file.filename.clone();
    let file = File::open(&filename).expect("could not open file");
    let fileiter = std::io::BufReader::new(&file).lines();
    let mut line_no: usize = 0; // Counts lines in the file.
    let mut num_matches: usize = 0; // Counts regex matches.
    let mut bytes_read: u64 = 0;
    let time_start = Instant::now();
    
    for line in fileiter {
        if let Ok(l) = line {
            //println!("{}", l);
	    if l.len() < 1 { // In case of empty lines.
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
		    mocap_file.marker_names = x.to_vec();
		    num_matches += 1;
		}		
		if let Some(x) = mocap::extract_time_stamp(&l) {
		    mocap_file.time_stamp = x.to_string();
		    num_matches += 1;
		}		
		if let Some(x) = mocap::extract_description(&l) {
		    mocap_file.description = x.to_string();
		    num_matches += 1;
		}
		if let Some(x) = mocap::extract_data_included(&l) {
		    mocap_file.data_included = x.to_string();
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
    
    Ok(())
}

/// Parse the data (must be run after having parsed the header).
fn parse_data(mocap_file: &mut MoCapFile, args: &Args) -> Result<()> {
    let filename = mocap_file.filename.clone();
    let file = File::open(&filename).expect("could not open file");
    let mut fileiter = std::io::BufReader::new(file).lines();

    // Skip the header.
    info!("Skipping {} header lines.", mocap_file.num_header_lines);
    for _ in fileiter.by_ref().take(mocap_file.num_header_lines) {
	// print, save, show?
    }
    
    let out_filename = &mocap_file.out_filename;
    let mut file_out = File::create(out_filename).unwrap();
    let mut buffer_out = BufWriter::new(file_out);
    
    info!("Reading file {}", filename);
    info!("Writing file {}", out_filename);

    let mut line_no: usize = 0; // Counts line in the file.
    let mut frame_no: usize = 0; // Counts the lines with sensor data.
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
	    if ch.is_ascii_uppercase() {
		// this shouldn't happen
	    } 
	    else { // Assume we are in the data part.
		if !mocap_file.is_valid() {
		    error!("The file does not seem to contain a header!");
		    break; // This creates a zero byte file.
		}
		// If we requested a header, print it first (at this point we have not
		// written to the output file yet.
		//
		if !wrote_header {
		    let mut output_strs = Vec::new();
		    for marker_name in &mocap_file.marker_names {
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
		// We we requested to skip, we remove the first args.skip.
		if args.skip > 0 {
		    //let u: Vec<_> = bits.drain(0..args.skip).collect(); // Keep them, output them?
		    bits.drain(0..args.skip);
		}
		let num_bits = bits.len(); // Should be 3 * marker_names.len()
		let expected_num_bits = (mocap_file.no_of_markers * 3) as usize;
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
			    //println!("{} {:?} {:?} {}", frame_no, slice, prev_slice, dist);
			    //write!(file_out, "{}\t{}\t{}\t{}", slice[0], slice[1], slice[2], dist);
			    output_bits.extend_from_slice(&slice);
			    output_bits.push(dist);
			} else {
			    // No previous bits, the dist is 0 (our starting value).
			    prev_slice = &slice;
			    let dist = 0.0;
			    //println!("{} {:?} {:?} {}", frame_no, slice, prev_slice, dist);
			    //write!(file_out, "{}\t{}\t{}\t{}", slice[0], slice[1], slice[2], dist);
			    output_bits.extend_from_slice(&slice);
			    output_bits.push(dist);
			}
		    }
		    prev_bits = Some(bits);
		    frame_no += 1;
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
    mocap_file.num_frames = frame_no;
    
    Ok(())
}

fn read_frames(mocap_file: &mut MoCapFile, args: &Args) -> Frames {
    let filename = mocap_file.filename.clone();
    let file = File::open(&filename).expect("could not open file");
    let mut fileiter = std::io::BufReader::new(file).lines();

    // Skip the header.
    info!("Skipping {} header lines.", mocap_file.num_header_lines);
    for _ in fileiter.by_ref().take(mocap_file.num_header_lines) {
	// print, save, show?
    }
    
    info!("Reading file {}", filename);

    let mut line_no: usize = 0; // Counts line in the file.
    let mut frame_no: usize = 0; // Counts the lines with sensor data.
    let mut frames = Frames::new(); // or ::with_capacity(1000); 
    
    let time_start = Instant::now();
    
    for line in fileiter {
        if let Ok(l) = line {
	    if l.len() < 1 {
		continue;
	    }
	    let ch = &l.chars().take(1).last().unwrap(); // Surely, this could be simplified?!
	    if ch.is_ascii_uppercase() {
		// this shouldn't happen
	    } 
	    else { // Assume we are in the data part.
		//let bits: Vec<&str> = l.split("\t").collect();
		let mut bits = l.split("\t").
		    filter_map(
			|s| s.parse::<SensorFloat>().ok() // We assume all SensorFloat values for now.
		    ).collect::<Vec<SensorFloat>>();
		// We we requested to skip, we remove the first args.skip.
		if args.skip > 0 {
		    //let u: Vec<_> = bits.drain(0..args.skip).collect(); // Keep them, output them?
		    bits.drain(0..args.skip);
		}
		let num_bits = bits.len(); // Should be 3 * marker_names.len()
		let expected_num_bits = (mocap_file.no_of_markers * 3) as usize;
		if num_bits > expected_num_bits {
		    // Two extra fields could mean a frame number and frame time!
		    let num_extra = num_bits - expected_num_bits;
		    info!("Got {} extra fields in line {}, skipping (or use -s{})!",
			  num_extra, line_no, num_extra);
		} else if num_bits < expected_num_bits {
		    info!("Got {} ({}) missing fields in line {}, skip!",
			  expected_num_bits - num_bits, expected_num_bits, line_no);
		} else {
		    let mut triplets = Frame::new(); 
		    for triplet in (0..num_bits).step_by(3) { // Process per triple.
			let slice = &bits[triplet..triplet+3];
			let triplet: Triplet = bits[triplet..triplet+3].to_vec(); //vec![1.0, 2.0, 3.0];
			triplets.push(triplet);
		    }
		    frames.push(triplets);
		    frame_no += 1;
		}
	    } // If sensor data.
	    line_no += 1;
        }
    }
    mocap_file.num_frames = frame_no;
    info!("Frames {:?}, capacity {:?}", frames.len(), frames.capacity());
    
    frames
}

/// Calculates the distances on the in-memory data frame. Returns a
/// vector with a vector containing distances for each sensor. Indexed
/// by position in the marker_names vector.
fn calculate_distances(mocap_file: &MoCapFile, frames: &Frames) -> Distances {
    let mut dist = 0.0;
    let mut prev_triplet: Option<&Triplet> = None;
    let mut distances: Distances = vec![Vec::<SensorFloat>::new(); mocap_file.marker_names.len()]; // HashMap?

    // We can even reserve the size of the distance vectors...
    for v in &mut distances {
	v.reserve_exact(mocap_file.num_frames);
    }
	
    let it = mocap_file.marker_names.iter();
    for (i, marker_name) in it.enumerate() {
	info!("Calculating distances for {}", marker_name);
	
	dist = 0.0;
	prev_triplet = None;
	
	for frame in frames {
	    let curr_triplet: &Triplet = &frame[i];

	    if prev_triplet.is_some() { // Do we have a saved "previous line/triplet"?
		let x = prev_triplet.clone().unwrap();
		dist = dist_3d_t(&curr_triplet, &x);
	    }
	    distances[i].push(dist);
	    
	    //println!("{:?} {}", curr_triplet, dist); //, dist_3d_t(&curr_triplet, &prev_triplet));
	    prev_triplet = Some(curr_triplet);
	}
    }

    distances
}

/// Calculates the velocities on the supplied Distance data.
/// Returns a vector with a vector containing velocities for each sensor. Indexed
/// by position in the marker_names vector.
fn calculate_velocities(mocap_file: &MoCapFile, distances: &Distances) -> Velocities {
    let mut velocities: Velocities = vec![SensorData::new(); mocap_file.marker_names.len()]; 
	
    let it = mocap_file.marker_names.iter();
    for (i, marker_name) in it.enumerate() {
	info!("Calculating velocities for {}", marker_name);

	velocities[i].push(0.0); // Need to anchor wity 0.
	let mut result = distances[i].windows(2).map(|d| d[1] - d[0]).collect::<Vec<SensorFloat>>();
	velocities[i].append(&mut result);
    }

    velocities
}

/// Calculates the accelerations on the supplied Acceleration data.
/// Returns a vector with a vector containing accelerations for each sensor. Indexed
/// by position in the marker_names vector.
fn calculate_accelerations(mocap_file: &MoCapFile, velocities: &Velocities) -> Accelerations {
    let mut accelerations: Accelerations = vec![SensorData::new(); mocap_file.marker_names.len()]; 
	
    let it = mocap_file.marker_names.iter();
    for (i, marker_name) in it.enumerate() {
	info!("Calculating accelerations for {}", marker_name);

	accelerations[i].push(0.0); // Need to anchor wity 0.
	let mut result = velocities[i].windows(2).map(|d| d[1] - d[0]).collect::<Vec<SensorFloat>>();
	accelerations[i].append(&mut result);
    }

    accelerations
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
