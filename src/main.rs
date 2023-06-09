#![allow(unused)] // Remove me for release build
use clap::Parser;
use color_eyre::Result;
use rand::Rng;
use regex::{Regex, RegexSet};
use reqwest::blocking::Client;
use std::env;
use std::fs::{remove_file, File, OpenOptions};
use std::io::{BufRead, BufWriter, Cursor, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use mocap::{dist_3d, dist_3d_t, Calculated, MoCapFile};
use mocap::{
    Accelerations, Distances, Frame, Frames, SensorData, SensorFloat, SensorInt, Triplet,
    Velocities,
};

#[macro_use]
extern crate log;
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

    #[arg(
        short = 'o',
        long,
        help = "Output filename (auto-generated if unspecified)."
    )]
    fileout: Option<String>,

    // Extra output
    #[clap(long, short, action, help = "Produce superfluous output.")]
    verbose: bool,

    // Output position coordinates
    #[clap(
        long,
        short,
        action,
        help = "Include X, Y and Z coordinates in output."
    )]
    coords: bool,

    // Skip fields
    #[clap(
        long,
        short,
        action,
        default_value_t = 0,
        help = "Skip first n columns in sensor data."
    )]
    skip: usize,

    // Step for readinf frames
    #[clap(long, action, default_value_t = 1, help = "Read frames in steps.")]
    framestep: usize,

    // Header output
    #[clap(long, action, help = "Do not output header row.")]
    noheader: bool,

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
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Info,
            Config::default(),
            TerminalMode::Stderr, // was Mixed
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Info,
            Config::default(),
            OpenOptions::new()
                .create(true) // To allow creating the file, if it doesn't exist,
                .append(true) // do not truncate the file, but instead add to it.
                .open("mocap.log")
                .unwrap(),
        ),
    ])
    .unwrap();

    let args = Args::parse();
    info!("{:?}", args);

    /*
    let t0 = vec![0.0, 0.0, 0.0];
    let t1 = vec![1.0, 0.0, 0.0];
    let result = mocap::calculate_azimuth_inclination(&t0, &t1);
    info!("{:?}", result);
    let t0 = vec![0.0, 0.0, 0.0];
    let t1 = vec![1.0, 0.0, 100.0];
    let result = mocap::calculate_azimuth_inclination(&t0, &t1);
    info!("{:?}", result);
    let t0 = vec![0.0, 0.0, 0.0];
    let t1 = vec![1.0, 0.0, -100.0];
    let result = mocap::calculate_azimuth_inclination(&t0, &t1);
    info!("{:?}", result);
    */
    /*
    let t0 = vec![0.0, 0.0, 0.0];
    let t1 = vec![-(3.0_f32.sqrt())/2.0, -0.5, 3.0_f32.sqrt()];
    let result = mocap::calculate_azimuth_inclination(&t0, &t1);
    info!("{:?}", result);

    let t0 = vec![0.0, 0.0, 0.0];
    let t1 = vec![-1.0, 1.0, 6.0_f32.sqrt()];
    let result = mocap::calculate_azimuth_inclination(&t0, &t1);
    info!("{:?}", result);
    let pi = std::f32::consts::PI;
    info!("{:?}", (2.0*2.0_f32.sqrt(), (3.0*pi/4.0).to_degrees(), (pi/6.0).to_degrees()));
     */

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
    if file_size < 28 {
        // Arbitrary size... but to prevent creation of 0-byte files.
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
    let mut mocap_file = MoCapFile {
        filename: filename.clone(),
        out_filename: out_filename,
        ..Default::default()
    };

    info!("Reading header file {}", filename);
    parse_header(&mut mocap_file);

    info!(
        "Header contains {} lines, {} matched.",
        mocap_file.num_header_lines, mocap_file.num_matches
    );
    info!("Expecting {} frames.", mocap_file.no_of_frames);

    // Without a header, we abort.
    if (mocap_file.num_header_lines == 0) || (mocap_file.num_matches == 0) {
        error!("File contains no header!");
        std::process::exit(3);
    }

    let time_start = Instant::now();
    let frames: Frames = read_frames(&mut mocap_file, &args);
    //mocap_file.add_frames(frames);
    let time_duration = time_start.elapsed().as_millis() + 1; // Add one to avoid division by zero.
    let lps = mocap_file.num_frames as u128 * 1000 / time_duration;

    info!(
        "Ready, frames: {} (in {} ms, {} l/s)",
        mocap_file.num_frames, time_duration, lps
    );

    info!("Calculating distances.");
    mocap_file.calculate_distances();
    let distances: Distances = calculate_distances(&mocap_file, &frames);
    //let distances: &Distances = mocap_file.calculated.unwrap().distances.unwrap().as_ref();
    //println!("{:?}", distances);

    info!("Calculating velocities.");
    let velocities: Velocities = calculate_velocities(&mocap_file, &distances);
    //println!("{:?}", velocities);

    info!("Calculating acceleratioons.");
    let accelerations: Accelerations = calculate_accelerations(&mocap_file, &velocities);
    //println!("{:?}", accelerations);

    info!("Calculating angles.");
    let (azimuths, inclinations): (Distances, Distances) = calculate_angles(&mocap_file, &frames);
    //println!("{:?}", azimuths);
    //println!("{:?}", inclinations);

    // Move them into the structure.
    // Should add the Frames here too, and Impl some of the functions
    // for the structure... Ideally all in the MoCapFile structure.b
    let mut calculated = Calculated {
        //distances: Some(distances),
        velocities: Some(velocities),
        accelerations: Some(accelerations),
        ..Default::default()
    };

    info!("Calculating min/max.");
    calculated.calculate_min_distances();
    calculated.calculate_max_distances();
    calculated.calculate_min_velocities();
    calculated.calculate_max_velocities();
    calculated.calculate_min_accelerations();
    calculated.calculate_max_accelerations();
    calculated.calculate_mean_distances();
    calculated.calculate_stdev_distances();
    calculated.calculate_mean_velocities();
    calculated.calculate_stdev_velocities();
    calculated.calculate_mean_accelerations();
    calculated.calculate_stdev_accelerations();

    /// Output to std out.
    info!("Outputting data.");
    let time_start = Instant::now();

    let mut distances = calculated.distances.as_ref().unwrap(); // distances, per sensor!!!
    let mut velocities = calculated.velocities.as_ref().unwrap();
    let mut accelerations = calculated.accelerations.as_ref().unwrap();

    if args.noheader == false {
        for (i, marker_name) in mocap_file.marker_names.iter().enumerate() {
            if i > 0 {
                print!("\t"); // Separator, but not at start/end.
            }
            // We need a "output fields for sensor" function, taking args to output relevant header.
	    emit_header(&marker_name, args.coords);
        }
        println!();
    }

    let f_it = frames.iter();
    for (frame_no, frame) in f_it.enumerate() {
        // Skip the first one (normalising the 0's doesn't make sense?
        if frame_no == 0 {
            //continue;
        }
        let it = mocap_file.marker_names.iter(); // Match to include?
        for (sensor_id, marker_name) in it.enumerate() {
            // The sensor_id-th column of triplets (a sensor)
            let the_triplet = &frame[sensor_id];

	    let the_d = calculated.get_distance(sensor_id, frame_no);
            let min_d = calculated.get_min_distance(sensor_id);
            let max_d = calculated.get_max_distance(sensor_id);
	    
	    let mean_d = calculated.get_mean_distance(sensor_id);
            let stdev_d = calculated.get_stdev_distance(sensor_id);
            let nor_d = mocap::normalise_minmax(&the_d, &min_d, &max_d);
            let std_d = mocap::standardise(&the_d, &mean_d, &stdev_d); // First one should really be 0.0?

            let the_v = calculated.get_velocity(sensor_id, frame_no);
            let min_v = calculated.get_min_velocity(sensor_id);
            let max_v = calculated.get_max_velocity(sensor_id);

            let mean_v = calculated.get_mean_velocity(sensor_id);
            let stdev_v = calculated.get_stdev_velocity(sensor_id);
            let nor_v = mocap::normalise_minmax(&the_v, &min_v, &max_v);
            let std_v = mocap::standardise(&the_v, &mean_v, &stdev_v); // First one should really be 0.0?

            let the_a = calculated.get_acceleration(sensor_id, frame_no);
            let min_a = calculated.get_min_acceleration(sensor_id);
            let max_a = calculated.get_max_acceleration(sensor_id);

            let mean_a = calculated.get_mean_acceleration(sensor_id);
            let stdev_a = calculated.get_stdev_acceleration(sensor_id);
            let nor_a = mocap::normalise_minmax(&the_a, &min_a, &max_a);
            let std_a = mocap::standardise(&the_a, &mean_a, &stdev_a); // First one should really be 0.0?

            let azim = azimuths.get(sensor_id).unwrap().get(frame_no).unwrap();
            let incl = inclinations.get(sensor_id).unwrap().get(frame_no).unwrap();

            if sensor_id > 0 {
                print!("\t");
            }
            if args.coords == true {
                print!("{:.2}\t{:.2}\t{:.2}\t{:.1}\t{:.1}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
		       the_triplet.get(0).unwrap(), the_triplet.get(1).unwrap(), the_triplet.get(2).unwrap(),
		       azim, incl, the_d, nor_d, std_d,
		       the_v, nor_v,
		       the_a, nor_a
		);
            } else {
                print!(
                    "{:.1}\t{:.1}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
                    azim, incl, the_d, nor_d, std_d, the_v, nor_v, std_v, the_a, nor_a, std_a
                );
            }
        }
        println!();
    }
    let time_duration = time_start.elapsed().as_millis() + 1; // Add one to avoid division by zero.
    let lps = mocap_file.num_frames as u128 * 1000 / time_duration;

    info!(
        "Ready, frames: {} (in {} ms, {} l/s)",
        mocap_file.num_frames, time_duration, lps
    );

    if args.verbose {
        println!("{:?}", mocap_file);
    }

    Ok(())
}

/// Parse the header, fill in the fields in the struct.
// Implement this in MoCapFile instead
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
            if l.len() < 1 {
                // In case of empty lines.
                continue;
            }
            let ch = &l.chars().take(1).last().unwrap(); // Surely, this could be simplified?!
            if ch.is_ascii_uppercase() {
                // Assume we are parsing the header.
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
fn parse_data_deprecated(mocap_file: &mut MoCapFile, args: &Args) -> Result<()> {
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
    let mut wrote_header = !args.noheader; // If we specify --header, wrote_header becomes false.
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
            } else {
                // Assume we are in the data part.
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
                let mut bits = l
                    .split("\t")
                    .filter_map(
                        |s| s.parse::<SensorFloat>().ok(), // We assume all SensorFloat values for now.
                    )
                    .collect::<Vec<_>>();
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
                    info!(
                        "Got {} extra fields in line {}, skipping (or use -s{})!",
                        num_extra, line_no, num_extra
                    );
                } else if num_bits < expected_num_bits {
                    info!(
                        "Got {} ({}) missing fields in line {}, skip!",
                        expected_num_bits - num_bits,
                        expected_num_bits,
                        line_no
                    );
                } else {
                    //let mut output_bits = Vec::new(); // Collect and save values at the end.
                    for triplet in (0..num_bits).step_by(3) {
                        // Process per triple.
                        let slice = &bits[triplet..triplet + 3];
                        if prev_bits.is_some() {
                            // Do we have a saved "previous line/triplet"?
                            let x = prev_bits.clone().unwrap();
                            prev_slice = &x[triplet..triplet + 3];
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
                        buffer_out.write_fmt(format_args!("{:.3}", value)).unwrap();
                        // Note the ".3"!
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

/// Read the data into memory. Can only be run after parse_header(...).
/// Returns a Frames structure containing a vector with vectors with triplets.
fn read_frames(mocap_file: &mut MoCapFile, args: &Args) -> Frames {
    let filename = mocap_file.filename.clone();
    info!("Reading file {} into memory", filename);
    let file = File::open(&filename).expect("could not open file");
    let mut fileiter = std::io::BufReader::new(file).lines();

    // Skip the header.
    info!("Skipping {} header lines.", mocap_file.num_header_lines);
    for _ in fileiter.by_ref().take(mocap_file.num_header_lines) {
        // print, save, show?
    }

    let mut line_no: usize = 0; // Counts line in the file.
    let mut frame_no: usize = 0; // Counts the lines with sensor data.
    let mut frames = Frames::with_capacity(mocap_file.no_of_frames as usize);

    let time_start = Instant::now();

    // Using a framestep > 1 can be used to "smooth" the data (kind of).
    for line in fileiter.step_by(args.framestep) {
        if let Ok(l) = line {
            if l.len() < 1 {
                continue;
            }
            let ch = &l.chars().take(1).last().unwrap(); // Surely, this could be simplified?!
            if ch.is_ascii_uppercase() {
                // this shouldn't happen
            } else {
                // Assume we are in the data part.
                //let bits: Vec<&str> = l.split("\t").collect();
                let mut bits = l
                    .split("\t")
                    .filter_map(
                        |s| s.parse::<SensorFloat>().ok(), // We assume all SensorFloat values for now.
                    )
                    .collect::<Vec<SensorFloat>>();
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
                    info!(
                        "Got {} extra fields in line {}, skipping (or use -s{})!",
                        num_extra, line_no, num_extra
                    );
                } else if num_bits < expected_num_bits {
                    info!(
                        "Got {} ({}) missing fields in line {}, skip!",
                        expected_num_bits - num_bits,
                        expected_num_bits,
                        line_no
                    );
                } else {
                    let mut triplets = Frame::new();
                    for triplet in (0..num_bits).step_by(3) {
                        // Process per triple.
                        let slice = &bits[triplet..triplet + 3];
                        let triplet: Triplet = bits[triplet..triplet + 3].to_vec(); //vec![1.0, 2.0, 3.0];
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
    info!(
        "Frames {:?}, capacity {:?}",
        frames.len(),
        frames.capacity()
    );

    frames
}

/// Prints az, in, d, dN, dS, v, vN, a, aN
fn emit_header(marker_name: &String, xyz: bool) {
    if xyz == true {
        print!(
            "{}_X\t{}_Y\t{}_Z\t{}_az\t{}_in\t{}_d\t{}_dN\t{}_dS\t{}_v\t{}_vN\t{}_a\t{}_aN",
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name
        );
    } else {
        print!(
	    "{}_az\t{}_in\t{}_d\t{}_dN\t{}_dS\t{}_v\t{}_vN\t{}_vS\t{}_a\t{}_aN\t{}_aS",
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name,
            marker_name
        );
    }
}

/// Calculates the distances on the in-memory data frame. Returns a
/// vector with a vector containing distances for each sensor. Indexed
/// by position in the marker_names vector.
// Move to calculated?
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
        //info!("Calculating distances for {}", marker_name);

        dist = 0.0;
        prev_triplet = None;

        for frame in frames {
            let curr_triplet: &Triplet = &frame[i];

            if prev_triplet.is_some() {
                // Do we have a saved "previous line/triplet"?
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
/// Note that the velocity per frame is the same as the distance calculated above,
/// so unless we convert to m/s, they are the same.
fn calculate_velocities(mocap_file: &MoCapFile, distances: &Distances) -> Velocities {
    let mut velocities: Velocities = vec![SensorData::new(); mocap_file.marker_names.len()];

    let it = mocap_file.marker_names.iter();
    for (i, marker_name) in it.enumerate() {
        let mut result = distances[i].clone();
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
        //info!("Calculating accelerations for {}", marker_name);

        accelerations[i].push(0.0); // Need to anchor wity 0.
        let mut result = velocities[i]
            .windows(2)
            .map(|d| d[1] - d[0])
            .collect::<Vec<SensorFloat>>();
        accelerations[i].append(&mut result);
    }

    accelerations
}

/// Return the azimuths and the inclinations between the points.
// Note that we discard the radii.
fn calculate_angles(mocap_file: &MoCapFile, frames: &Frames) -> (Distances, Distances) {
    let mut angle = (0.0, 0.0, 0.0);
    let mut prev_triplet: Option<&Triplet> = None;
    let mut azis: Distances = vec![Vec::<SensorFloat>::new(); mocap_file.marker_names.len()];
    let mut incs: Distances = vec![Vec::<SensorFloat>::new(); mocap_file.marker_names.len()];

    // We can even reserve the size of the distance vectors...
    for v in &mut azis {
        v.reserve_exact(mocap_file.num_frames);
    }
    for v in &mut incs {
        v.reserve_exact(mocap_file.num_frames);
    }

    let it = mocap_file.marker_names.iter();
    for (i, marker_name) in it.enumerate() {
        //info!("Calculating distances for {}", marker_name);

        angle = (0.0, 0.0, 0.0);
        prev_triplet = None;

        for frame in frames {
            let curr_triplet: &Triplet = &frame[i];

            if prev_triplet.is_some() {
                // Do we have a saved "previous line/triplet"?
                let x = prev_triplet.clone().unwrap();
                angle = mocap::calculate_azimuth_inclination(&curr_triplet, &x);
            }
            azis[i].push(angle.1);
            incs[i].push(angle.2);

            //println!("{:?} {}", curr_triplet, dist); //, dist_3d_t(&curr_triplet, &prev_triplet));
            prev_triplet = Some(curr_triplet);
        }
    }

    (azis, incs)
}

/// Create a new output filename, tries to append "_d3D" to
/// the filename.
///
/// Short input filenames will return `output_d3D.tsv`.
///
fn create_outputfilename(filename: &str) -> String {
    let len = filename.len();
    if len > 4 {
        let suffix = &filename[len - 4..len]; // Also test for ".tsv" suffix.
        if suffix == ".tsv" {
            format!("{}{}", &filename[0..len - 4], "_d3D.tsv")
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
        assert!(result == "filename_d3D.tsv");
    }

    #[test]
    fn filename_short() {
        let result = create_outputfilename("");
        assert!(result == "output_d3D.tsv");
    }

    #[test]
    fn filename_four_chars() {
        let result = create_outputfilename("abcd");
        assert!(result == "output_d3D.tsv");
    }

    #[test]
    fn filename_five_chars() {
        let result = create_outputfilename("a.tsv");
        assert!(result == "a_d3D.tsv");
    }

    #[test]
    fn filename_no_tsv() {
        let result = create_outputfilename("abcde");
        assert!(result == "abcde_d3D.tsv");
    }
}
