use clap::Parser;
use color_eyre::Result;
use std::fs::{File, OpenOptions};
use std::io::BufRead;
use std::time::Instant;

use mocap::{dist_3d_t, MoCapFile};
use mocap::{
    extract_values_inplace, Accelerations, Distances, Frame, Frames, SensorData, SensorFloat,
    Triplet, Velocities,
};

#[macro_use]
extern crate log;
extern crate simplelog;
use simplelog::*;
use std::process::Command;

// =====================================================================
// Command line arguments.
// =====================================================================

#[derive(Parser, Debug, Clone)]
struct Args {
    // Filename
    #[arg(short, long, help = "The input file with MoCap sensor data.")]
    file: String, // Path thingy?

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

    // Output position coordinates ONLYU
    #[clap(long, action, help = "Only X, Y and Z coordinates in output.")]
    coordsonly: bool,

    // Skip fields
    #[clap(
        long,
        short,
        action,
        default_value_t = 0,
        help = "Skip first n columns in sensor data."
    )]
    skip: usize,

    // Step for reading frames
    #[clap(long, action, default_value_t = 1, help = "Read frames in steps.")]
    framestep: usize,

    // Header output
    #[clap(long, action, help = "Do not output header row.")]
    noheader: bool,

    // Frameno/Timestamp output
    #[clap(long, action, help = "Add frame number and timestamp.")]
    timestamp: bool,

    // Timestamp start (also similar for output?
    #[clap(long, action, default_value_t = 0, help = "Starting time (ms).")]
    starttime: usize,

    // Frameno start (starttime could be calculated from this, but we keep it
    // separate for now).
    #[clap(long, action, default_value_t = 0, help = "Starting frame number.")]
    startframe: usize,

    // Output starting time?
    #[clap(
        long,
        action,
        default_value_t = 0,
        help = "Start outputting at this timestamp (ms)."
    )]
    outputstarttimestamp: usize,

    // Output starting frame_number? this is count, not the number in the file?
    #[clap(
        long,
        action,
        default_value_t = 0,
        help = "Start outputting at this frame number."
    )]
    outputstartframe: usize,

    // Output this number if frames.
    #[clap(
        long,
        action,
        default_value_t = 0,
        help = "Output this many frames."
    )]
    outputframes: usize,

    #[clap(long, short, action, help = "Convert vel/acc to m/s.")]
    metric: bool,

    #[arg(num_args(0..))]
    #[clap(short, long, action, help = "Sensor IDs")]
    keep: Option<Vec<usize>>,

    // Info only output
    #[clap(long, short, action, help = "Only parse header and print info.")]
    info: bool,

    // Do not warn for NaN, it appears the  LogLevel cannot
    // be changed after initialisation? CHECK
    // This hack works for now.
    #[clap(long, short, action, help = "Do not emit warnings for NaNs.")]
    nonans: bool,
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

    // This only works in the git branch...
    let output = Command::new("git").args(&["rev-parse", "HEAD"]).output();
    let git_hash = match output {
        Ok(output) => {
            match String::from_utf8(output.stdout) {
                Ok(mut hash) => {
                    hash.pop(); // Remove trailing newline.
                    hash
                },
                Err(e) => {
                    error!("Failed to convert output to UTF-8: {}", e);
                    "??".to_string()
                }
            }
        },
        Err(e) => {
            error!("Command execution failed: {}", e);
            "??".to_string()
        }
    };
    info!("GIT_HASH={}", git_hash);

    
/*    let output = Command::new("git").args(&["rev-parse", "HEAD"]).output().unwrap();
    let mut git_hash = String::from_utf8(output.stdout).unwrap();
    git_hash.pop(); // Remove trailing newline.
    info!("GIT_HASH={}", git_hash);*/

    let args = Args::parse();
    info!("{:?}", args);
    
    /*
    let bits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    let keep = vec![1, 3];
    let result = extract_values(&bits, &keep);
    println!("{:?}", result); // This should print [0.4, 0.5, 0.6, 1.0, 1.1, 1.2]
    */

    /*
    let mut bits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    let keep = vec![1, 3];
    extract_values_inplace(&mut bits, &keep);
    println!("{:?}", bits); // This should print [0.4, 0.5, 0.6, 1.0, 1.1, 1.2]
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
    //
    // 12000 lines @200Hz of gestures_ML_05 is 4181327 bytes on disk.
    // 1 hour would be 720000 lines (plus header) = 250879620 bytes (240MB).
    // =====================================================================

    let filename = &args.file;
    let file_size = std::fs::metadata(filename)?.len();
    if file_size < 8 {
        // Arbitrary size... but to prevent creation of 0-byte files.
        error!("Error: Inputfile is too small!");
        std::process::exit(2);
    }

    // We have the filenames, create the structure.
    let mut mocap_file = MoCapFile {
        filename: filename.clone(),
        ..Default::default()
    };

    info!("Reading header file {}", filename);
    parse_header(&mut mocap_file)?;

    // Print markers and indices.
    match &args.keep {
        Some(keep) => {
            for (i, marker_name) in mocap_file.marker_names.iter().enumerate() {
                if keep.contains(&i) {
                    info!("{i:02}: {marker_name} *");
                } else {
                    //info!("{i:02}: {marker_name}");
                }
            }
        }
        None => {
            for (i, marker_name) in mocap_file.marker_names.iter().enumerate() {
                info!("{i:02}: {marker_name}");
            }
        }
    }

    // We also need to filter the marker_names, if we select
    // only some sensors...
    if let Some(keep) = &args.keep {
        let mut kept_marker_names: Vec<String> = vec![];
        for idx in keep.iter() {
            kept_marker_names.push(mocap_file.marker_names[*idx].clone());
        }
        mocap_file.marker_names = kept_marker_names;
    }

    info!(
        "Header contains {} lines, {} matched.",
        mocap_file.num_header_lines, mocap_file.num_matches
    );
    let (hours, minutes, seconds, milliseconds) = frames_to_time(&mocap_file);
    info!("Expecting {} frames ({:02}:{:02}:{:02}.{:03}).", mocap_file.no_of_frames, hours, minutes, seconds, milliseconds);
    
    // Without a header, we abort.
    if (mocap_file.num_header_lines == 0) || (mocap_file.num_matches == 0) {
        error!("File contains no header!");
        std::process::exit(3);
    }

    if args.info { // Print info only, exit here.
        std::process::exit(0);
    }

    let time_start = Instant::now();
    let (frames, frame_numbers, timestamps) = read_frames(&mut mocap_file, &args);
    let time_duration = time_start.elapsed().as_millis() + 1; // Add one to avoid division by zero.
    let lps = mocap_file.num_frames as u128 * 1000 / time_duration;

    info!(
        "Ready, frames: {} (in {} ms, {} l/s)",
        mocap_file.num_frames, time_duration, lps
    );

    // We could/should exit here if no frames.

    mocap_file.add_frames(frames);
    mocap_file.add_frame_numbers(frame_numbers);
    mocap_file.add_timestamps(timestamps);

    info!("Calculating distances.");
    mocap_file.calculate_distances();

    if args.metric {
        info!("Calculating velocities in m/s.");
        mocap_file.calculate_velocities_ms();
    } else {
        info!("Calculating velocities.");
        mocap_file.calculate_velocities();
    }

    // Acceleration becomes in m/s/s automatically if velocity
    // is in m/s. Does it? CHECK
    if args.metric {
        info!("Calculating accelerations in m/s.");
        mocap_file.calculate_accelerations_ms();
    } else {
        info!("Calculating accelerations.");
        mocap_file.calculate_accelerations();
    }

    info!("Calculating angles.");
    mocap_file.calculate_angles();

    info!("Calculating min/max/mean/stdev.");
    mocap_file.calculate_min_distances();
    mocap_file.calculate_max_distances();
    mocap_file.calculate_min_velocities();
    mocap_file.calculate_max_velocities();
    mocap_file.calculate_min_accelerations();
    mocap_file.calculate_max_accelerations();
    mocap_file.calculate_mean_distances();
    mocap_file.calculate_stdev_distances();
    mocap_file.calculate_mean_velocities();
    mocap_file.calculate_stdev_velocities();
    mocap_file.calculate_mean_accelerations();
    mocap_file.calculate_stdev_accelerations();

    // Output to std out.
    if mocap_file.num_frames == 0 {
        info!("No data to output!");
        std::process::exit(1);
    } else {
        info!("Outputting data.");
    }

    let time_start = Instant::now();
    let f_it = mocap_file.get_frames().iter();

    if !args.noheader {
        if args.timestamp {
            print!("Frame\tTimestamp\t");
        }
        for (i, marker_name) in mocap_file.marker_names.iter().enumerate() {
            if i > 0 {
                print!("\t"); // Separator, but not at start/end.
            }
            // We need a "output fields for sensor" function, taking args to output relevant header.
            emit_header(marker_name, args.coords, args.coordsonly);
        }
        println!();
    }

    let mut output_frames: usize = 0; // Counter.
    
    for (frame_no, frame) in f_it.enumerate() {
        // We skip if we have supplied outputstartframe/outputstarttimestamp
        // and we have not reached the lower threshold yet.
        let the_frame_no: &usize = mocap_file.get_frame_number(frame_no);
        if *the_frame_no < args.outputstartframe {
            continue;
        }
        let the_timestamp: &usize = mocap_file.get_timestamp(frame_no);
        if *the_timestamp < args.outputstarttimestamp {
            continue;
        }

        if args.timestamp {
            print!(
                "{:.3}\t{:.3}\t",
                the_frame_no,
                *the_timestamp as f64 / 1000.0
            );
        }

        let it = mocap_file.marker_names.iter(); // Match to include?
        for (sensor_id, marker_name) in it.enumerate() {
            // The sensor_id-th column of triplets (a sensor)
            let the_triplet = &frame[sensor_id];

            let the_d = mocap_file.get_distance(sensor_id, frame_no);
            let min_d = mocap_file.get_min_distance(sensor_id);
            let max_d = mocap_file.get_max_distance(sensor_id);

            let mean_d = mocap_file.get_mean_distance(sensor_id);
            let stdev_d = mocap_file.get_stdev_distance(sensor_id);
            let nor_d = mocap::normalise_minmax(the_d, min_d, max_d);
            let std_d = mocap::standardise(the_d, mean_d, stdev_d); // First one should really be 0.0?

            let the_v = mocap_file.get_velocity(sensor_id, frame_no);
            let min_v = mocap_file.get_min_velocity(sensor_id);
            let max_v = mocap_file.get_max_velocity(sensor_id);

            let mean_v = mocap_file.get_mean_velocity(sensor_id);
            let stdev_v = mocap_file.get_stdev_velocity(sensor_id);
            let nor_v = mocap::normalise_minmax(the_v, min_v, max_v);
            let std_v = mocap::standardise(the_v, mean_v, stdev_v); // First one should really be 0.0?

            let the_a = mocap_file.get_acceleration(sensor_id, frame_no);
            let min_a = mocap_file.get_min_acceleration(sensor_id);
            let max_a = mocap_file.get_max_acceleration(sensor_id);

            let mean_a = mocap_file.get_mean_acceleration(sensor_id);
            let stdev_a = mocap_file.get_stdev_acceleration(sensor_id);
            let nor_a = mocap::normalise_minmax(the_a, min_a, max_a);
            let std_a = mocap::standardise(the_a, mean_a, stdev_a); // First one should really be 0.0?

            let azim = mocap_file.get_azimuth(sensor_id, frame_no);
            let mut incl = mocap_file.get_inclination(sensor_id, frame_no);

            // We fix NaNs (make them 0.0). Default is to warn, but
            // this can cause a lot of output.
            if incl.is_nan() {
                if !args.nonans {
                    warn!(
                        "Inclination NaN fixed in frame {}, sensor {}/{}.",
                        frame_no, sensor_id, marker_name
                    );
                }
                incl = &0.0;
            }
            if sensor_id > 0 {
                print!("\t");
            }
            if args.coordsonly {
                print!(
                    "{:.2}\t{:.2}\t{:.2}",
                    the_triplet.first().unwrap(),
                    the_triplet.get(1).unwrap(),
                    the_triplet.get(2).unwrap(),
                );
            } else if args.coords {
                print!("{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
            	    the_triplet.first().unwrap(), the_triplet.get(1).unwrap(), the_triplet.get(2).unwrap(),
            	    azim, incl, the_d, nor_d, std_d,
            	    the_v, nor_v, std_v,
            	    the_a, nor_a, std_a,
            	);
            } else {
                print!(
                    "{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
                    azim, incl, the_d, nor_d, std_d, the_v, nor_v, std_v, the_a, nor_a, std_a
                );
            }
        }
        println!();
        output_frames += 1;
        if (args.outputframes > 0) && (output_frames >= args.outputframes) { // we could count-down this one...
            break;
        }
    }
    let time_duration = time_start.elapsed().as_millis() + 1; // Add one to avoid division by zero.
    let lps = output_frames as u128 * 1000 / time_duration;

    info!(
        "Ready, frames: {}/{} (in {} ms, {} l/s).",
        output_frames, mocap_file.num_frames, time_duration, lps
    );
    
    if args.verbose {
        info!("{}", mocap_file);
    }

    Ok(())
}

/// Parse the header, fill in the fields in the struct.
// Implement this in MoCapFile instead
fn parse_header(mocap_file: &mut MoCapFile) -> Result<()> {
    let file = File::open(&mocap_file.filename).expect("could not open file");
    let fileiter = std::io::BufReader::new(&file).lines();
    let mut line_no: usize = 0; // Counts lines in the file.
    let mut num_matches: usize = 0; // Counts regex matches.

    for line in fileiter {
        if let Ok(l) = line {
            //println!("{}", l);
            if l.is_empty() {
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

/// Read the data into memory. Can only be run after parse_header(...).
/// Returns a Frames structure containing a vector with vectors with triplets.
fn read_frames(mocap_file: &mut MoCapFile, args: &Args) -> (Frames, Vec<usize>, Vec<usize>) {
    info!("Reading file {} into memory", &mocap_file.filename);
    let file = File::open(&mocap_file.filename).expect("could not open file");
    let mut fileiter = std::io::BufReader::new(file).lines();

    // Skip the header.
    info!("Skipping {} header lines.", mocap_file.num_header_lines);
    for _ in fileiter.by_ref().take(mocap_file.num_header_lines) {
        // print, save, show?
    }

    let mut line_no: usize = 0; // Counts line in the file.
    let mut frame_no: usize = 0; // Counts the lines with sensor data.
    let mut frames = Frames::with_capacity(mocap_file.no_of_frames as usize);
    let mut frame_numbers = Vec::<usize>::with_capacity(mocap_file.no_of_frames as usize);
    let mut timestamps = Vec::<usize>::with_capacity(mocap_file.no_of_frames as usize);
    let mut timestamp: usize = 0;

    // Using a framestep > 1 can be used to "smooth" the data (kind of).
    // TODO: this affects the calculations!
    for line in fileiter.step_by(args.framestep) {
        if let Ok(l) = line {
            if l.is_empty() {
                continue; // Very short lines, empty, &c.
            }
            // Should we check if it is identical to the previous line?
            let ch = &l.chars().take(1).last().unwrap(); // Surely, this could be simplified?!
            if ch.is_ascii_uppercase() {
                // this shouldn't happen, FIX.
                // Actually it should be numeric only...
            } else {
                // Assume we are in the data part.
                //let bits: Vec<&str> = l.split("\t").collect();
                let mut bits = l
                    .split('\t')
                    .filter_map(
                        |s| s.parse::<SensorFloat>().ok(), // We assume all SensorFloat values for now.
                    )
                    .collect::<Vec<SensorFloat>>();
                // We we requested to skip, we remove the first args.skip.
                if args.skip > 0 {
                    //let u: Vec<_> = bits.drain(0..args.skip).collect(); // Keep them, output them?
                    bits.drain(0..args.skip);
                }

                // We now have a vector with all the values.
                // If we want to select sensors... would we do that here?
                // Imagine we want sensors index 1 and 3.
                // x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, ...
                // 0   1   2   3   4   5   6   7   8   9   10  11  ...
                // So we want (1*3)+(0,1,2) and (3*3)+(0,1,2)
                // let keep = vec![1, 3];
                // step_by? or just iter over keep? if x in keep?
                //let kept = keep.iter().enumerate().filter();

                let mut num_bits = bits.len(); // Should be 3 * marker_names.len()
                let expected_num_bits = (mocap_file.no_of_markers * 3) as usize;
                let num_extra = num_bits.checked_sub(expected_num_bits); // Unsigned values.
                match num_extra {
                    Some(0) => {
                        // We got exactly what we expected.
                        //extract_values_inplace(&mut bits, &keep);
                        //let num_bits = bits.len(); // Should be 3 * marker_names.len()
                        if let Some(keep) = &args.keep {
                            extract_values_inplace(&mut bits, keep);
                            num_bits = bits.len(); // Should be 3 * marker_names.len()
                        }

                        let mut triplets = Frame::new();
                        for triplet in (0..num_bits).step_by(3) {
                            // Process per triple. (FIX duplicate code, see below!)
                            // Here we should check whether to include or not?
                            // if index in keep ...
                            let _slice = &bits[triplet..triplet + 3];
                            let triplet: Triplet = bits[triplet..triplet + 3].to_vec(); //vec![1.0, 2.0, 3.0];
                            triplets.push(triplet);
                        }
                        frames.push(triplets);
                        // Generate fake frame_number/timestamp
                        frame_numbers.push(frame_no + args.startframe);
                        timestamps.push(timestamp + args.starttime);
                        frame_no += 1;
                        timestamp += mocap_file.get_timeinc();
                    }
                    Some(2) => {
                        // Two extra, we assume frame number and timestamp.
                        let frame_number = bits[0];
                        let frame_number = frame_number as usize;
                        let timestamp = &bits[1] * 1000.0; // We convert to milliseconds.
                        let timestamp = timestamp as usize;
                        //info!("{}/{}", frame_number, timestamp);
                        frame_numbers.push(frame_number);
                        timestamps.push(timestamp);
                        let mut triplets = Frame::new();
                        for triplet in (2..num_bits).step_by(3) {
                            // Process per triple.
                            let _slice = &bits[triplet..triplet + 3];
                            let triplet: Triplet = bits[triplet..triplet + 3].to_vec(); //vec![1.0, 2.0, 3.0];
                            triplets.push(triplet);
                        }
                        frames.push(triplets);
                        frame_no += 1;
                    }
                    None => {
                        // Negative, we got fewer fields than expected.
                        info!(
                            "Got {} (want {}); missing fields in line {}, skipping!",
                            num_bits, expected_num_bits, line_no
                        );
                    }
                    _ => {
                        // More (other than two) fields than expected.
                        info!(
                            "Got {} extra fields in line {}, skipping!",
                            num_bits - expected_num_bits,
                            line_no
                        );
                    }
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

    (frames, frame_numbers, timestamps)
}

/// Prints sensor name plus az, in, d, dN, dS, v, vN, vS, a, aN, aS 
/// Optionally includes X, Y, Z coordinates.
fn emit_header(marker_name: &String, xyz: bool, xyzonly: bool) {
    if xyzonly {
        print!("{}_X\t{}_Y\t{}_Z", marker_name, marker_name, marker_name);
        return;
    }
    if xyz {
        print!(
            "{}_X\t{}_Y\t{}_Z\t{}_az\t{}_in\t{}_d\t{}_dN\t{}_dS\t{}_v\t{}_vN\t{}_vS\t{}_a\t{}_aN\t{}_aS",
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
fn _calculate_distances(mocap_file: &MoCapFile, frames: &Frames) -> Distances {
    let mut dist;
    let mut prev_triplet: Option<&Triplet>;
    let mut distances: Distances = vec![Vec::<SensorFloat>::new(); mocap_file.marker_names.len()]; // HashMap?

    // We can even reserve the size of the distance vectors...
    for v in &mut distances {
        v.reserve_exact(mocap_file.num_frames);
    }

    let it = mocap_file.marker_names.iter();
    for (i, _marker_name) in it.enumerate() {
        //info!("Calculating distances for {}", marker_name);

        dist = 0.0;
        prev_triplet = None;

        for frame in frames {
            let curr_triplet: &Triplet = &frame[i];

            if let Some(pt) = prev_triplet {
                dist = dist_3d_t(curr_triplet, pt);
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
fn _calculate_velocities(mocap_file: &MoCapFile, distances: &Distances) -> Velocities {
    let mut velocities: Velocities = vec![SensorData::new(); mocap_file.marker_names.len()];

    let it = mocap_file.marker_names.iter();
    for (i, _marker_name) in it.enumerate() {
        let mut result = distances[i].clone();
        velocities[i].append(&mut result);
    }

    velocities
}

/// Calculates the accelerations on the supplied Acceleration data.
/// Returns a vector with a vector containing accelerations for each sensor. Indexed
/// by position in the marker_names vector.
fn _calculate_accelerations(mocap_file: &MoCapFile, velocities: &Velocities) -> Accelerations {
    let mut accelerations: Accelerations = vec![SensorData::new(); mocap_file.marker_names.len()];

    let it = mocap_file.marker_names.iter();
    for (i, _marker_name) in it.enumerate() {
        //info!("Calculating accelerations for {}", marker_name);

        accelerations[i].push(0.0); // Need to anchor with 0.
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
fn _calculate_angles(mocap_file: &MoCapFile, frames: &Frames) -> (Distances, Distances) {
    let mut angle;
    let mut prev_triplet: Option<&Triplet>;
    // Prepare the vector of vectors for the data.
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
    for (i, _marker_name) in it.enumerate() {
        //info!("Calculating distances for {}", marker_name);

        angle = (0.0, 0.0, 0.0);
        prev_triplet = None;

        for frame in frames {
            let curr_triplet: &Triplet = &frame[i];

            if let Some(pt) = prev_triplet {
                angle = mocap::calculate_azimuth_inclination(curr_triplet, pt);
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
fn _create_outputfilename(filename: &str) -> String {
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

fn frames_to_time(mocap_file: &MoCapFile) -> (usize, usize, usize, usize) {
    let total_seconds = mocap_file.no_of_frames as usize / mocap_file.frequency as usize;
    let hours = total_seconds / 3600; // Find total hours
    let minutes = (total_seconds % 3600) / 60; // Find remaining minutes
    let seconds = total_seconds % 60; // Find remaining seconds
    let milliseconds = (mocap_file.no_of_frames as usize % mocap_file.frequency as usize)
        * mocap_file.get_timeinc();

    (hours, minutes, seconds, milliseconds)
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_normal() {
        let result = _create_outputfilename("filename.tsv");
        assert!(result == "filename_d3D.tsv");
    }

    #[test]
    fn filename_short() {
        let result = _create_outputfilename("");
        assert!(result == "output_d3D.tsv");
    }

    #[test]
    fn filename_four_chars() {
        let result = _create_outputfilename("abcd");
        assert!(result == "output_d3D.tsv");
    }

    #[test]
    fn filename_five_chars() {
        let result = _create_outputfilename("a.tsv");
        assert!(result == "a_d3D.tsv");
    }

    #[test]
    fn filename_no_tsv() {
        let result = _create_outputfilename("abcde");
        assert!(result == "abcde_d3D.tsv");
    }
}
