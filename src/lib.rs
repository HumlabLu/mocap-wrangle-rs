use lazy_static::lazy_static;
use regex::Regex;

/// The float type for the coordinates/velocities, etc.
/// Note that there are different sizes of integers as well.
/// # (Maybe use usize for all the integers?)
pub type SensorFloat = f32;
pub type SensorInt = u32;
pub type SensorData = Vec<SensorFloat>;
pub type Triplet = Vec<SensorFloat>;
pub type Frame = Vec<Triplet>;
pub type Frames = Vec<Frame>;
pub type Distances = Vec<SensorData>;
pub type Velocities = Vec<SensorData>;
pub type Accelerations = Vec<SensorData>;

/// Calculate the distance in 3D.
///
/// # Example
///
/// ```rust
/// let dist = mocap::dist_3d(&[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
/// ```
pub fn dist_3d(coords0: &[SensorFloat], coords1: &[SensorFloat]) -> SensorFloat {
    assert!(coords0.len() == 3);
    assert!(coords1.len() == 3);

    let squared_sum = coords0
        .iter()
        .zip(coords1.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .fold(0.0, |acc, x| acc + x);
    squared_sum.sqrt()
}

pub fn dist_3d_t(coords0: &Triplet, coords1: &Triplet) -> SensorFloat {
    let squared_sum = coords0
        .iter()
        .zip(coords1.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .fold(0.0, |acc, x| acc + x);
    squared_sum.sqrt()
}

// https://en.wikipedia.org/wiki/Spherical_coordinate_system
// Not sure how the MoCap coordinate system is oriented. Z is up/down,
// Y forward/backwards, X right/left?
// Note that for the inclination, 0 degs is straight up, and 180 is straight down.
// An inclination of 90 is on the same Z-coordinate.
// 90.0-incl gives a +90/-90 range.
pub fn calculate_azimuth_inclination(
    coords0: &Triplet,
    coords1: &Triplet,
) -> (SensorFloat, SensorFloat, SensorFloat) {
    // We calculate angles from point 0 to point 1, so we assume
    // point 0 is the origin.
    let x = coords1[0] - coords0[0];
    let y = coords1[1] - coords0[1];
    let z = coords1[2] - coords0[2];

    let r = dist_3d_t(&coords0, &coords1); // sqrt of sum coordinates^2
    let inc = (z / r).acos();
    let azimuth = y.atan2(x);

    //(r, azimuth.to_degrees(), inc.to_degrees())
    (r, azimuth, inc)
}

/// Struct to contain the metadata. In the MoCap file, the metadata
/// looks as follows.
/// ```text
/// NO_OF_FRAMES	16722
/// NO_OF_CAMERAS	20
/// NO_OF_MARKERS	64
/// FREQUENCY	200
/// NO_OF_ANALOG	0
/// ANALOG_FREQUENCY	0
/// DESCRIPTION	--
/// TIME_STAMP	2022-06-03, 10:47:36.627	94247.45402301
/// TIME_STAMP	2022-11-22, 22:00:35
/// DATA_INCLUDED	3D
/// MARKER_NAMES	x_HeadL	x_HeadTop	x_HeadR	x_HeadFront	x_LShoulderTop
/// TRAJECTORY_TYPES	Measured	Measured
/// 23.2 34.34 ... (sensor data)
/// ```
#[derive(Debug, Clone)]
pub struct MoCapFile {
    pub filename: String,
    pub out_filename: String,
    pub num_header_lines: usize,
    pub num_frames: usize,
    pub num_matches: usize,
    pub checked_header: bool,
    pub no_of_frames: SensorInt, // These should be Option<>s.
    pub no_of_cameras: SensorInt,
    pub no_of_markers: SensorInt,
    pub frequency: SensorInt,
    pub no_of_analog: SensorInt,
    pub description: String,
    pub time_stamp: String,
    pub data_included: String,
    pub marker_names: Vec<String>,
}

impl Default for MoCapFile {
    fn default() -> MoCapFile {
        MoCapFile {
            filename: String::new(),
            out_filename: String::new(),
            num_header_lines: 0,
            num_frames: 0,
            num_matches: 0,
            checked_header: false,
            no_of_frames: 0,
            no_of_cameras: 0,
            no_of_markers: 0,
            frequency: 0,
            no_of_analog: 0,
            description: String::new(),
            time_stamp: String::new(),
            data_included: String::new(),
            marker_names: vec![],
        }
    }
}

impl MoCapFile {
    // The number of markers in the vector.
    fn num_markers(&self) -> usize {
        self.marker_names.len()
    }

    // Quick and dirty method to determine if the file
    // contained a valid header. Maybe frequency should be
    // an Option?
    pub fn is_valid(&self) -> bool {
        if self.frequency > 0 && self.num_markers() > 0 {
            true
        } else {
            false
        }
    }
}

/// Outputs the header info in "mocap" style.
impl std::fmt::Display for MoCapFile {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // write!(f, "NAME {}\n", self.name); // Not part of original meta data.
        write!(f, "NO_OF_FRAMES\t{}\n", self.no_of_frames).unwrap(); // Should print the real count?
        write!(f, "NO_OF_CAMERAS\t{}\n", self.no_of_cameras).unwrap();
        write!(f, "NO_OF_MARKERS\t{}\n", self.no_of_markers).unwrap();
        write!(f, "FREQUENCY\t{}\n", self.frequency).unwrap();
        write!(f, "NO_OF_ANALOG\t{}\n", self.no_of_analog).unwrap();
        write!(f, "DESCRIPTION\t{}\n", self.description).unwrap();
        write!(f, "TIME_STAMP\t{}\n", self.time_stamp).unwrap();
        write!(f, "DATA_INCLUDED\t{}\n", self.data_included).unwrap();
        write!(f, "MARKER_NAMES\t{:?}", self.marker_names) // Needs fixing!
    }
}

// map, index by first word?
// let re_frames = Regex::new(r"NO_OF_FRAMES\t(\d+)").unwrap();

// Move all "header lines" into a data structure, then apply the regexen
// one by one? We could create one big string to work on?
// Or give it the bufreader and consume until we have what we need?

// Extract the number of frames from a header string.
const FRAMES_REGEX: &str = r"NO_OF_FRAMES\t(\d+)";
lazy_static! {
    static ref RE_FRAMES: Regex = Regex::new(FRAMES_REGEX).unwrap();
    static ref RE_CAMERAS: Regex = Regex::new(r"NO_OF_CAMERAS\t(\d+)").unwrap();
    static ref RE_MARKERS: Regex = Regex::new(r"NO_OF_MARKERS\t(\d+)").unwrap();
    static ref RE_FREQUENCY: Regex = Regex::new(r"^FREQUENCY\t(\d+)").unwrap();
    static ref RE_MARKER_NAMES: Regex = Regex::new(r"MARKER_NAMES\t(.+)").unwrap();
    static ref RE_TIME_STAMP: Regex = Regex::new(r"TIME_STAMP\t(.+?)(\t(.+)|\z)").unwrap();
    static ref RE_DESCRIPTION: Regex = Regex::new(r"DESCRIPTION\t(.+)").unwrap();
    static ref RE_DATA_INCLUDED: Regex = Regex::new(r"DATA_INCLUDED\t(.+)").unwrap();
}

// If we add the MoCapFile struct as parameter, we can
// store the matches directly in the function, meaning we can
// save all the extract_X functins in a vector and loop
// over it.
// (or even give a regexp and the mocap field/type, so we only
// need one generic match function).
pub fn extract_no_of_frames(l: &str) -> Option<SensorInt> {
    match RE_FRAMES.captures(l) {
        Some(caps) => {
            let cap = caps.get(1).unwrap().as_str();
            let cap_int = cap.parse::<SensorInt>().unwrap();
            //println!("cap '{}'", cap_int);
            Some(cap_int)
        }
        None => {
            // The regex did not match.
            None
        }
    }
}

pub fn extract_no_of_cameras(l: &str) -> Option<SensorInt> {
    match RE_CAMERAS.captures(l) {
        Some(caps) => {
            let cap = caps.get(1).unwrap().as_str();
            let cap_int = cap.parse::<SensorInt>().unwrap();
            //println!("cap '{}'", cap_int);
            Some(cap_int)
        }
        None => {
            // The regex did not match.
            None
        }
    }
}

pub fn extract_no_of_markers(l: &str) -> Option<SensorInt> {
    match RE_MARKERS.captures(l) {
        Some(caps) => {
            let cap = caps.get(1).unwrap().as_str();
            let cap_int = cap.parse::<SensorInt>().unwrap();
            //println!("cap '{}'", cap_int);
            Some(cap_int)
        }
        None => {
            // The regex did not match.
            None
        }
    }
}

pub fn extract_frequency(l: &str) -> Option<SensorInt> {
    match RE_FREQUENCY.captures(l) {
        Some(caps) => {
            let cap = caps.get(1).unwrap().as_str();
            let cap_int = cap.parse::<SensorInt>().unwrap();
            //println!("cap '{}'", cap_int);
            Some(cap_int)
        }
        None => {
            // The regex did not match.
            None
        }
    }
}

// These are to be used again if we request the header in the output.
pub fn extract_marker_names(l: &str) -> Option<Vec<String>> {
    match RE_MARKER_NAMES.captures(l) {
        Some(caps) => {
            let cap = caps.get(1).unwrap().as_str();
            //println!("cap '{}'", cap);
            let seperator = Regex::new(r"(\t)").expect("Invalid regex");
            // Split, convert to String, iterate and collect.
            let splits: Vec<String> = seperator
                .split(cap)
                .map(|s| s.to_string())
                .into_iter()
                .collect();
            //println!( "{:?}", splits );
            Some(splits)
        }
        None => {
            // No match.
            None
        }
    }
}

pub fn extract_time_stamp(l: &str) -> Option<String> {
    match RE_TIME_STAMP.captures(&l) {
        Some(caps) => {
            //println!("caps {:?}", caps);
            let cap = caps.get(1).unwrap().as_str();
            Some(cap.to_string())
        }
        None => {
            // No match.
            None
        }
    }
}

pub fn extract_description(l: &str) -> Option<String> {
    match RE_DESCRIPTION.captures(&l) {
        Some(caps) => {
            //println!("caps {:?}", caps);
            let cap = caps.get(1).unwrap().as_str();
            Some(cap.to_string())
        }
        None => {
            // No match.
            None
        }
    }
}

pub fn extract_data_included(l: &str) -> Option<String> {
    match RE_DATA_INCLUDED.captures(&l) {
        Some(caps) => {
            //println!("caps {:?}", caps);
            let cap = caps.get(1).unwrap().as_str();
            Some(cap.to_string())
        }
        None => {
            // No match.
            None
        }
    }
}

/// Struct for calculated data, such as distance, velocity, &c.
// Add to MoCapFile?
// Add min/max traits?
#[derive(Debug, Clone)]
pub struct Calculated {
    pub distances: Option<Distances>,
    pub velocities: Option<Velocities>,
    pub accelerations: Option<Accelerations>,
    pub min_distances: Option<SensorData>,
    pub max_distances: Option<SensorData>,
    pub min_velocities: Option<SensorData>,
    pub max_velocities: Option<SensorData>,
    pub min_accelerations: Option<SensorData>,
    pub max_accelerations: Option<SensorData>,
    pub mean_distances: Option<SensorData>,
    pub stdev_distances: Option<SensorData>,
}

impl Default for Calculated {
    fn default() -> Calculated {
        Calculated {
            distances: None,
            velocities: None,
            accelerations: None,
            min_distances: None,
            max_distances: None,
            min_velocities: None,
            max_velocities: None,
            min_accelerations: None,
            max_accelerations: None,
            mean_distances: None,
            stdev_distances: None,
        }
    }
}

impl Calculated {
    pub fn is_valid(&self) -> bool {
        self.distances.is_some() && self.velocities.is_some() && self.accelerations.is_some()
    }

    pub fn calculate_min_distances(&mut self) {
        if self.min_distances.is_none() {
            let mut min_distances: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let distances = self.distances.as_mut().unwrap();
            for d in distances {
                let min_d: SensorFloat = d.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                min_distances.push(min_d);
            }
            self.min_distances = Some(min_distances);
        }
    }

    pub fn calculate_max_distances(&mut self) {
        if self.max_distances.is_none() {
            let mut max_distances: SensorData = vec![];
            let distances = self.distances.as_mut().unwrap();
            for d in distances {
                let max_d: SensorFloat = d.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                max_distances.push(max_d);
            }
            self.max_distances = Some(max_distances);
        }
    }

    pub fn calculate_min_velocities(&mut self) {
        if self.min_velocities.is_none() {
            let mut min_velocities: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let velocities = self.velocities.as_mut().unwrap();
            for d in velocities {
                let min_d: SensorFloat = d.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                min_velocities.push(min_d);
            }
            self.min_velocities = Some(min_velocities);
        }
    }

    pub fn calculate_max_velocities(&mut self) {
        if self.max_velocities.is_none() {
            let mut max_velocities: SensorData = vec![];
            let velocities = self.velocities.as_mut().unwrap();
            for d in velocities {
                let max_d: SensorFloat = d.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                max_velocities.push(max_d);
            }
            self.max_velocities = Some(max_velocities);
        }
    }

    pub fn calculate_min_accelerations(&mut self) {
        if self.min_accelerations.is_none() {
            let mut min_accelerations: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let accelerations = self.accelerations.as_mut().unwrap();
            for d in accelerations {
                let min_d: SensorFloat = d.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                min_accelerations.push(min_d);
            }
            self.min_accelerations = Some(min_accelerations);
        }
    }

    pub fn calculate_max_accelerations(&mut self) {
        if self.max_accelerations.is_none() {
            let mut max_accelerations: SensorData = vec![];
            let accelerations = self.accelerations.as_mut().unwrap();
            for d in accelerations {
                let max_d: SensorFloat = d.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                max_accelerations.push(max_d);
            }
            self.max_accelerations = Some(max_accelerations);
        }
    }

    pub fn calculate_mean_distances(&mut self) {
        if self.mean_distances.is_none() {
            let mut mean_distances: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let distances = self.distances.as_mut().unwrap(); // Vector with the distances
            for d in distances {
                let mean_d = mean(&d);
                mean_distances.push(mean_d);
            }
            self.mean_distances = Some(mean_distances);
        }
    }

    pub fn calculate_stdev_distances(&mut self) {
        if self.stdev_distances.is_none() {
            let mut stdev_distances: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let distances = self.distances.as_mut().unwrap(); // Vector with the distances
            for d in distances {
                let stdev_d = standard_dev(&d);
                stdev_distances.push(stdev_d);
            }
            self.stdev_distances = Some(stdev_distances);
        }
    }
}

impl Calculated {
    pub fn get_distance(&self, sensor_id: usize, frame_no: usize) -> mut &SensorFloat {
	self.distances.as_ref().unwrap()
            .get(sensor_id) // Get the data for the i-th sensor.
            .unwrap()
            .get(frame_no) // Get the value for the f-th frame.
            .unwrap()
    }
}

pub fn normalise_minmax(val: &SensorFloat, min: &SensorFloat, max: &SensorFloat) -> SensorFloat {
    (val - min) / (max - min)
}

pub fn mean(data: &SensorData) -> SensorFloat {
    let sum: SensorFloat = data.iter().sum();
    sum / data.len() as SensorFloat
}

pub fn variance(data: &SensorData) -> SensorFloat {
    let mean = mean(&data);
    let sum_diffs = data
        .iter()
        .map(|value| {
            let diff = mean - (*value as SensorFloat);
            diff * diff
        })
        .sum::<SensorFloat>();
    sum_diffs / data.len() as SensorFloat
}

pub fn standard_dev(data: &SensorData) -> SensorFloat {
    let variance = variance(&data);
    variance.sqrt()
}

pub fn standardise(val: &SensorFloat, mean: &SensorFloat, stddev: &SensorFloat) -> SensorFloat {
    (val - mean) / stddev
}

/// A generalised mean calculator.
pub fn calculate_means(data: &Vec<SensorData>) -> SensorData {
    let mut means: SensorData = vec![];
    for d in data {
        let mean = mean(&d);
        means.push(mean);
    }
    means
}

/// A general version of the calculate_stdev function.
//  let mut velocities = calculated.velocities.as_ref().unwrap();
//  let foo = mocap::calculate_stdevs(&velocities);
pub fn calculate_stdevs(data: &Vec<SensorData>) -> SensorData {
    let mut stdevs: SensorData = vec![];
    for d in data {
        let stdev_d = standard_dev(&d);
        stdevs.push(stdev_d);
    }
    stdevs
}
