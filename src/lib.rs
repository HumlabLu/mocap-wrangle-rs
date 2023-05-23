use regex::Regex;
use lazy_static::lazy_static;

/// The float type for the coordinates/velocities, etc.
/// Note that there are different sizes of integers as well.
/// # (Maybe use usize for all the integers?)
pub type SensorFloat = f32;
pub type SensorInt = u32;

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

    let squared_sum = coords0.iter()
        .zip(coords1.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .fold(0.0, |acc, x| acc + x);
    squared_sum.sqrt()
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
    pub name: String,
    pub no_of_frames: SensorInt,
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
}

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
	    let splits: Vec<String> = seperator.split(cap).map(|s| s.to_string()).into_iter().collect();
	    //println!( "{:?}", splits );
	    Some(splits)
	}
	None => {
	    // No match.
	    None
	}
    }
}
