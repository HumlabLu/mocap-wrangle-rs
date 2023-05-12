use regex::Regex;
    
/// The float type for the coordinates/velocities, etc.
/// Note that there are different sizes of integers as well.
/// # (Maybe use usize for all the integers?)
pub type SensorFloat = f32;

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

    // Move all "header lines" into a data structure, then apply the regexen
    // one by one? We could create one big string to work on?
    // Or give it the bufreader and consume until we have what we need?
    pub fn extract_no_of_frames(&mut self, l:&str) {
	let re_frames = Regex::new(r"NO_OF_FRAMES\t(\d+)").unwrap();
	match re_frames.captures(l) {
	    Some(caps) => {
		let cap = caps.get(1).unwrap().as_str();
		let cap_int = cap.parse::<u32>().unwrap();
		//println!("cap '{}'", cap_int);
		self.no_of_frames = cap_int;
	    }
	    None => {
		// The regex did not match.
	    }
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
