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

impl MoCapFile {
    fn num_markers(&self) -> usize {
        self.marker_names.len()
    }
}

// Quick and dirty method to determine if the file
// contained a valid header. Maybe frequency should be
// an Option?
impl MoCapFile {
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
