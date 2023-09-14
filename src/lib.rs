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
    let inc = (z / r).acos(); // r can be 0 if identical coordinates, then inc == NaN.
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
    pub frames: Option<Frames>,
    pub frame_numbers: Option<Vec<usize>>, // usize
    pub timestamps: Option<Vec<usize>>,    // usize (in ms)
    pub distances: Option<Distances>,
    pub velocities: Option<Velocities>,
    pub accelerations: Option<Accelerations>,
    pub azimuths: Option<Distances>,
    pub inclinations: Option<Distances>,
    pub min_distances: Option<SensorData>,
    pub max_distances: Option<SensorData>,
    pub min_velocities: Option<SensorData>,
    pub max_velocities: Option<SensorData>,
    pub min_accelerations: Option<SensorData>,
    pub max_accelerations: Option<SensorData>,
    pub mean_distances: Option<SensorData>,
    pub stdev_distances: Option<SensorData>,
    pub mean_velocities: Option<SensorData>,
    pub stdev_velocities: Option<SensorData>,
    pub mean_accelerations: Option<SensorData>,
    pub stdev_accelerations: Option<SensorData>,
    pub resolution: SensorInt,
}

impl Default for MoCapFile {
    fn default() -> MoCapFile {
        MoCapFile {
            filename: String::new(),
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
            frames: None,
            frame_numbers: None,
            timestamps: None,
            distances: None,
            velocities: None,
            azimuths: None,
            inclinations: None,
            accelerations: None,
            min_distances: None,
            max_distances: None,
            min_velocities: None,
            max_velocities: None,
            min_accelerations: None,
            max_accelerations: None,
            mean_distances: None,
            stdev_distances: None,
            mean_velocities: None,
            stdev_velocities: None,
            mean_accelerations: None,
            stdev_accelerations: None,
            resolution: 100, // we have centimeters in data, factor 100 to meters.
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

    // Should be precalculated and stored.
    /// Returns the frame gap in ms, e.g 200Hz -> 5ms.
    pub fn get_timeinc(&self) -> usize {
        (1000 / self.frequency).try_into().unwrap()
    }

    // We could store the frames and calculate struc/functions here?
    pub fn add_frames(&mut self, frames: Frames) {
        self.frames = Some(frames);
        // call calculate dist/vel/acc here ?
    }

    pub fn add_frame_numbers(&mut self, frame_numbers: Vec<usize>) {
        self.frame_numbers = Some(frame_numbers);
    }

    pub fn add_timestamps(&mut self, timestamps: Vec<usize>) {
        self.timestamps = Some(timestamps);
    }

    pub fn calculate_distances(&mut self) {
        let mut dist; // = 0.0;
        let mut prev_triplet: Option<&Triplet>; // = None;
        let mut distances: Distances = vec![Vec::<SensorFloat>::new(); self.num_markers()]; // HashMap?

        // We can even reserve the size of the distance vectors...
        for v in &mut distances {
            v.reserve_exact(self.num_frames);
        }

        let it = self.marker_names.iter();
        for (i, _marker_name) in it.enumerate() {
            dist = 0.0;
            prev_triplet = None;

            let frames: &Frames = self.frames.as_mut().unwrap();
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
        self.distances = Some(distances);
    }

    /// Calculates the velocities on the distance data.
    /// Note that the velocity per frame is the same as the distance calculated above,
    /// so unless we convert to m/s, they are the same.
    pub fn calculate_velocities(&mut self) {
        self.velocities = self.distances.clone();
    }

    // Distance per second. Use resolution value to get meters.
    pub fn calculate_velocities_ms(&mut self) {
        let mut velocities: Velocities = vec![SensorData::new(); self.num_markers()];
        let f = self.frequency as f32;
        let r = self.resolution as f32;
        let it = self.marker_names.iter();
        for (i, _marker_name) in it.enumerate() {
            let distances: &Distances = &self.distances.as_ref().unwrap().as_ref();
            //velocities[i].push(0.0); // Need to anchor with 0.
            let mut result = distances[i]
                .windows(1)
                .map(|d| d[0] * f / r)
                .collect::<Vec<SensorFloat>>();
            velocities[i].append(&mut result);
        }
        self.velocities = Some(velocities);
    }

    // Calculate the acceleration (derivative of velocities).
    pub fn calculate_accelerations(&mut self) {
        let mut accelerations: Accelerations = vec![SensorData::new(); self.num_markers()];

        let it = self.marker_names.iter();
        for (i, _marker_name) in it.enumerate() {
            //info!("Calculating accelerations for {}", marker_name);

            let velocities: &Velocities = &self.velocities.as_ref().unwrap().as_ref();
            accelerations[i].push(0.0); // Need to anchor with 0.
            let mut result = velocities[i]
                .windows(2)
                .map(|d| d[1] - d[0])
                .collect::<Vec<SensorFloat>>();
            accelerations[i].append(&mut result);
        }
        self.accelerations = Some(accelerations);
    }

    // Calculate the acceleration in m/s. The above is in meter/second/frame.
    pub fn calculate_accelerations_ms(&mut self) {
        let mut accelerations: Accelerations = vec![SensorData::new(); self.num_markers()];
        let f = self.frequency as f32;

        let it = self.marker_names.iter();
        for (i, _marker_name) in it.enumerate() {
            //info!("Calculating accelerations for {}", marker_name);

            let velocities: &Velocities = &self.velocities.as_ref().unwrap().as_ref();
            accelerations[i].push(0.0); // Need to anchor with 0.
            let mut result = velocities[i]
                .windows(2)
                .map(|d| (d[1] - d[0]) * f)
                .collect::<Vec<SensorFloat>>();
            accelerations[i].append(&mut result);
        }
        self.accelerations = Some(accelerations);
    }

    pub fn calculate_angles(&mut self) {
        let mut angle;
        let mut prev_triplet: Option<&Triplet>;
        let mut azis: Distances = vec![Vec::<SensorFloat>::new(); self.num_markers()];
        let mut incs: Distances = vec![Vec::<SensorFloat>::new(); self.num_markers()];

        // We can even reserve the size of the distance vectors...
        for v in &mut azis {
            v.reserve_exact(self.num_frames);
        }
        for v in &mut incs {
            v.reserve_exact(self.num_frames);
        }

        let frames: &Frames = self.frames.as_mut().unwrap();
        let it = self.marker_names.iter();
        for (i, _marker_name) in it.enumerate() {
            //info!("Calculating distances for {}", marker_name);

            angle = (0.0, 0.0, 0.0);
            prev_triplet = None;

            for frame in frames {
                let curr_triplet: &Triplet = &frame[i];

                if prev_triplet.is_some() {
                    // Do we have a saved "previous line/triplet"?
                    let x = prev_triplet.clone().unwrap();
                    angle = calculate_azimuth_inclination(&curr_triplet, &x);
                }
                azis[i].push(angle.1);
                incs[i].push(angle.2);

                //println!("{:?} {}", curr_triplet, dist); //, dist_3d_t(&curr_triplet, &prev_triplet));
                prev_triplet = Some(curr_triplet);
            }
        }

        self.azimuths = Some(azis);
        self.inclinations = Some(incs);
    }

    // ---

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

    pub fn calculate_mean_velocities(&mut self) {
        if self.mean_velocities.is_none() {
            let mut mean_velocities: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let velocities = self.velocities.as_mut().unwrap(); // Vector with the velocities
            for d in velocities {
                let mean_d = mean(&d);
                mean_velocities.push(mean_d);
            }
            self.mean_velocities = Some(mean_velocities);
        }
    }

    pub fn calculate_stdev_velocities(&mut self) {
        if self.stdev_velocities.is_none() {
            let mut stdev_velocities: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let velocities = self.velocities.as_mut().unwrap(); // Vector with the velocities
            for d in velocities {
                let stdev_d = standard_dev(&d);
                stdev_velocities.push(stdev_d);
            }
            self.stdev_velocities = Some(stdev_velocities);
        }
    }

    pub fn calculate_mean_accelerations(&mut self) {
        if self.mean_accelerations.is_none() {
            let mut mean_accelerations: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let accelerations = self.accelerations.as_mut().unwrap(); // Vector with the accelerations
            for d in accelerations {
                let mean_d = mean(&d);
                mean_accelerations.push(mean_d);
            }
            self.mean_accelerations = Some(mean_accelerations);
        }
    }

    pub fn calculate_stdev_accelerations(&mut self) {
        if self.stdev_accelerations.is_none() {
            let mut stdev_accelerations: SensorData = vec![];
            // .as_mut() returns a mutable reference.
            let accelerations = self.accelerations.as_mut().unwrap(); // Vector with the accelerations
            for d in accelerations {
                let stdev_d = standard_dev(&d);
                stdev_accelerations.push(stdev_d);
            }
            self.stdev_accelerations = Some(stdev_accelerations);
        }
    }

    // Getters

    pub fn get_frames(&self) -> &Frames {
        self.frames.as_ref().unwrap()
    }

    pub fn get_frame_number(&self, frame_no: usize) -> &usize {
        self.frame_numbers
            .as_ref()
            .unwrap()
            .get(frame_no) // Get the value for the f-th frame.
            .unwrap()
    }

    pub fn get_timestamp(&self, frame_no: usize) -> &usize {
        self.timestamps
            .as_ref()
            .unwrap()
            .get(frame_no) // Get the value for the f-th frame.
            .unwrap()
    }

    pub fn get_distance(&self, sensor_id: usize, frame_no: usize) -> &SensorFloat {
        self.distances
            .as_ref()
            .unwrap()
            .get(sensor_id) // Get the data for the i-th sensor.
            .unwrap()
            .get(frame_no) // Get the value for the f-th frame.
            .unwrap()
    }
    pub fn get_min_distance(&self, sensor_id: usize) -> &SensorFloat {
        self.min_distances
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }
    pub fn get_max_distance(&self, sensor_id: usize) -> &SensorFloat {
        self.max_distances
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }
    pub fn get_mean_distance(&self, sensor_id: usize) -> &SensorFloat {
        self.mean_distances
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }
    pub fn get_stdev_distance(&self, sensor_id: usize) -> &SensorFloat {
        self.stdev_distances
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }

    pub fn get_velocity(&self, sensor_id: usize, frame_no: usize) -> &SensorFloat {
        self.velocities
            .as_ref()
            .unwrap()
            .get(sensor_id) // Get the data for the i-th sensor.
            .unwrap()
            .get(frame_no) // Get the value for the f-th frame.
            .unwrap()
    }

    // Convert to m/s from frequency. The stored value is in "frame" units.
    pub fn get_velocity_ms(&self, sensor_id: usize, frame_no: usize) -> SensorFloat {
        let v = self
            .velocities
            .as_ref()
            .unwrap()
            .get(sensor_id) // Get the data for the i-th sensor.
            .unwrap()
            .get(frame_no) // Get the value for the f-th frame.
            .unwrap();
        *v * self.frequency as f32
    }

    // Get all the min velocities for one sensor.
    pub fn get_min_velocity(&self, sensor_id: usize) -> &SensorFloat {
        self.min_velocities
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }
    pub fn get_max_velocity(&self, sensor_id: usize) -> &SensorFloat {
        self.max_velocities
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }
    pub fn get_mean_velocity(&self, sensor_id: usize) -> &SensorFloat {
        self.mean_velocities
            .as_ref()
            .unwrap()
            .get(sensor_id)
            .unwrap()
    }
    pub fn get_stdev_velocity(&self, sensor_id: usize) -> &SensorFloat {
        self.stdev_velocities
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }

    // Get one acceleration for a sensor and frame.
    pub fn get_acceleration(&self, sensor_id: usize, frame_no: usize) -> &SensorFloat {
        self.accelerations
            .as_ref()
            .unwrap()
            .get(sensor_id) // Get the data for the i-th sensor.
            .unwrap()
            .get(frame_no) // Get the value for the f-th frame.
            .unwrap()
    }

    pub fn get_acceleration_ms(&self, sensor_id: usize, frame_no: usize) -> SensorFloat {
        let a = self
            .accelerations
            .as_ref()
            .unwrap()
            .get(sensor_id) // Get the data for the i-th sensor.
            .unwrap()
            .get(frame_no) // Get the value for the f-th frame.
            .unwrap();
        *a * self.frequency as f32
    }

    pub fn get_min_acceleration(&self, sensor_id: usize) -> &SensorFloat {
        self.min_accelerations
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }
    pub fn get_max_acceleration(&self, sensor_id: usize) -> &SensorFloat {
        self.max_accelerations
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }
    pub fn get_mean_acceleration(&self, sensor_id: usize) -> &SensorFloat {
        self.mean_accelerations
            .as_ref()
            .unwrap()
            .get(sensor_id)
            .unwrap()
    }
    pub fn get_stdev_acceleration(&self, sensor_id: usize) -> &SensorFloat {
        self.stdev_accelerations
            .as_ref()
            .unwrap()
            .get(sensor_id) // The minimum value of the i-th sensor data.
            .unwrap()
    }

    pub fn get_azimuth(&self, sensor_id: usize, frame_no: usize) -> &SensorFloat {
        self.azimuths
            .as_ref()
            .unwrap()
            .get(sensor_id)
            .unwrap()
            .get(frame_no)
            .unwrap()
    }

    pub fn get_inclination(&self, sensor_id: usize, frame_no: usize) -> &SensorFloat {
        self.inclinations
            .as_ref()
            .unwrap()
            .get(sensor_id)
            .unwrap()
            .get(frame_no)
            .unwrap()
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

// These are used if we request the header in the output.
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

/// A generalised means calculator.
pub fn calculate_means(data: &Vec<SensorData>) -> SensorData {
    let mut means: SensorData = vec![];
    for d in data {
        let mean = mean(&d);
        means.push(mean);
    }
    means
}

/// A general version of the calculate_stdev function.
pub fn calculate_stdevs(data: &Vec<SensorData>) -> SensorData {
    let mut stdevs: SensorData = vec![];
    for d in data {
        let stdev_d = standard_dev(&d);
        stdevs.push(stdev_d);
    }
    stdevs
}

pub fn extract_values(bits: &Vec<f32>, keep: &Vec<usize>) -> Vec<f32> {
    keep.iter()
        .filter_map(|&index| {
            let start = index * 3;
            let end = start + 3;
            if end <= bits.len() {
                Some(&bits[start..end])
            } else {
                None
            }
        })
        .flatten()
        .cloned()
        .collect()
}

pub fn extract_values_inplace(bits: &mut Vec<f32>, keep: &Vec<usize>) {
    let mut to_retain = vec![false; bits.len()];

    for &index in keep {
        let start = index * 3;
        if start + 2 < bits.len() {
            to_retain[start] = true;
            to_retain[start + 1] = true;
            to_retain[start + 2] = true;
        }
    }

    let mut i = 0;
    bits.retain(|_| {
        let keep = to_retain[i];
        i += 1;
        keep
    });
}
