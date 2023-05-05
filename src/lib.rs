/// The float type for the coordinates/velocities, etc.
/// Note that there are different sizes of integers as well.
/// # (Maybe use usize for all the integers?)
pub type SensorFloat = f32;

/// Calculate the distance in 3D.
///
/// # Example
///
/// ```rust
/// let dist = dist_3d(&[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
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

/// Create a new output filename, tries to append "_d3D" to
/// the filename.
///
/// Short input filenames will return `output_d3D.tsv`.
///
pub fn create_outputfilename(filename: &str) -> String {
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
