#[cfg(test)]
mod tests {
    use mocap::{dist_3d, MoCapFile};
    use mocap::SensorFloat;
    
    
    #[test]
    fn test_zero_dist() {
	let dist = dist_3d(&[0.0,0.0,0.0], &[0.0,0.0,0.0]);
	assert!(dist==0.0);
    }

    #[test]
    fn test_normal_dist() {
	let dist = dist_3d(&[1.0,0.0,0.0], &[0.0,0.0,0.0]);
	assert!(dist==1.0);
    }

    #[test]
    #[should_panic]
    fn test_wrong_params_lhs() {
	let dist = dist_3d(&[1.0,0.0,0.0,4.0], &[0.0,0.0,0.0]);
    }

    #[test]
    fn test_dist_wrong_params_rhs() {
	let result = std::panic::catch_unwind(|| dist_3d(&[1.0,0.0,0.0], &[0.0,0.0,0.0,4.0]));
	assert!(result.is_err()); 
    }

    #[test]
    fn struct_is_valid() {
	let mut myfile = MoCapFile {
	    name: String::new(),
	    no_of_frames: 0,
	    no_of_cameras: 0,
	    no_of_markers: 0,
	    frequency: 200,
	    no_of_analog: 0,
	    description: String::new(),
	    time_stamp: String::new(),
	    data_included: String::new(),
	    marker_names: vec!["X".to_string()],
	};
	assert!(myfile.is_valid()==true);
    }

    #[test]
    fn struct_is_invalid() {
	let mut myfile = MoCapFile {
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
	};
	assert!(myfile.is_valid()==false);
    }

    // Regular expressions
    
    #[test]
    fn extract_frequency() {
	let result = mocap::extract_frequency("FREQUENCY\t28").unwrap();
	assert!(result==28);
    }

    #[test]
    fn extract_no_of_cameras() {
	let result = mocap::extract_no_of_cameras("NO_OF_CAMERAS\t28").unwrap();
	assert!(result==28);
    }

    #[test]
    fn extract_no_of_markers() {
	let result = mocap::extract_no_of_markers("NO_OF_MARKERS\t28").unwrap();
	assert!(result==28);
    }

    #[test]
    fn extract_time_stamp() {
	let result = mocap::extract_time_stamp("TIME_STAMP\t2022-06-03, 10:47:36.627\t94247.45402301").unwrap();
	assert!(result=="2022-06-03, 10:47:36.627");

	let result = mocap::extract_time_stamp("TIME_STAMP\t2022-11-22, 22:00:35").unwrap();
	assert!(result=="2022-11-22, 22:00:35");
    }

    #[test]
    fn extract_marker_names() {
	let result = mocap::extract_marker_names("MARKER_NAMES\tx_HeadL\tx_HeadTop\tx_HeadR").unwrap();
	assert!(result==vec!["x_HeadL","x_HeadTop","x_HeadR"]);
    }
    
    /*
    mocap::extract_no_of_cameras(&l)
	mocap::extract_no_of_markers(&l) {
	    mocap::extract_frequency(&l) {
		mocap::extract_marker_names(&l) {
		    mocap::extract_timestamp(&l) {
    mocap::extract_description(&l) {
    */
}
