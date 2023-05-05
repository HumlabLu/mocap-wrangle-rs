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
}
