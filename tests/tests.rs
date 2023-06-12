#[cfg(test)]
mod tests {
    use mocap::{dist_3d, dist_3d_t, MoCapFile};
        
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
    fn test_dist_t() {
	let t0 = vec![0.0,0.0,0.0];
	let t1 = vec![0.0,0.0,0.0];
	let dist = dist_3d_t(&t0, &t1);
	assert!(dist==0.0);
    }

    #[test]
    fn test_normal_dist_t() {
	let t0 = vec![0.0,0.0,0.0];
	let t1 = vec![1.0,0.0,0.0];
	let dist = dist_3d_t(&t0, &t1);
	assert!(dist==1.0);
    }

    #[test]
    #[should_panic]
    fn test_wrong_params_lhs() {
	let _dist = dist_3d(&[1.0,0.0,0.0,4.0], &[0.0,0.0,0.0]);
    }

    #[test]
    fn test_dist_wrong_params_rhs() {
	let result = std::panic::catch_unwind(|| dist_3d(&[1.0,0.0,0.0], &[0.0,0.0,0.0,4.0]));
	assert!(result.is_err()); 
    }

    #[test]
    fn struct_is_valid() {
	let myfile = MoCapFile {
	    filename: String::new(),
	    out_filename: String::new(),
	    num_header_lines: 0,
	    num_matches: 0,
	    num_frames: 0,
	    checked_header: false,
	    no_of_frames: 0,
	    no_of_cameras: 0,
	    no_of_markers: 0,
	    frequency: 200,
	    no_of_analog: 0,
	    description: String::new(),
	    time_stamp: String::new(),
	    data_included: String::new(),
	    marker_names: vec!["X".to_string()],
	    frames: None,
	    distances: None,
	    velocities: None,
	    accelerations: None,
	    ..Default::default()
	};
	assert!(myfile.is_valid()==true);
    }

    #[test]
    fn struct_is_invalid() {
	let myfile = MoCapFile {
	    filename: String::new(),
	    out_filename: String::new(),
	    num_header_lines: 0,
	    num_matches: 0,
	    num_frames: 0,
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
	    distances: None,
	    velocities: None,
	    accelerations: None,
	    ..Default::default()
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

    #[test]
    fn extract_description() {
	let result = mocap::extract_description("DESCRIPTION\tThe answer is 42!").unwrap();
	assert!(result=="The answer is 42!");
    }

    #[test]
    fn extract_data_included() {
	let result = mocap::extract_data_included("DATA_INCLUDED\t3D").unwrap();
	assert!(result=="3D");
    }

    // From the basic example at https://en.wikipedia.org/wiki/Standard_deviation.
    #[test]
    fn mean() {
	let result = mocap::mean(&vec![2.0,4.0,4.0,4.0,5.0,5.0,7.0,9.0]);
	assert!(result==5.0);
    }
    
    #[test]
    fn variance() {
	let result = mocap::variance(&vec![2.0,4.0,4.0,4.0,5.0,5.0,7.0,9.0]);
	assert!(result==4.0);
    }
    
    #[test]
    fn standard_dev() {
	let result = mocap::standard_dev(&vec![2.0,4.0,4.0,4.0,5.0,5.0,7.0,9.0]);
	assert!(result==2.0);
    }

    #[test]
    fn standardise() {
	let result = mocap::standardise(&190.0,&150.0,&25.0);
	assert!(result==1.6);
    }
    
    #[test]
    fn calculate_azimuth_inclination() {
	let t0 = vec![5.0,6.7,1.5];
	let t1 = vec![4.0,1.2,1.6];
	let result = mocap::calculate_azimuth_inclination(&t0, &t1);
	assert!(result==(5.5910645, -1.7506498, 1.5529097));
    }
}
