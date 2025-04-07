use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn powerfit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rotate_grid3d, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn rotate_grid3d(
    grid: Vec<f32>,
    rotmat: Vec<f32>,
    mut out: Vec<f32>,
    nearest: bool,
    shape: (usize, usize, usize),
    llength: i32,
) {
    let (shape_x, shape_y, shape_z) = shape;
    let slice = shape_x * shape_y;
    let llength_sq = (llength * llength) as f32;

    for z in -llength..=llength {
        for y in -llength..=llength {
            for x in -llength..=llength {
                let r_sq = (x * x + y * y + z * z) as f32;
                if r_sq > llength_sq {
                    continue;
                }

                // Apply rotation matrix
                let x_rot = rotmat[0] * x as f32 + rotmat[3] * y as f32 + rotmat[6] * z as f32;
                let y_rot = rotmat[1] * x as f32 + rotmat[4] * y as f32 + rotmat[7] * z as f32;
                let z_rot = rotmat[2] * x as f32 + rotmat[5] * y as f32 + rotmat[8] * z as f32;

                let value = if nearest {
                    // Nearest-neighbor interpolation
                    let x0 = x_rot.round() as i32;
                    let y0 = y_rot.round() as i32;
                    let z0 = z_rot.round() as i32;

                    let x0_adj = x0.rem_euclid(shape_x as i32) as usize;
                    let y0_adj = y0.rem_euclid(shape_y as i32) as usize;
                    let z0_adj = z0.rem_euclid(shape_z as i32) as usize;

                    grid[z0_adj * slice + y0_adj * shape_x + x0_adj]
                } else {
                    // Trilinear interpolation
                    let x0 = x_rot.floor() as i32;
                    let y0 = y_rot.floor() as i32;
                    let z0 = z_rot.floor() as i32;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    let z1 = z0 + 1;

                    let dx = x_rot - x0 as f32;
                    let dy = y_rot - y0 as f32;
                    let dz = z_rot - z0 as f32;
                    let dx1 = 1.0 - dx;
                    let dy1 = 1.0 - dy;
                    let dz1 = 1.0 - dz;

                    // Adjust indices with wrapping
                    let x0_adj = x0.rem_euclid(shape_x as i32) as usize;
                    let x1_adj = x1.rem_euclid(shape_x as i32) as usize;
                    let y0_adj = y0.rem_euclid(shape_y as i32) as usize;
                    let y1_adj = y1.rem_euclid(shape_y as i32) as usize;
                    let z0_adj = z0.rem_euclid(shape_z as i32) as usize;
                    let z1_adj = z1.rem_euclid(shape_z as i32) as usize;

                    // Fetch surrounding values
                    let c000 = grid[z0_adj * slice + y0_adj * shape_x + x0_adj];
                    let c001 = grid[z0_adj * slice + y0_adj * shape_x + x1_adj];
                    let c010 = grid[z0_adj * slice + y1_adj * shape_x + x0_adj];
                    let c011 = grid[z0_adj * slice + y1_adj * shape_x + x1_adj];
                    let c100 = grid[z1_adj * slice + y0_adj * shape_x + x0_adj];
                    let c101 = grid[z1_adj * slice + y0_adj * shape_x + x1_adj];
                    let c110 = grid[z1_adj * slice + y1_adj * shape_x + x0_adj];
                    let c111 = grid[z1_adj * slice + y1_adj * shape_x + x1_adj];

                    // Interpolate
                    let c00 = c000 * dx1 + c001 * dx;
                    let c01 = c010 * dx1 + c011 * dx;
                    let c10 = c100 * dx1 + c101 * dx;
                    let c11 = c110 * dx1 + c111 * dx;

                    let c0 = c00 * dy1 + c01 * dy;
                    let c1 = c10 * dy1 + c11 * dy;

                    c0 * dz1 + c1 * dz
                };

                // Compute output index
                let x_out = x.rem_euclid(shape_x as i32) as usize;
                let y_out = y.rem_euclid(shape_y as i32) as usize;
                let z_out = z.rem_euclid(shape_z as i32) as usize;
                let out_idx = z_out * slice + y_out * shape_x + x_out;

                out[out_idx] = value;
            }
        }
    }
}

#[pyfunction]
pub fn rotate_image3d(
    image: Vec<f32>,
    rotmat: Vec<f32>,
    mut out: Vec<f32>,
    shape: (usize, usize, usize),
    llength: i32,
) {
    let (shape_x, shape_y, shape_z) = shape;
    let slice = shape_x * shape_y;
    let llength_sq = (llength * llength) as f32;

    for z in -llength..=llength {
        for y in -llength..=llength {
            for x in -llength..=llength {
                let r_sq = (x * x + y * y + z * z) as f32;
                if r_sq > llength_sq {
                    continue;
                }

                // Apply rotation matrix with offset
                let x_rot =
                    rotmat[0] * x as f32 + rotmat[3] * y as f32 + rotmat[6] * z as f32 + 0.5;
                let y_rot =
                    rotmat[1] * x as f32 + rotmat[4] * y as f32 + rotmat[7] * z as f32 + 0.5;
                let z_rot =
                    rotmat[2] * x as f32 + rotmat[5] * y as f32 + rotmat[8] * z as f32 + 0.5;

                // Normalize coordinates
                let nx = x_rot / shape_x as f32;
                let ny = y_rot / shape_y as f32;
                let nz = z_rot / shape_z as f32;

                // Convert back to grid coordinates
                let x_coord = nx * shape_x as f32 - 0.5;
                let y_coord = ny * shape_y as f32 - 0.5;
                let z_coord = nz * shape_z as f32 - 0.5;

                // Trilinear interpolation
                let x0 = x_coord.floor() as i32;
                let y0 = y_coord.floor() as i32;
                let z0 = z_coord.floor() as i32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let z1 = z0 + 1;

                let dx = x_coord - x0 as f32;
                let dy = y_coord - y0 as f32;
                let dz = z_coord - z0 as f32;
                let dx1 = 1.0 - dx;
                let dy1 = 1.0 - dy;
                let dz1 = 1.0 - dz;

                // Adjust indices with wrapping
                let x0_adj = x0.rem_euclid(shape_x as i32) as usize;
                let x1_adj = x1.rem_euclid(shape_x as i32) as usize;
                let y0_adj = y0.rem_euclid(shape_y as i32) as usize;
                let y1_adj = y1.rem_euclid(shape_y as i32) as usize;
                let z0_adj = z0.rem_euclid(shape_z as i32) as usize;
                let z1_adj = z1.rem_euclid(shape_z as i32) as usize;

                // Fetch surrounding values
                let c000 = image[z0_adj * slice + y0_adj * shape_x + x0_adj];
                let c001 = image[z0_adj * slice + y0_adj * shape_x + x1_adj];
                let c010 = image[z0_adj * slice + y1_adj * shape_x + x0_adj];
                let c011 = image[z0_adj * slice + y1_adj * shape_x + x1_adj];
                let c100 = image[z1_adj * slice + y0_adj * shape_x + x0_adj];
                let c101 = image[z1_adj * slice + y0_adj * shape_x + x1_adj];
                let c110 = image[z1_adj * slice + y1_adj * shape_x + x0_adj];
                let c111 = image[z1_adj * slice + y1_adj * shape_x + x1_adj];

                // Interpolate
                let c00 = c000 * dx1 + c001 * dx;
                let c01 = c010 * dx1 + c011 * dx;
                let c10 = c100 * dx1 + c101 * dx;
                let c11 = c110 * dx1 + c111 * dx;

                let c0 = c00 * dy1 + c01 * dy;
                let c1 = c10 * dy1 + c11 * dy;

                let value = c0 * dz1 + c1 * dz;

                // Compute output index
                let x_out = x.rem_euclid(shape_x as i32) as usize;
                let y_out = y.rem_euclid(shape_y as i32) as usize;
                let z_out = z.rem_euclid(shape_z as i32) as usize;
                let out_idx = z_out * slice + y_out * shape_x + x_out;

                out[out_idx] = value;
            }
        }
    }
}
