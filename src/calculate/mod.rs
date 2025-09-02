use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

use image::GenericImageView;
use pathfinding::prelude::Weights;
use std::sync::mpsc;

pub struct GenerationSettings {
    proximity_importance: i64,
    rescale: Option<u32>,
}

impl GenerationSettings {
    pub fn default() -> Self {
        Self {
            proximity_importance: 13,
            rescale: None,
        }
    }

    pub fn quick_process() -> Self {
        Self {
            proximity_importance: 10,
            rescale: Some(64),
        }
    }
}

struct ImgDiffWeights {
    source: Vec<(u8, u8, u8)>,
    target: Vec<(u8, u8, u8)>,
    weights: Vec<i64>,
    sidelen: usize,
    settings: GenerationSettings,
}

const TARGET_IMAGE_PATH: &str = "./target.png";
const TARGET_WEIGHTS_PATH: &str = "./weights.png";

impl Weights<i64> for ImgDiffWeights {
    fn rows(&self) -> usize {
        self.target.len()
    }

    fn columns(&self) -> usize {
        self.source.len()
    }

    fn at(&self, row: usize, col: usize) -> i64 {
        let (x1, y1) = (row % self.sidelen, row / self.sidelen);
        let (x2, y2) = (col % self.sidelen, col / self.sidelen);
        let dist = (x1 as i64 - x2 as i64).pow(2) + (y1 as i64 - y2 as i64).pow(2);

        let (r1, g1, b1) = self.target[row];
        let (r2, g2, b2) = self.source[col];

        let dr = r1 as i64 - r2 as i64;
        let dg = g1 as i64 - g2 as i64;
        let db = b1 as i64 - b2 as i64;

        -((dr.pow(2) + dg.pow(2) + db.pow(2)) * self.weights[row]
            + (dist * self.settings.proximity_importance).pow(2))
    }

    fn neg(&self) -> Self {
        todo!()
    }
}

pub enum ProgressMsg {
    Progress(f32),
    Done(PathBuf), // result directory
    Error(String),
    Cancelled,
}


pub fn process<P: AsRef<Path>>(
    source_path: P,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    cancelled: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let start_time = std::time::Instant::now();
    let mut target = image::open(TARGET_IMAGE_PATH)?.to_rgb8();
    let mut target_weights = image::open(TARGET_WEIGHTS_PATH)?.to_rgb8();
    if target.dimensions().0 != target.dimensions().1 {
        return Err("Target image must be square".into());
    }
    if target.dimensions() != target_weights.dimensions() {
        return Err("Target and weights images must have the same dimensions".into());
    }

    if let Some(rescale) = settings.rescale {
        target = image::imageops::resize(
            &target,
            rescale,
            rescale,
            image::imageops::FilterType::Lanczos3,
        );
        target_weights = image::imageops::resize(
            &target_weights,
            rescale,
            rescale,
            image::imageops::FilterType::Lanczos3,
        );
    }
    let base_name = source_path
        .as_ref()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let source = image::open(source_path)?.to_rgb8();
    // rescale source to match target dimensions
    let source = image::imageops::resize(
        &source,
        target.width(),
        target.height(),
        image::imageops::FilterType::Lanczos3,
    );

    let source_pixels = source
        .pixels()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<Vec<_>>();

    let weights = ImgDiffWeights {
        source: source_pixels.clone(),
        target: target.pixels().map(|p| (p[0], p[1], p[2])).collect(),
        weights: load_weights(&target_weights.into()),
        sidelen: target.width() as usize,
        settings,
    };

    let assignments = match auction_assign(&weights, &tx, &cancelled) {
        Ok(a) => a,
        Err(()) => return Ok(()), // already sent Cancelled
    };

    let mut img = image::ImageBuffer::new(target.width(), target.height());

    for (target_idx, source_idx) in assignments.iter().enumerate() {
        let (x, y) = (
            (target_idx % target.width() as usize) as u32,
            (target_idx / target.width() as usize) as u32,
        );
        let (r, g, b) = source_pixels[*source_idx];
        img.put_pixel(x, y, image::Rgb([r, g, b]));
    }

    let mut dir_name = base_name.clone();
    let mut counter = 1;

    while std::path::Path::new(&format!("./presets/{}", dir_name)).exists() {
        dir_name = format!("{}_{}", base_name, counter);
        counter += 1;
    }

    std::fs::create_dir_all(format!("./presets/{}", dir_name))?;
    img.save(format!("./presets/{}/output.png", dir_name))?;
    source.save(format!("./presets/{}/source.png", dir_name))?;
    target.save(format!("./presets/{}/target.png", dir_name))?;
    std::fs::write(
        format!("./presets/{}/assignments.json", dir_name),
        serialize_assignments(assignments),
    )?;

    tx.send(ProgressMsg::Done(PathBuf::from(format!(
        "./presets/{}",
        dir_name
    ))))?;

    println!(
        "finished in {:.2?} seconds",
        std::time::Instant::now().duration_since(start_time)
    );

    Ok(())
}

fn load_weights(source: &image::DynamicImage) -> Vec<i64> {
    let (width, height) = source.dimensions();
    let mut weights = vec![0; (width * height) as usize];
    for (x, y, pixel) in source.pixels() {
        let weight = pixel[0] as i64;
        weights[(y * width + x) as usize] = weight;
    }
    weights
}

fn serialize_assignments(assignments: Vec<usize>) -> String {
    format!(
        "[{}]",
        assignments
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn auction_assign(
    weights: &ImgDiffWeights,
    tx: &mpsc::SyncSender<ProgressMsg>,
    cancelled: &Arc<AtomicBool>,
) -> Result<Vec<usize>, ()> {
    let n = weights.rows();
    let m = weights.columns();
    assert_eq!(n, m, "assignment requires square matrix");

    // Window radius for candidate pruning
    let sidelen = weights.sidelen as i32;
    let win: i32 = if sidelen <= 64 {
        6
    } else if sidelen <= 128 {
        10
    } else if sidelen <= 256 {
        14
    } else {
        20
    };

    // Auction variables
    let epsilon: i64 = 1; // integer epsilon
    let mut price = vec![0i64; m];
    let mut owner: Vec<Option<usize>> = vec![None; m]; // which row owns column j
    let mut assignment: Vec<Option<usize>> = vec![None; n]; // assignment for row i -> col j

    let mut unassigned: Vec<usize> = (0..n).collect();
    let mut assigned_count: usize = 0;

    // Helper to compute best and second-best candidate for row i
    let mut iter_counter: usize = 0;
    while let Some(i) = unassigned.pop() {
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tx.send(ProgressMsg::Cancelled);
            return Err(());
        }

        // Extract (x, y) for this row to build a local window
        let x1 = (i % (weights.sidelen)) as i32;
        let y1 = (i / (weights.sidelen)) as i32;
        let xmin = (x1 - win).max(0);
        let xmax = (x1 + win).min(sidelen - 1);
        let ymin = (y1 - win).max(0);
        let ymax = (y1 + win).min(sidelen - 1);

        let mut best_j: usize = 0;
        let mut best_v: i64 = i64::MIN;
        let mut second_v: i64 = i64::MIN;

        for yy in ymin..=ymax {
            let base = (yy as usize) * weights.sidelen;
            for xx in xmin..=xmax {
                let j = base + (xx as usize);
                // Score is benefit minus price (maximization)
                let v = weights.at(i, j) - price[j];
                if v > best_v {
                    second_v = best_v;
                    best_v = v;
                    best_j = j;
                } else if v > second_v {
                    second_v = v;
                }
            }
        }

        if best_v == i64::MIN {
            // Fallback: scan entire row to avoid dead-ends (shouldn't happen)
            for j in 0..m {
                let v = weights.at(i, j) - price[j];
                if v > best_v {
                    second_v = best_v;
                    best_v = v;
                    best_j = j;
                } else if v > second_v {
                    second_v = v;
                }
            }
        }

        if second_v == i64::MIN {
            // If there is only one candidate, make a minimal bid
            second_v = best_v - 1;
        }

        // Bid and assign
        let bid = (best_v - second_v) + epsilon;
        price[best_j] = price[best_j] + bid;

        if let Some(prev_i) = owner[best_j] {
            // Kick out previous owner
            assignment[prev_i] = None;
            unassigned.push(prev_i);
        } else {
            assigned_count += 1;
        }
        owner[best_j] = Some(i);
        assignment[i] = Some(best_j);

        iter_counter += 1;
        if iter_counter % 1024 == 0 || assigned_count % (n.max(1) / 100 + 1) == 0 {
            let _ = tx.send(ProgressMsg::Progress(assigned_count as f32 / n as f32));
        }
    }

    Ok(assignment.into_iter().map(|o| o.unwrap()).collect())
}
