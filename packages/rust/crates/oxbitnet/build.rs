use std::fs;
use std::path::Path;

fn main() {
    let shader_src = Path::new("../../packages/core/src/shaders");
    // Fallback: try relative to workspace root
    let shader_src = if shader_src.exists() {
        shader_src.to_path_buf()
    } else {
        // When building from packages/rust/crates/oxbitnet
        Path::new("../../../core/src/shaders").to_path_buf()
    };

    let out_dir = Path::new("src/shaders");
    fs::create_dir_all(out_dir).expect("Failed to create shaders output dir");

    if shader_src.exists() {
        for entry in glob::glob(shader_src.join("*.wgsl").to_str().unwrap())
            .expect("Failed to glob shaders")
        {
            let path = entry.expect("Failed to read shader path");
            let filename = path.file_name().unwrap();
            let dest = out_dir.join(filename);
            fs::copy(&path, &dest).expect("Failed to copy shader");
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    if shader_src.exists() {
        println!(
            "cargo:rerun-if-changed={}",
            shader_src.to_str().unwrap()
        );
    }
}
