use std::{env, fs, path::PathBuf};

use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let source_path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: compile_ptx <source.cu> <destination.ptx>")?,
    );
    let destination_path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: compile_ptx <source.cu> <destination.ptx>")?,
    );
    if arguments.next().is_some() {
        return Err("usage: compile_ptx <source.cu> <destination.ptx>".into());
    }

    let source = fs::read_to_string(&source_path)?;
    let ptx = compile_ptx_with_opts(
        source,
        CompileOptions {
            arch: Some("compute_70"),
            fmad: Some(true),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            ..CompileOptions::default()
        },
    )?;
    fs::write(destination_path, ptx.to_src())?;
    Ok(())
}
