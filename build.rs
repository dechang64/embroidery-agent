fn main() {
    tonic_build::compile_protos("proto/embroidery.proto")
        .expect("Failed to compile proto");
}
