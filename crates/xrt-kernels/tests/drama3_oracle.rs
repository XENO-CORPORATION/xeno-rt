use xrt_kernels::cpu::{
    delta_state::{GatedDeltaState, HeadExpansion},
    fp8::BlockFp8Matrix,
};
fn floats(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 2e-5 * b.abs().max(1.0), "{a} != {b}");
    }
}
#[test]
fn fp8_group_quantization_matches_actual_triton_rounding_and_partial_group() {
    use xrt_kernels::cpu::fp8::{decode_e4m3fn, quantize_e4m3fn};
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../reference/audio/drama3/fp8-quantization-fixture.json"
    ))
    .unwrap();
    let input = floats(&fixture["input"]);
    let (encoded, scales) = quantize_e4m3fn(&input, 128).unwrap();
    assert_eq!(scales, floats(&fixture["scales"]));
    assert_eq!(
        encoded.into_iter().map(decode_e4m3fn).collect::<Vec<_>>(),
        floats(&fixture["quantized"])
    );
    let precise: Vec<f32> = input
        .chunks(128)
        .map(|v| v.iter().fold(1e-10f32, |m, x| m.max(x.abs())) / 448.0)
        .collect();
    assert!(
        precise != scales,
        "fixture must reject the old precise division"
    );
}
#[test]
fn packed_causal_convolution_matches_pytorch_padding_groups_and_transpose() {
    use xrt_kernels::cpu::causal_conv::PackedCausalConv1d;
    let all: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/drama3_numeric.json")).unwrap();
    for f in all["causal_conv"].as_array().unwrap() {
        let n = |name: &str| f[name].as_u64().unwrap() as usize;
        let conv = PackedCausalConv1d::new(
            &floats(&f["weights"]),
            n("input_channels"),
            n("output_channels"),
            n("kernel"),
            n("stride"),
            n("dilation"),
            n("groups"),
            f["transpose"].as_bool().unwrap(),
            1024 * 1024,
        )
        .unwrap();
        let expected = floats(&f["expected"]);
        let mut output = vec![0.0; expected.len()];
        conv.apply(
            &floats(&f["input"]),
            n("frames"),
            &floats(&f["bias"]),
            &mut output,
        )
        .unwrap();
        close(&output, &expected);
    }
}
#[test]
fn fp8_and_w8a8_match_pytorch_oracle() {
    let all: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/drama3_numeric.json")).unwrap();
    let f = &all["fp8"];
    let bytes: Vec<_> = f["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u8)
        .collect();
    let scales = floats(&f["scales"]);
    let rows = f["rows"].as_u64().unwrap() as usize;
    let cols = f["cols"].as_u64().unwrap() as usize;
    let edge = f["edge"].as_u64().unwrap() as usize;
    let matrix = BlockFp8Matrix::new(&bytes, &scales, rows, cols, edge, edge).unwrap();
    let mut out = vec![0.0; rows];
    let input = floats(&f["input"]);
    matrix.matvec(&input, &mut out).unwrap();
    close(&out, &floats(&f["expected"]));
    matrix.matvec_w8a8(&input, &mut out).unwrap();
    close(&out, &floats(&f["w8a8"]));
}
#[test]
fn recurrent_steps_and_cache_match_pytorch_oracle() {
    let all: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/drama3_numeric.json")).unwrap();
    let f = &all["delta"];
    let kh = f["key_heads"].as_u64().unwrap() as usize;
    let vh = f["value_heads"].as_u64().unwrap() as usize;
    let kd = f["key_dim"].as_u64().unwrap() as usize;
    let vd = f["value_dim"].as_u64().unwrap() as usize;
    let mut state =
        GatedDeltaState::new(kh, vh, kd, vd, HeadExpansion::RepeatInterleave, 1024).unwrap();
    let mut ordered = state.clone();
    let q = floats(&f["query"]);
    let k = floats(&f["key"]);
    let v = floats(&f["value"]);
    let g = floats(&f["decay"]);
    let beta = floats(&f["beta"]);
    for (i, expected) in f["outputs"].as_array().unwrap().iter().enumerate() {
        let mut out = vec![0.0; vh * vd];
        state
            .step(
                &q[i * kh * kd..][..kh * kd],
                &k[i * kh * kd..][..kh * kd],
                &v[i * vh * vd..][..vh * vd],
                &g[i * vh..][..vh],
                &beta[i * vh..][..vh],
                &mut out,
            )
            .unwrap();
        close(&out, &floats(expected));
        ordered
            .step_ordered(
                &q[i * kh * kd..][..kh * kd],
                &k[i * kh * kd..][..kh * kd],
                &v[i * vh * vd..][..vh * vd],
                &g[i * vh..][..vh],
                &beta[i * vh..][..vh],
                &mut out,
            )
            .unwrap();
        close(&out, &floats(expected));
    }
    close(state.snapshot(), &floats(&f["final_state_value_major"]));
    close(ordered.snapshot(), &floats(&f["final_state_value_major"]));
}
