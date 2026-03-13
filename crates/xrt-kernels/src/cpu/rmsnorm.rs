pub fn apply_rmsnorm(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    assert_eq!(input.len(), weight.len());
    assert_eq!(input.len(), output.len());

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    let mut input_chunks = input.chunks_exact(8);
    for chunk in input_chunks.by_ref() {
        sum0 = chunk[0].mul_add(chunk[0], sum0);
        sum1 = chunk[1].mul_add(chunk[1], sum1);
        sum2 = chunk[2].mul_add(chunk[2], sum2);
        sum3 = chunk[3].mul_add(chunk[3], sum3);
        sum4 = chunk[4].mul_add(chunk[4], sum4);
        sum5 = chunk[5].mul_add(chunk[5], sum5);
        sum6 = chunk[6].mul_add(chunk[6], sum6);
        sum7 = chunk[7].mul_add(chunk[7], sum7);
    }

    let mut sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    for &value in input_chunks.remainder() {
        sum = value.mul_add(value, sum);
    }

    let mean_square = sum / input.len() as f32;
    let inv_rms = 1.0 / (mean_square + eps).sqrt();

    let mut input_chunks = input.chunks_exact(8);
    let mut weight_chunks = weight.chunks_exact(8);
    let mut output_chunks = output.chunks_exact_mut(8);

    for ((input_chunk, weight_chunk), output_chunk) in input_chunks
        .by_ref()
        .zip(weight_chunks.by_ref())
        .zip(output_chunks.by_ref())
    {
        output_chunk[0] = (input_chunk[0] * inv_rms) * weight_chunk[0];
        output_chunk[1] = (input_chunk[1] * inv_rms) * weight_chunk[1];
        output_chunk[2] = (input_chunk[2] * inv_rms) * weight_chunk[2];
        output_chunk[3] = (input_chunk[3] * inv_rms) * weight_chunk[3];
        output_chunk[4] = (input_chunk[4] * inv_rms) * weight_chunk[4];
        output_chunk[5] = (input_chunk[5] * inv_rms) * weight_chunk[5];
        output_chunk[6] = (input_chunk[6] * inv_rms) * weight_chunk[6];
        output_chunk[7] = (input_chunk[7] * inv_rms) * weight_chunk[7];
    }

    for ((&input, &weight), output) in input_chunks
        .remainder()
        .iter()
        .zip(weight_chunks.remainder().iter())
        .zip(output_chunks.into_remainder().iter_mut())
    {
        *output = (input * inv_rms) * weight;
    }
}
