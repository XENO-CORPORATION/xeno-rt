pub fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    let mut max0 = f32::NEG_INFINITY;
    let mut max1 = f32::NEG_INFINITY;
    let mut max2 = f32::NEG_INFINITY;
    let mut max3 = f32::NEG_INFINITY;
    let mut max4 = f32::NEG_INFINITY;
    let mut max5 = f32::NEG_INFINITY;
    let mut max6 = f32::NEG_INFINITY;
    let mut max7 = f32::NEG_INFINITY;

    let mut max_chunks = values.chunks_exact(8);
    for chunk in max_chunks.by_ref() {
        max0 = max0.max(chunk[0]);
        max1 = max1.max(chunk[1]);
        max2 = max2.max(chunk[2]);
        max3 = max3.max(chunk[3]);
        max4 = max4.max(chunk[4]);
        max5 = max5.max(chunk[5]);
        max6 = max6.max(chunk[6]);
        max7 = max7.max(chunk[7]);
    }
    let mut max = max0
        .max(max1)
        .max(max2)
        .max(max3)
        .max(max4)
        .max(max5)
        .max(max6)
        .max(max7);
    for &value in max_chunks.remainder() {
        max = max.max(value);
    }

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    let mut value_chunks = values.chunks_exact_mut(8);
    for chunk in value_chunks.by_ref() {
        let value0 = (chunk[0] - max).exp();
        let value1 = (chunk[1] - max).exp();
        let value2 = (chunk[2] - max).exp();
        let value3 = (chunk[3] - max).exp();
        let value4 = (chunk[4] - max).exp();
        let value5 = (chunk[5] - max).exp();
        let value6 = (chunk[6] - max).exp();
        let value7 = (chunk[7] - max).exp();

        chunk[0] = value0;
        chunk[1] = value1;
        chunk[2] = value2;
        chunk[3] = value3;
        chunk[4] = value4;
        chunk[5] = value5;
        chunk[6] = value6;
        chunk[7] = value7;

        sum0 += value0;
        sum1 += value1;
        sum2 += value2;
        sum3 += value3;
        sum4 += value4;
        sum5 += value5;
        sum6 += value6;
        sum7 += value7;
    }

    let mut sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    for value in value_chunks.into_remainder() {
        *value = (*value - max).exp();
        sum += *value;
    }

    if sum == 0.0 {
        return;
    }

    let inv_sum = sum.recip();
    let mut value_chunks = values.chunks_exact_mut(8);
    for chunk in value_chunks.by_ref() {
        chunk[0] *= inv_sum;
        chunk[1] *= inv_sum;
        chunk[2] *= inv_sum;
        chunk[3] *= inv_sum;
        chunk[4] *= inv_sum;
        chunk[5] *= inv_sum;
        chunk[6] *= inv_sum;
        chunk[7] *= inv_sum;
    }
    for value in value_chunks.into_remainder() {
        *value *= inv_sum;
    }
}
