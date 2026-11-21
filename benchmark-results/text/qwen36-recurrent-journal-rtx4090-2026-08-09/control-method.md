# Legacy-copy workload control

The control executable was built from the candidate source with one
benchmark-only operation added after each fused recurrent layer had published
its final state. For every rollback boundary except the penultimate boundary,
it copied the journal's convolution and recurrent slices into the layer's
otherwise-dead pending buffers:

```rust
for boundary in 0..rows.saturating_sub(2) {
    device.copy_f32_device_range(
        &state.verify_rebase_conv,
        boundary * conv_len,
        &mut state.pending_conv,
    )?;
    device.copy_f32_device_range(
        &state.verify_rebase_recurrent,
        boundary * recurrent_len,
        &mut state.pending_recurrent,
    )?;
}
```

This reconstructs the number and byte volume of the per-boundary D2D copies
removed by direct journaling. It runs after state publication, so it cannot
change accepted tokens, rollback selection, or the next layer's input. The
control executable SHA-256 is recorded in `environment.txt`; this code is not
part of the production source tree.
