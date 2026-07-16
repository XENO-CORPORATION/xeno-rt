use xrt_core::KvCache;
use xrt_runtime::{KvCacheMode, PagedKvCache, SessionKvCache};

fn assert_slice_close(lhs: &[f32], rhs: &[f32], tolerance: f32) {
    assert_eq!(lhs.len(), rhs.len());
    for (index, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() <= tolerance,
            "index {index}: left={lhs}, right={rhs}, tolerance={tolerance}"
        );
    }
}

#[test]
fn allocates_and_deallocates_pages() {
    let mut cache = PagedKvCache::new(1, 4, 2);

    cache
        .append(0, &[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0])
        .expect("first append should allocate");
    cache
        .append(0, &[5.0, 6.0, 7.0, 8.0], &[8.0, 7.0, 6.0, 5.0])
        .expect("second append should fill the first page");
    cache
        .append(0, &[9.0, 10.0, 11.0, 12.0], &[12.0, 11.0, 10.0, 9.0])
        .expect("third append should allocate a second page");

    assert_eq!(cache.len(0), 3);
    assert_eq!(cache.key(0, 2), Some(&[9.0, 10.0, 11.0, 12.0][..]));
    assert_eq!(cache.value(0, 2), Some(&[12.0, 11.0, 10.0, 9.0][..]));

    cache.clear();

    assert_eq!(cache.len(0), 0);
    assert_eq!(cache.key(0, 0), None);
    assert_eq!(cache.value(0, 0), None);

    cache
        .append(0, &[13.0, 14.0, 15.0, 16.0], &[16.0, 15.0, 14.0, 13.0])
        .expect("cache should allocate again after clear");
    assert_eq!(cache.len(0), 1);
    assert_eq!(cache.key(0, 0), Some(&[13.0, 14.0, 15.0, 16.0][..]));
}

#[test]
fn writes_and_reads_back_kv_pairs() {
    let mut cache = PagedKvCache::new(2, 3, 2);

    cache
        .append(0, &[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0])
        .expect("layer 0 append should succeed");
    cache
        .append(1, &[4.0, 5.0, 6.0], &[6.0, 5.0, 4.0])
        .expect("layer 1 append should succeed");
    cache
        .append(0, &[7.0, 8.0, 9.0], &[9.0, 8.0, 7.0])
        .expect("second layer 0 append should succeed");

    assert_eq!(cache.layers(), 2);
    assert_eq!(cache.width(), 3);
    assert_eq!(cache.len(0), 2);
    assert_eq!(cache.len(1), 1);
    assert_eq!(cache.key(0, 0), Some(&[1.0, 2.0, 3.0][..]));
    assert_eq!(cache.value(0, 1), Some(&[9.0, 8.0, 7.0][..]));
    assert_eq!(cache.key(1, 0), Some(&[4.0, 5.0, 6.0][..]));
    assert_eq!(cache.value(1, 0), Some(&[6.0, 5.0, 4.0][..]));
}

#[test]
fn grows_across_multiple_pages() {
    let mut cache = PagedKvCache::new(1, 2, 2);

    for index in 0..5 {
        let base = index as f32;
        cache
            .append(0, &[base, base + 0.5], &[base + 1.0, base + 1.5])
            .expect("append should succeed");
    }

    assert_eq!(cache.len(0), 5);
    for index in 0..5 {
        let base = index as f32;
        assert_eq!(cache.key(0, index), Some(&[base, base + 0.5][..]));
        assert_eq!(cache.value(0, index), Some(&[base + 1.0, base + 1.5][..]));
    }
    assert_eq!(cache.key(0, 5), None);
    assert_eq!(cache.value(0, 5), None);
}

#[test]
fn key_q4_value_q8_mode_roundtrips_and_truncates() {
    let width = 70;
    let mut cache = SessionKvCache::new(KvCacheMode::KeyQ4ValueQ8, 1, width, 2);

    let key0 = (0..width)
        .map(|index| (index as f32 - 35.0) / 8.0)
        .collect::<Vec<_>>();
    let val0 = (0..width)
        .map(|index| (index as f32 - 20.0) / 3.0)
        .collect::<Vec<_>>();
    let key1 = key0.iter().map(|value| value * -0.75).collect::<Vec<_>>();
    let val1 = val0.iter().map(|value| value * 0.5).collect::<Vec<_>>();
    let key2 = key0.iter().map(|value| value + 0.25).collect::<Vec<_>>();
    let val2 = val0.iter().map(|value| value - 0.125).collect::<Vec<_>>();

    cache
        .append(0, &key0, &val0)
        .expect("first append should succeed");
    cache
        .append(0, &key1, &val1)
        .expect("second append should succeed");
    cache
        .append(0, &key2, &val2)
        .expect("third append should succeed");

    let mut key_buf = vec![0.0; width];
    let mut value_buf = vec![0.0; width];

    cache
        .copy_key_into(0, 0, &mut key_buf)
        .expect("first key should round-trip");
    cache
        .copy_value_into(0, 0, &mut value_buf)
        .expect("first value should round-trip");
    assert_slice_close(&key_buf, &key0, 0.6);
    assert_slice_close(&value_buf, &val0, 0.08);

    cache
        .copy_key_into(0, 2, &mut key_buf)
        .expect("third key should round-trip");
    cache
        .copy_value_into(0, 2, &mut value_buf)
        .expect("third value should round-trip");
    assert_slice_close(&key_buf, &key2, 0.6);
    assert_slice_close(&value_buf, &val2, 0.08);

    cache.truncate(2);
    assert_eq!(cache.len(0), 2);
    assert!(cache.copy_key_into(0, 2, &mut key_buf).is_err());
}

#[test]
fn kv_cache_mode_parses_key_first_aliases() {
    assert_eq!(
        KvCacheMode::parse("kq4_vq8"),
        Some(KvCacheMode::KeyQ4ValueQ8)
    );
    assert_eq!(
        KvCacheMode::parse("key_q4_value_q8"),
        Some(KvCacheMode::KeyQ4ValueQ8)
    );
    assert_eq!(KvCacheMode::parse("kq4"), Some(KvCacheMode::KeyQ4ValueQ8));
    assert_eq!(
        KvCacheMode::parse("agent"),
        Some(KvCacheMode::AgentAdaptive)
    );
}
