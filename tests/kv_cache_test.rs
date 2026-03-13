use xrt_core::KvCache;
use xrt_runtime::PagedKvCache;

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
