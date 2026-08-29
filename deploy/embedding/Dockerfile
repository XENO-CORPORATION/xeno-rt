FROM rust:1.85.1-bookworm@sha256:bf7d87666c4da6eace19e06d21bc4859c6e2a5c97a21ac273b0e082112753cf0 AS builder

WORKDIR /src
COPY . .

RUN cargo build --release --locked -p xrt-server -p xrt-cli
RUN ./target/release/xrt-cli bundle install \
      --manifest reference/embedding/nomic-embed-text-v1.5-a15734e.json \
      --cache-dir /bundle-cache \
    | tee /tmp/bundle-install.txt \
 && grep -F '2dc870de10066111e27bc6c25375d27f455e1de8a277b9bc5623f473ac9d2121' /tmp/bundle-install.txt

FROM debian:bookworm-slim@sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867

ARG ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
ARG ONNXRUNTIME_SHA256=aa70d48b22e264b82e83f63245b51ddc9a47ae4a3a66903efaff1ba68b7b5930
ARG ONNXRUNTIME_SIZE=6184676

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl --fail --location --proto '=https' --tlsv1.2 \
      --output /tmp/onnxruntime.tgz "$ONNXRUNTIME_URL" \
 && test "$(wc -c < /tmp/onnxruntime.tgz)" = "$ONNXRUNTIME_SIZE" \
 && echo "$ONNXRUNTIME_SHA256  /tmp/onnxruntime.tgz" | sha256sum --check --strict \
 && mkdir -p /opt/onnxruntime \
 && tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C /opt/onnxruntime \
 && test -s /opt/onnxruntime/lib/libonnxruntime.so.1.20.0 \
 && ln -s libonnxruntime.so.1.20.0 /opt/onnxruntime/lib/libonnxruntime.so \
 && rm /tmp/onnxruntime.tgz

RUN groupadd --system --gid 10001 xrt \
 && useradd --system --uid 10001 --gid xrt --home-dir /nonexistent --shell /usr/sbin/nologin xrt

COPY --from=builder /src/target/release/xrt-server /usr/local/bin/xrt-server
COPY --from=builder --chown=10001:10001 /bundle-cache /var/lib/xrt/models
COPY reference/embedding/nomic-embed-text-v1.5-a15734e.json /usr/share/xrt/embedding-manifest.json
COPY reference/runtime/onnxruntime-1.20.0-linux-x64.json /usr/share/xrt/onnxruntime-manifest.json
COPY LICENSE NOTICE /usr/share/doc/xeno-rt/

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so.1.20.0
ENV XRT_EMBEDDING_MODEL_DIR=/var/lib/xrt/models/bundles/nomic-embed-text-v1.5-a15734e/2dc870de10066111e27bc6c25375d27f455e1de8a277b9bc5623f473ac9d2121
ENV XRT_EMBEDDING_MAX_CONCURRENT_REQUESTS=4

USER 10001:10001
EXPOSE 3099
ENTRYPOINT ["/usr/local/bin/xrt-server"]
CMD ["--host", "0.0.0.0", "--port", "3099"]
