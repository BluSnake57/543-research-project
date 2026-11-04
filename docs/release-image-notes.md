This release includes a CUDA-enabled Sentinel-X desktop image for Linux x86-64.
Use the repository's `compose.yaml` for noVNC, authenticated conversation downloads,
NVIDIA GPU passthrough, persistent storage, and configurable resource limits.

Pull the tagged image from GitHub Container Registry using the reference in
`IMAGE.txt`, or download **all** `sentinel-x-image.tar.gz.part-*` files and
`SHA256SUMS`/`IMAGE.txt` into one directory, then run:

```sh
sha256sum -c SHA256SUMS
cat sentinel-x-image.tar.gz.part-* | gzip -dc | docker image load
```

The archive is split because GitHub limits individual release assets to less than
2 GiB. Set `SENTINEL_IMAGE` in the repository's `.env` to the reference in `IMAGE.txt`,
run `python3 scripts/create_container_secret.py`, and start the loaded image with
`docker compose up --no-build --wait -d`. See `docs/container.md` for NVIDIA host
requirements and model setup. Model weights downloaded after startup and user
conversations are never included in the published image.
