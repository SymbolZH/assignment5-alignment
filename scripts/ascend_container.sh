#!/usr/bin/env bash
# Build and manage the isolated Ascend development environment.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=${ASSIGNMENT5_CONTAINER:-assignment5-ascend-dev}
image=${ASSIGNMENT5_IMAGE:-assignment5-ascend:vllm0.23.0}
action=${1:-shell}
if [[ $# -gt 0 ]]; then shift; fi

case "$action" in
    build)
        proxy_args=()
        for proxy_name in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
            if [[ -v $proxy_name ]]; then
                proxy_args+=(--build-arg "$proxy_name")
            fi
        done
        docker build "${proxy_args[@]}" \
            --build-arg "DEV_UID=$(id -u)" --build-arg "DEV_GID=$(id -g)" \
            -t "$image" "$repo_root/docker/ascend"
        ;;
    start)
        if docker container inspect "$container" >/dev/null 2>&1; then
            expected_image=$(docker image inspect --format '{{.Id}}' "$image")
            actual_image=$(docker container inspect --format '{{.Image}}' "$container")
            if [[ $actual_image != "$expected_image" ]]; then
                echo "$container uses an older image; use '$0 recreate' to replace it." >&2
                exit 1
            fi
            docker start "$container"
        else
            proxy_args=()
            for proxy_name in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
                if [[ -v $proxy_name ]]; then
                    proxy_args+=(-e "$proxy_name")
                fi
            done
            docker run -d --name "$container" --runtime=ascend \
                "${proxy_args[@]}" \
                -e "NO_PROXY=${NO_PROXY:-},localhost,127.0.0.1,::1" \
                -e "no_proxy=${no_proxy:-},localhost,127.0.0.1,::1" \
                --network=host --ipc=host \
                --ulimit memlock=-1 --ulimit stack=67108864 \
                -e "ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
                --mount "type=bind,src=$repo_root,dst=/workspace/assignment5-alignment" \
                --mount type=volume,src=assignment5-ascend-cache,dst=/home/zhanghao/.cache \
                "$image"
        fi
        ;;
    recreate)
        # The project bind mount and named cache volume survive replacement.
        if docker container inspect "$container" >/dev/null 2>&1; then
            docker stop "$container"
            docker rm "$container"
        fi
        exec bash "$0" start
        ;;
    shell)
        exec docker exec -it -w /workspace/assignment5-alignment "$container" bash
        ;;
    exec)
        exec docker exec -w /workspace/assignment5-alignment "$container" "$@"
        ;;
    status)
        docker ps -a --filter "name=^/${container}$" \
            --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
        ;;
    stop)
        docker stop "$container"
        ;;
    *)
        echo "Usage: $0 {build|start|recreate|shell|exec COMMAND...|status|stop}" >&2
        exit 2
        ;;
esac
