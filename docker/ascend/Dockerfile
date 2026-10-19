# Official ARM64 vLLM Ascend 0.23.0 release, with native HCCL weight transfer.
FROM quay.io/ascend/vllm-ascend@sha256:4ee78def8f33d59d48f116d1dfa793332c23c99ecab4f0d7dd5cd62d0fb4e6c1

ARG DEV_UID=1014
ARG DEV_GID=1002
USER root
COPY constraints.txt requirements.txt /opt/assignment5/
COPY check_dependencies.py /opt/assignment5/check_dependencies.py
RUN python3 -m pip install --no-cache-dir -c /opt/assignment5/constraints.txt \
        -r /opt/assignment5/requirements.txt && \
    python3 /opt/assignment5/check_dependencies.py && \
    (getent group "${DEV_GID}" || groupadd --gid "${DEV_GID}" assignment5) && \
    useradd --uid "${DEV_UID}" --gid "${DEV_GID}" --create-home \
        --home-dir /home/zhanghao --shell /bin/bash zhanghao && \
    mkdir -p /home/zhanghao/.cache && \
    chown -R "${DEV_UID}:${DEV_GID}" /home/zhanghao
COPY entrypoint.sh /opt/assignment5/entrypoint.sh
# The official image runs as root; make its packaged operators readable by
# the development user as well. No framework or operator code is changed.
RUN chmod 755 /opt/assignment5/entrypoint.sh && \
    chmod -R a+rX /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom \
        /usr/local/Ascend/cann-9.1.0/opp/vendors
ENV HOME=/home/zhanghao \
    PATH=/home/zhanghao/.local/bin:${PATH} \
    OMP_NUM_THREADS=8 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_ASCEND_ENABLE_NZ=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
USER ${DEV_UID}:${DEV_GID}
WORKDIR /workspace/assignment5-alignment
ENTRYPOINT ["/opt/assignment5/entrypoint.sh"]
CMD ["sleep", "infinity"]
