docker ps -a -q --filter=label=mirantis.kubeadm_dind_cluster | while read container_id; do
    docker exec ${container_id} /bin/bash -c "docker rm -fv registry-proxy || true"
    # run registry proxy: forward from localhost:5000 on each node to host:5000
    docker exec ${container_id} /bin/bash -c \
      "docker run --name registry-proxy -d -e LISTEN=':5000' -e TALK=\"\$(/sbin/ip route|awk '/default/ { print \$3 }'):5000\" -p 5000:5000 tecnativa/tcp-proxy"
  done
