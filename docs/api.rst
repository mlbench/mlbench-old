========
REST API
========

MLBench provides a basic REST Api though which most functionality can also be used. It's accessible through the /api/ endpoints on the dashboard URL.

.. http:get:: /api/nodes/

   All Worker-Nodes available in the cluster, including status information

   **Example request**:

   .. sourcecode:: http

      GET /api/nodes HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      [
        {
          "name":"worn-mouse-mlbench-worker-55bbdd4d8c-4mxh5",
          "labels":"{'app': 'mlbench', 'component': 'worker', 'pod-template-hash': '1166880847', 'release': 'worn-mouse'}",
          "phase":"Running",
          "ip":"10.244.2.58"
        },
        {
          "name":"worn-mouse-mlbench-worker-55bbdd4d8c-bwwsp",
          "labels":"{'app': 'mlbench', 'component': 'worker', 'pod-template-hash': '1166880847', 'release': 'worn-mouse'}",
          "phase":"Running",
          "ip":"10.244.3.57"
        }
      ]

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 200: no error


.. http:post:: /api/mpi_jobs/

   Starts an MPI Job

   **Example request**:

   .. sourcecode:: http

      POST /api/mpi_jobs HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
        "nodes":[
          ["10.244.2.58","default","worn-mouse-mlbench-worker-55bbdd4d8c-4mxh5","{'app': 'mlbench', 'component': 'worker', 'pod-template-hash': '1166880847', 'release': 'worn-mouse'}"],
          ["10.244.3.57","default","worn-mouse-mlbench-worker-55bbdd4d8c-bwwsp","{'app': 'mlbench', 'component': 'worker', 'pod-template-hash': '1166880847', 'release': 'worn-mouse'}"]
        ],
        "command":"['sh', '/usr/bin/mpirun', '--host', '10.244.2.58,10.244.3.57', '/usr/local/bin/python', '/app/main.py']",
        "master_name":"worn-mouse-mlbench-worker-55bbdd4d8c-4mxh5",
        "response":"Warning: Permanently added '10.244.3.57' (RSA) to the list of known hosts.\r\nFinished\nFinished\n"
      }

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 200: no error