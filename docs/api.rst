========
REST API
========

MLBench provides a basic REST Api though which most functionality can also be used. It's accessible through the /api/ endpoints on the dashboard URL.

Pods
----

.. http:get:: /api/pods/

   All Worker-Pods available in the cluster, including status information

   **Example request**:

   .. sourcecode:: http

      GET /api/pods HTTP/1.1
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

Metrics
-------

.. http:get:: /api/metrics/

   Get metrics (Cpu, Memory etc.) for all Worker Pods

   **Example request**:

   .. sourcecode:: http

      GET /api/metrics HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
        "quiet-mink-mlbench-worker-0": {
            "container_cpu_usage_seconds_total": [
                {
                    "date": "2018-08-03T09:21:38.594282Z",
                    "value": "0.188236813"
                },
                {
                    "date": "2018-08-03T09:21:50.244277Z",
                    "value": "0.215950298"
                }
            ]
        },
        "quiet-mink-mlbench-worker-1": {
            "container_cpu_usage_seconds_total": [
                {
                    "date": "2018-08-03T09:21:29.347960Z",
                    "value": "0.149286015"
                },
                {
                    "date": "2018-08-03T09:21:44.266181Z",
                    "value": "0.15325329"
                }
            ],
            "container_cpu_user_seconds_total": [
                {
                    "date": "2018-08-03T09:21:29.406238Z",
                    "value": "0.1"
                },
                {
                    "date": "2018-08-03T09:21:44.331823Z",
                    "value": "0.1"
                }
            ]
        }
    }

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 200: no error

.. http:get:: /api/metrics/(str:pod_name)/

   Get metrics (Cpu, Memory etc.) for all Worker Pods

   **Example request**:

   .. sourcecode:: http

      GET /api/metrics HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
        "container_cpu_usage_seconds_total": [
            {
                "date": "2018-08-03T09:21:29.347960Z",
                "value": "0.149286015"
            },
            {
                "date": "2018-08-03T09:21:44.266181Z",
                "value": "0.15325329"
            }
        ],
        "container_cpu_user_seconds_total": [
            {
                "date": "2018-08-03T09:21:29.406238Z",
                "value": "0.1"
            },
            {
                "date": "2018-08-03T09:21:44.331823Z",
                "value": "0.1"
            }
        ]
      }

   :query since: only get metrics newer than this date, default 1970-01-01T00:00:00.000000Z

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 200: no error

.. http:post:: /api/metrics

   Save metrics

   **Example request**:

   .. sourcecode:: http

      POST /api/metrics HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript

      {
        "pod_name": "quiet-mink-mlbench-worker-1",
        "name": "accuracy",
        "date": "2018-08-03T09:21:44.331823Z",
        "value": "0.7845",
        "metadata": "some additional data"
      }

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 201 CREATED
      Vary: Accept
      Content-Type: text/javascript

      {
        "pod_name": "quiet-mink-mlbench-worker-1",
        "name": "accuracy",
        "date": "2018-08-03T09:21:44.331823Z",
        "value": "0.7845",
        "metadata": "some additional data"
      }

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 201: no error

MPI Jobs
--------
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
        "pods":[
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