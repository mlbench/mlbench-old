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

.. http:get:: /api/metrics/(str:pod_name_or_run_id)/

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

   :query since: only get metrics newer than this date, (Default `1970-01-01T00:00:00.000000Z`)
   :query metric_type: one of `pod` or `run` to determine what kind of metric to get (Default: `pod`)

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 200: no error

.. http:post:: /api/metrics

   Save metrics. "pod_name" and "run_id" are mutually exclusive. The fields of metrics and their types are defined in `mlbench/api/models/kubemetrics.py`.

   **Example request**:

   .. sourcecode:: http

      POST /api/metrics HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript

      {
        "pod_name": "quiet-mink-mlbench-worker-1",
        "run_id": 2,
        "name": "accuracy",
        "date": "2018-08-03T09:21:44.331823Z",
        "value": "0.7845",
        "cumulative": False,
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
        "cumulative": False,
        "metadata": "some additional data"
      }

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 201: no error

Runs
----
.. http:get:: /api/runs/

   Gets all active/failed/finished runs

   **Example request**:

   .. sourcecode:: http

      GET /api/runs/ HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      [
        {
          "id": 1,
          "name": "Name of the run",
          "created_at": "2018-08-03T09:21:29.347960Z",
          "state": "STARTED",
          "job_id": "5ec9f286-e12d-41bc-886e-0174ef2bddae",
          "job_metadata": {...}
        },
        {
          "id": 2,
          "name": "Another run",
          "created_at": "2018-08-02T08:11:22.123456Z",
          "state": "FINISHED",
          "job_id": "add4de0f-9705-4618-93a1-00bbc8d9498e",
          "job_metadata": {...}
        },
      ]

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 200: no error

.. http:get:: /api/runs/(int:run_id)/

   Gets a run by id

   **Example request**:

   .. sourcecode:: http

      GET /api/runs/1/ HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
        "id": 1,
        "name": "Name of the run",
        "created_at": "2018-08-03T09:21:29.347960Z",
        "state": "STARTED",
        "job_id": "5ec9f286-e12d-41bc-886e-0174ef2bddae",
        "job_metadata": {...}
      }

   :run_id The id of the run

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 200: no error

.. http:post:: /api/runs/

   Starts a new Run

   **Example request**:

   .. sourcecode:: http

      POST /api/runs/ HTTP/1.1
      Host: example.com
      Accept: application/json, text/javascript
      
    

   :<json json body:       {
                            "name": "Name of the run",
                            "num_workers": 5,
                            "num_cpus": 4,
                            "max_bandwidth": 10000
                          }

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
        "id": 1,
        "name": "Name of the run",
        "created_at": "2018-08-03T09:21:29.347960Z",
        "state": "STARTED",
        "job_id": "5ec9f286-e12d-41bc-886e-0174ef2bddae",
        "job_metadata": {...}
      }

   :reqheader Accept: the response content type depends on
                      :mailheader:`Accept` header
   :resheader Content-Type: this depends on :mailheader:`Accept`
                            header of request
   :statuscode 200: no error
   :statuscode 409: a run is already active

