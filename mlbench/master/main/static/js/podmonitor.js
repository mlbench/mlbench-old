var PodMonitor = function(parent_id, metric_selector, target_element, metric_type){
    this.node_data = {'last_metrics_update': new Date(0)};
    this.nodeRefreshInterval = 1 * 1000;
    this.metricsRefreshInterval = 20 * 1000;
    this.renderInterval = 1 * 1000;
    this.parent_id = parent_id;
    this.target_element = target_element;
    this.metric_selector = metric_selector;
    this.metric_type = metric_type;

    // this.updateNodes = function(){
    //     var nodes = this.nodes;
    //     $.getJSON("/api/pods", function(data){
    //         $.each(data, function(index, value){
    //             if(!(value['name'] in nodes)){
    //                 nodes[value['name']] = {};
    //                 nodes[value['name']]['last_metrics_update'] = new Date(0);
    //             }

    //             nodes[value['name']]['node_info'] = value;
    //         });
    //     });
    // }

    this.updateMetrics = function(){
        var parent_id = this.parent_id;
        var value = this.node_data;
        var metric_type = this.metric_type;

        $.getJSON("/api/metrics/" + parent_id + "/",
            {since: value['last_metrics_update'].toJSON(),
            metric_type: metric_type},
            function(data){
                if(!('node_metrics' in value)){
                    value['node_metrics'] = [];
                }

                $.each(data, function(key, values){
                    if(!(key in value['node_metrics'])){
                        value['node_metrics'][key] = [];
                    }
                    value['node_metrics'][key] = value['node_metrics'][key].concat(values);
                    value['last_metrics_update'] = new Date();
                });
            });
    }

    this.plotMetrics = function(element, metrics, value, title){
        var self = this;

        if(!(metrics['node_metrics']) || metrics['node_metrics'][value].length == 0){
            return;
        }

        var el = $(element);
        d3.select(el[0]).selectAll("*").remove();

        var cumulative = metrics['node_metrics'][value][0]['cumulative'];

        if(cumulative){
            var transform = function(cur, prev){return 1000 * (cur - prev) / self.metricsRefreshInterval;};
        }else{
            var transform = function(cur, prev){return cur;};
        }

        prev = metrics['node_metrics'][value][0]['value'];
        data = [];

        len = metrics['node_metrics'][value].length;

        var max = 0;

        var parseTime = d3.timeParse("%Y-%m-%dT%H:%M:%S.%fZ");

        for(var i = 0; i < len; i++){
            var cur = metrics['node_metrics'][value][i];
            cur_val = transform(cur['value'], prev);

            if(cur_val > max){
                max = cur_val;
            }

            data.push({x: parseTime(cur['date']), y: cur_val});
            prev = cur['value'];
        }

        if(data.length == 1){
            var newData = {x: data[0].x, y: data[0].y};
            data[0].x = new Date(newData.x.getTime() - 1000)
            data.push(newData);
        }
        var svg = d3.select(el[0]),
            margin = {top: 10, right: 10, bottom: 30, left: 40},
            width = el.width() - margin.left - margin.right,
            height = el.height() - margin.top - margin.bottom,
            g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var x = d3.scaleTime().range([0, width]);
        var y = d3.scaleLinear().range([height, 0]);

        // define the line
        var line = d3.line()
            .x(function(d) { return x(d.x); })
            .y(function(d) { return y(d.y); })
            .curve(d3.curveMonotoneX);

        x.domain(d3.extent(data, function(d) { return d.x; }));
        y.domain([0, Math.max(1, d3.max(data, function(d) { return d.y; }))]);

        g.append("path")
            .data([data])
            .attr("class", "line")
            .attr("d", line);

        // Add the X Axis
        g.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x));

        // Add the Y Axis
        g.append("g")
            .call(d3.axisLeft(y));

        g.append("text")
            .attr("x", (width / 2))
            .attr("y", margin.top)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text(title);
    }

    this.renderNodes = function(){
        var self = this;

        value = self.node_data;
        name = self.metric_selector();

        if(('node_metrics' in value)){
            self.plotMetrics(self.target_element, value, name, name);
        }

        feather.replace();
    }

    //this.updateNodes();
    setTimeout(this.updateMetrics, 100);

    //setInterval(this.updateNodes, this.nodeRefreshInterval);
    setInterval(this.updateMetrics, this.metricsRefreshInterval);
    setInterval(this.renderNodes, this.renderInterval);
}