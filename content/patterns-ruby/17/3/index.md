---
canonical: "https://softwarepatternslexicon.com/patterns-ruby/17/3"

title: "Prometheus Monitoring for Ruby Applications: A Comprehensive Guide"
description: "Learn how to effectively monitor Ruby applications using Prometheus. This guide covers setting up Prometheus, exposing metrics, and integrating with visualization tools like Grafana."
linkTitle: "17.3 Monitoring Applications with Prometheus"
categories:
- Ruby
- Monitoring
- Observability
tags:
- Prometheus
- Ruby
- Monitoring
- Metrics
- Grafana
date: 2024-11-23
type: docs
nav_weight: 173000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 17.3 Monitoring Applications with Prometheus

In today's fast-paced software development environment, monitoring applications is crucial for ensuring performance, reliability, and user satisfaction. Prometheus, an open-source monitoring and alerting toolkit, has become a popular choice for developers due to its powerful features and flexibility. In this section, we'll explore how to use Prometheus to monitor Ruby applications, collect metrics, and set up alerts.

### Introduction to Prometheus

Prometheus is a robust system designed for monitoring and alerting. It was originally developed at SoundCloud and has since become a part of the Cloud Native Computing Foundation. Prometheus is known for its multidimensional data model, flexible query language (PromQL), and efficient time-series database.

**Key Features of Prometheus:**

- **Multidimensional Data Model**: Allows for the collection of metrics with labels, enabling powerful querying and aggregation.
- **PromQL**: A flexible query language for extracting and analyzing metrics.
- **Pull-Based Model**: Prometheus scrapes metrics from instrumented applications, ensuring that the monitoring system is in control.
- **Alerting**: Supports alerting based on metric thresholds and conditions.
- **Integration with Visualization Tools**: Easily integrates with tools like Grafana for creating dashboards.

### Exposing Metrics from Ruby Applications

To monitor a Ruby application with Prometheus, we need to expose application metrics in a format that Prometheus can scrape. The `prometheus-client-ruby` library is a popular choice for this purpose.

#### Setting Up `prometheus-client-ruby`

1. **Installation**: Add the `prometheus-client` gem to your Gemfile and run `bundle install`.

   ```ruby
   gem 'prometheus-client'
   ```

2. **Creating a Metrics Endpoint**: Create an endpoint in your application to expose metrics. This is typically done by adding a route that returns metrics in the Prometheus format.

   ```ruby
   require 'prometheus/client'

   # Create a new registry
   prometheus = Prometheus::Client.registry

   # Create a new counter metric
   http_requests_total = Prometheus::Client::Counter.new(:http_requests_total, docstring: 'A counter of HTTP requests made')
   prometheus.register(http_requests_total)

   # Increment the counter
   http_requests_total.increment

   # Expose metrics
   get '/metrics' do
     content_type 'text/plain'
     prometheus.text
   end
   ```

#### Instrumenting Code to Collect Metrics

Prometheus supports several types of metrics, including counters, gauges, and histograms. Let's explore how to use these in a Ruby application.

- **Counters**: Used to count occurrences of events. They only increase.

  ```ruby
  http_requests_total.increment(labels: { method: 'get', handler: '/home' })
  ```

- **Gauges**: Used to measure values that can go up and down, such as temperature or current memory usage.

  ```ruby
  memory_usage = Prometheus::Client::Gauge.new(:memory_usage, docstring: 'Current memory usage in bytes')
  prometheus.register(memory_usage)
  memory_usage.set(1024 * 1024 * 512) # Set to 512MB
  ```

- **Histograms**: Used to observe and bucket values, such as request durations.

  ```ruby
  request_duration = Prometheus::Client::Histogram.new(:request_duration_seconds, docstring: 'Request duration in seconds')
  prometheus.register(request_duration)
  request_duration.observe(0.5) # Observe a request that took 0.5 seconds
  ```

### Setting Up a Prometheus Server

Once your application is instrumented to expose metrics, you'll need to set up a Prometheus server to scrape these metrics.

1. **Download and Install Prometheus**: Download the latest version of Prometheus from the [official website](https://prometheus.io/download/).

2. **Configure Prometheus**: Create a configuration file (`prometheus.yml`) to define scrape targets.

   ```yaml
   global:
     scrape_interval: 15s # Set the scrape interval to 15 seconds

   scrape_configs:
     - job_name: 'ruby_app'
       static_configs:
         - targets: ['localhost:4567'] # Replace with your application's metrics endpoint
   ```

3. **Start Prometheus**: Run Prometheus with the configuration file.

   ```bash
   ./prometheus --config.file=prometheus.yml
   ```

### Creating Alerts Based on Metric Thresholds

Prometheus supports alerting based on metric thresholds. Alerts are defined in a separate configuration file and evaluated by the Prometheus server.

1. **Define Alert Rules**: Create an alert rule file (`alerts.yml`).

   ```yaml
   groups:
   - name: example
     rules:
     - alert: HighMemoryUsage
       expr: memory_usage > 1024 * 1024 * 1024 # 1GB
       for: 5m
       labels:
         severity: critical
       annotations:
         summary: "High memory usage detected"
         description: "Memory usage is above 1GB for more than 5 minutes."
   ```

2. **Configure Prometheus to Use Alert Rules**: Update `prometheus.yml` to include the alert rule file.

   ```yaml
   rule_files:
     - 'alerts.yml'
   ```

3. **Set Up Alertmanager**: Prometheus uses Alertmanager to handle alerts. Configure Alertmanager to send notifications via email, Slack, or other channels.

### Benefits of Proactive Monitoring and Observability

Proactive monitoring and observability provide several benefits:

- **Early Detection of Issues**: Identify and resolve issues before they impact users.
- **Performance Optimization**: Gain insights into application performance and optimize resource usage.
- **Improved Reliability**: Ensure application reliability by monitoring key metrics and setting up alerts.
- **Data-Driven Decision Making**: Use metrics to make informed decisions about application architecture and infrastructure.

### Integrating with Visualization Tools like Grafana

Grafana is a powerful visualization tool that integrates seamlessly with Prometheus. It allows you to create interactive dashboards and visualize metrics.

1. **Install Grafana**: Download and install Grafana from the [official website](https://grafana.com/get).

2. **Add Prometheus as a Data Source**: In Grafana, add Prometheus as a data source by providing the Prometheus server URL.

3. **Create Dashboards**: Use Grafana's dashboard editor to create visualizations of your metrics.

   ```mermaid
   graph TD;
       A[Prometheus] -->|Scrapes Metrics| B[Grafana];
       B -->|Visualizes Data| C[User];
   ```

   *Figure: Integration of Prometheus with Grafana for Visualization*

### Conclusion

Monitoring Ruby applications with Prometheus provides a powerful way to ensure performance and reliability. By exposing metrics, setting up a Prometheus server, and integrating with visualization tools like Grafana, you can gain valuable insights into your application's behavior. Remember, proactive monitoring is key to maintaining a robust and scalable application.

### Try It Yourself

Experiment with the code examples provided in this guide. Try adding new metrics, setting different thresholds for alerts, and creating custom dashboards in Grafana. The more you practice, the more proficient you'll become in monitoring applications with Prometheus.

## Quiz: Monitoring Applications with Prometheus

{{< quizdown >}}

### What is Prometheus primarily used for?

- [x] Monitoring and alerting
- [ ] Data storage
- [ ] Web development
- [ ] Machine learning

> **Explanation:** Prometheus is an open-source system primarily used for monitoring and alerting.

### Which Ruby library is commonly used to expose metrics for Prometheus?

- [x] prometheus-client-ruby
- [ ] sinatra
- [ ] rails
- [ ] devise

> **Explanation:** The `prometheus-client-ruby` library is used to expose metrics in Ruby applications for Prometheus.

### What type of metric is used to count occurrences of events in Prometheus?

- [x] Counter
- [ ] Gauge
- [ ] Histogram
- [ ] Summary

> **Explanation:** Counters are used to count occurrences of events and only increase.

### How does Prometheus collect metrics from applications?

- [x] By scraping metrics from endpoints
- [ ] By pushing metrics to a central server
- [ ] By reading log files
- [ ] By querying databases

> **Explanation:** Prometheus uses a pull-based model to scrape metrics from instrumented applications.

### What is the purpose of Grafana in the context of Prometheus?

- [x] Visualization of metrics
- [ ] Data storage
- [ ] Alerting
- [ ] Application development

> **Explanation:** Grafana is used to visualize metrics collected by Prometheus.

### Which of the following is a benefit of proactive monitoring?

- [x] Early detection of issues
- [ ] Increased code complexity
- [ ] Reduced application performance
- [ ] Higher development costs

> **Explanation:** Proactive monitoring helps in the early detection of issues, improving application reliability.

### What is the role of Alertmanager in Prometheus?

- [x] Handling alerts
- [ ] Storing metrics
- [ ] Visualizing data
- [ ] Developing applications

> **Explanation:** Alertmanager is used to handle alerts generated by Prometheus based on metric thresholds.

### What type of metric is used to measure values that can go up and down?

- [x] Gauge
- [ ] Counter
- [ ] Histogram
- [ ] Summary

> **Explanation:** Gauges are used to measure values that can increase or decrease, such as temperature or memory usage.

### How often does Prometheus scrape metrics by default?

- [x] Every 15 seconds
- [ ] Every 5 seconds
- [ ] Every 30 seconds
- [ ] Every minute

> **Explanation:** The default scrape interval in Prometheus is 15 seconds.

### True or False: Prometheus uses a push-based model to collect metrics.

- [ ] True
- [x] False

> **Explanation:** Prometheus uses a pull-based model to scrape metrics from applications.

{{< /quizdown >}}

Remember, this is just the beginning. As you progress, you'll build more complex monitoring setups and gain deeper insights into your applications. Keep experimenting, stay curious, and enjoy the journey!
