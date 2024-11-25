---
canonical: "https://softwarepatternslexicon.com/patterns-elixir/20/11"

title: "Containerization and Orchestration with Docker and Kubernetes for Elixir"
description: "Explore advanced containerization and orchestration techniques using Docker and Kubernetes for Elixir applications. Learn to create, manage, and orchestrate container images with best practices for distributed systems."
linkTitle: "20.11. Containerization and Orchestration (Docker, Kubernetes)"
categories:
- Elixir
- Containerization
- Orchestration
tags:
- Docker
- Kubernetes
- Elixir
- Containerization
- Orchestration
date: 2024-11-23
type: docs
nav_weight: 211000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 20.11. Containerization and Orchestration (Docker, Kubernetes)

As Elixir developers and architects, embracing containerization and orchestration is crucial for building scalable, resilient, and portable applications. This section will guide you through the essentials of using Docker and Kubernetes with Elixir, providing insights into creating container images, managing distributed systems, and implementing best practices.

### Docker Basics

Docker is a platform that enables developers to package applications into containers—standardized executable components that combine application source code with the operating system libraries and dependencies required to run that code in any environment. Let's delve into the basics of Docker and its role in Elixir development.

#### Creating and Managing Container Images

Docker images are the blueprints for containers. They contain everything needed to run an application, including the code, runtime, libraries, and environment variables. Here's how you can create and manage Docker images for Elixir applications.

1. **Dockerfile Basics**

   A `Dockerfile` is a script that contains a series of instructions on how to build a Docker image. Here is a basic example for an Elixir application:

   ```dockerfile
   # Use the official Elixir image
   FROM elixir:1.14

   # Set the working directory
   WORKDIR /app

   # Copy the mix.exs and mix.lock files
   COPY mix.exs mix.lock ./

   # Install Hex and Rebar
   RUN mix local.hex --force && \
       mix local.rebar --force

   # Install dependencies
   RUN mix deps.get

   # Copy the rest of the application code
   COPY . .

   # Compile the application
   RUN mix compile

   # Set the command to run the application
   CMD ["mix", "phx.server"]
   ```

   **Key Points:**
   - **Base Image**: We start with an official Elixir image.
   - **Working Directory**: Set to `/app`, where the application code will reside.
   - **Dependency Management**: Install Hex and Rebar for dependency management.
   - **Build Process**: Copy source files and compile the application.

2. **Building the Image**

   To build the Docker image from the Dockerfile, use the following command:

   ```bash
   docker build -t my_elixir_app .
   ```

   This command creates a Docker image named `my_elixir_app`.

3. **Running a Container**

   Once the image is built, you can run a container using:

   ```bash
   docker run -p 4000:4000 my_elixir_app
   ```

   This command maps port 4000 on your host to port 4000 in the container, allowing you to access your Elixir application.

4. **Managing Containers**

   Use Docker commands to manage your containers:

   - **List Containers**: `docker ps -a`
   - **Stop a Container**: `docker stop <container_id>`
   - **Remove a Container**: `docker rm <container_id>`

### Orchestrating Containers

While Docker provides the ability to run applications in containers, Kubernetes offers a robust orchestration system to manage these containers at scale. Kubernetes automates deployment, scaling, and management of containerized applications.

#### Automating Deployment, Scaling, and Management with Kubernetes

Kubernetes, often abbreviated as K8s, is an open-source platform designed to automate deploying, scaling, and operating application containers. Here's how it can be used with Elixir applications.

1. **Kubernetes Architecture**

   Kubernetes consists of several key components:

   - **Master Node**: Manages the cluster and orchestrates workloads.
   - **Worker Nodes**: Execute the workloads.
   - **Pods**: The smallest deployable units in Kubernetes, encapsulating one or more containers.

   ```mermaid
   graph TD;
       A[Master Node] -->|Schedules| B[Worker Node 1];
       A -->|Schedules| C[Worker Node 2];
       B -->|Runs| D[Pod];
       C -->|Runs| E[Pod];
   ```

   *Diagram: Kubernetes Architecture*

2. **Deploying an Elixir Application**

   To deploy an Elixir application on Kubernetes, you need to define a deployment configuration. Here's an example of a Kubernetes deployment YAML file:

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: elixir-app
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: elixir-app
     template:
       metadata:
         labels:
           app: elixir-app
       spec:
         containers:
         - name: elixir-app
           image: my_elixir_app
           ports:
           - containerPort: 4000
   ```

   **Key Points:**
   - **Replicas**: Specifies the number of pod instances.
   - **Selector**: Matches labels to identify pods.
   - **Template**: Defines the pod configuration, including the container image and ports.

3. **Scaling Applications**

   Kubernetes makes scaling applications straightforward. You can scale the number of replicas using the `kubectl` command:

   ```bash
   kubectl scale deployment/elixir-app --replicas=5
   ```

   This command increases the number of running pods to 5.

4. **Managing Updates**

   Kubernetes supports rolling updates, allowing you to update applications with zero downtime. Use the following command to apply updates:

   ```bash
   kubectl apply -f deployment.yaml
   ```

5. **Monitoring and Logging**

   Kubernetes provides tools for monitoring and logging, such as Prometheus and Grafana, to ensure your applications are running smoothly.

### Elixir-Specific Considerations

When deploying Elixir applications in containers, especially in distributed systems, there are specific considerations to keep in mind.

#### Handling Distributed Erlang Clusters within Containers

Elixir runs on the Erlang VM (BEAM), which is designed for distributed systems. Here are some tips for handling distributed Erlang clusters within Docker and Kubernetes:

1. **Node Discovery**

   Use Kubernetes DNS for node discovery. Ensure that your Elixir nodes can discover each other using service names.

2. **Cookie Management**

   Erlang nodes use cookies for authentication. Ensure that all nodes in a cluster share the same cookie, which can be managed through Kubernetes secrets.

3. **Networking**

   Configure Kubernetes networking to allow inter-node communication. Use `Cluster` libraries like `libcluster` to automate node discovery and connection.

4. **Persistent Storage**

   Use Kubernetes Persistent Volumes (PVs) and Persistent Volume Claims (PVCs) to manage stateful data.

### Best Practices

Implementing best practices ensures that your containerized Elixir applications are robust, secure, and efficient.

#### Configuring Networking, Environment Variables, Persistent Storage

1. **Networking**

   - **Service Discovery**: Use Kubernetes services to expose your Elixir application.
   - **Load Balancing**: Configure load balancers to distribute traffic across pods.

2. **Environment Variables**

   - Use Kubernetes ConfigMaps and Secrets to manage configuration and sensitive data.
   - Inject environment variables into your containers to configure application settings.

3. **Persistent Storage**

   - Use Kubernetes Persistent Volumes for data that needs to persist beyond the lifecycle of a pod.
   - Ensure data integrity and availability by configuring appropriate storage classes.

4. **Security**

   - Implement role-based access control (RBAC) to manage permissions.
   - Regularly update images to include security patches.
   - Use network policies to restrict communication between pods.

5. **Monitoring and Logging**

   - Integrate tools like Prometheus for monitoring and Grafana for visualization.
   - Use ELK stack (Elasticsearch, Logstash, Kibana) for centralized logging.

### Try It Yourself

Experiment with the Docker and Kubernetes examples provided. Try modifying the Dockerfile to use a different base image or add additional dependencies. Deploy the modified image to Kubernetes and observe how the changes affect the application's behavior.

### Knowledge Check

- What are the key components of a Dockerfile?
- How does Kubernetes manage scaling of applications?
- What are some Elixir-specific considerations when using Docker and Kubernetes?
- How can you manage environment variables in Kubernetes?
- What tools can be used for monitoring and logging in Kubernetes?

### Embrace the Journey

Remember, mastering containerization and orchestration is a journey. As you continue to experiment and learn, you'll discover new ways to optimize and scale your Elixir applications. Stay curious, keep exploring, and enjoy the process!

## Quiz Time!

{{< quizdown >}}

### What is the purpose of a Dockerfile?

- [x] To define the steps to create a Docker image
- [ ] To manage Kubernetes deployments
- [ ] To configure network policies
- [ ] To scale applications

> **Explanation:** A Dockerfile contains instructions to build a Docker image, specifying the base image, dependencies, and commands to run.

### How does Kubernetes handle application scaling?

- [x] By adjusting the number of pod replicas
- [ ] By increasing the CPU and memory of containers
- [ ] By modifying the Dockerfile
- [ ] By changing the base image

> **Explanation:** Kubernetes scales applications by adjusting the number of pod replicas, ensuring that the application can handle varying loads.

### What is a key consideration for Elixir applications in containers?

- [x] Managing distributed Erlang clusters
- [ ] Using a specific base image
- [ ] Avoiding environment variables
- [ ] Disabling networking

> **Explanation:** Elixir applications often run in distributed Erlang clusters, requiring careful management of node discovery and communication.

### How can environment variables be managed in Kubernetes?

- [x] Using ConfigMaps and Secrets
- [ ] By hardcoding them in the application
- [ ] Through Dockerfiles
- [ ] By modifying the Kubernetes API

> **Explanation:** ConfigMaps and Secrets are used in Kubernetes to manage environment variables and sensitive data securely.

### Which tool is commonly used for monitoring in Kubernetes?

- [x] Prometheus
- [ ] Docker
- [ ] ELK stack
- [ ] Grafana

> **Explanation:** Prometheus is a popular tool for monitoring Kubernetes clusters, often used in conjunction with Grafana for visualization.

### What is the smallest deployable unit in Kubernetes?

- [x] Pod
- [ ] Container
- [ ] Node
- [ ] Service

> **Explanation:** A pod is the smallest deployable unit in Kubernetes, encapsulating one or more containers.

### Which command is used to build a Docker image?

- [x] docker build -t my_image .
- [ ] docker run my_image
- [ ] kubectl apply -f deployment.yaml
- [ ] docker ps -a

> **Explanation:** The `docker build -t my_image .` command is used to build a Docker image from a Dockerfile.

### What is the role of Kubernetes Secrets?

- [x] To manage sensitive data securely
- [ ] To define network policies
- [ ] To expose services
- [ ] To scale applications

> **Explanation:** Kubernetes Secrets are used to manage sensitive data like passwords and API keys securely.

### How can you perform a rolling update in Kubernetes?

- [x] By applying a new deployment configuration
- [ ] By restarting all pods
- [ ] By deleting the current deployment
- [ ] By scaling down to zero replicas

> **Explanation:** Applying a new deployment configuration initiates a rolling update, allowing updates with zero downtime.

### True or False: Kubernetes can manage both stateless and stateful applications.

- [x] True
- [ ] False

> **Explanation:** Kubernetes can manage both stateless and stateful applications, providing features like Persistent Volumes for stateful data.

{{< /quizdown >}}


