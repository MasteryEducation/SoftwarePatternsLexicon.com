---
canonical: "https://softwarepatternslexicon.com/patterns-elixir/12/14"
title: "Configuration Management in Elixir Microservices"
description: "Explore advanced configuration management techniques in Elixir microservices, including externalized configurations, configuration servers, and environment-specific settings."
linkTitle: "12.14. Configuration Management"
categories:
- Elixir
- Microservices
- Configuration Management
tags:
- Elixir
- Microservices
- Configuration Management
- Consul
- Vault
date: 2024-11-23
type: docs
nav_weight: 134000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 12.14. Configuration Management

In the realm of microservices, configuration management is a critical aspect that ensures your applications are flexible, scalable, and maintainable. As systems grow in complexity, managing configurations effectively becomes paramount. In this section, we will delve into the intricacies of configuration management in Elixir microservices, focusing on externalized configurations, configuration servers, and environment-specific settings.

### Externalized Configurations

Externalizing configurations means storing configuration data outside the application's codebase. This approach offers several advantages, such as separating configuration from code, enabling dynamic updates without redeploying applications, and enhancing security by keeping sensitive information out of the codebase.

#### Benefits of Externalized Configurations

- **Separation of Concerns**: By keeping configuration separate from code, you can manage changes to configuration without affecting the application logic.
- **Dynamic Updates**: Externalized configurations allow you to update settings without redeploying the application, reducing downtime.
- **Security**: Sensitive information, such as API keys and database credentials, can be stored securely outside the codebase.
- **Environment Flexibility**: Easily switch configurations between different environments (development, staging, production) without changing the code.

#### Implementing Externalized Configurations in Elixir

In Elixir, externalized configurations can be managed using environment variables, configuration files, or dedicated configuration management tools. Let's explore these options:

##### Using Environment Variables

Environment variables are a simple and effective way to manage configurations. They are accessible within your Elixir application using the `System.get_env/1` function.

```elixir
# Fetching an environment variable
db_url = System.get_env("DATABASE_URL")

# Using the environment variable in your application
{:ok, _} = Ecto.Adapters.SQL.Sandbox.start_link(url: db_url)
```

##### Configuration Files

Elixir applications often use configuration files, such as `config/config.exs`, to manage settings. These files can be dynamically loaded based on the environment.

```elixir
# config/config.exs
use Mix.Config

config :my_app, MyApp.Repo,
  username: System.get_env("DB_USERNAME"),
  password: System.get_env("DB_PASSWORD"),
  database: System.get_env("DB_NAME"),
  hostname: System.get_env("DB_HOST"),
  pool_size: 10
```

##### Using Libraries for Configuration Management

Libraries like `Confex` and `Dotenv` can help manage configurations more effectively by loading environment variables from files and providing additional features.

```elixir
# Using Confex to fetch configuration
config :my_app, MyApp.Repo,
  username: {:system, "DB_USERNAME"},
  password: {:system, "DB_PASSWORD"},
  database: {:system, "DB_NAME"},
  hostname: {:system, "DB_HOST"},
  pool_size: 10
```

### Configuration Servers

Configuration servers provide a centralized way to manage configurations across multiple services. They offer features like versioning, access control, and dynamic updates. Popular tools include Consul and Vault.

#### Consul

Consul is a service mesh solution that provides service discovery, configuration management, and health checking. It allows you to store and retrieve configuration data using a key-value store.

##### Setting Up Consul

1. **Install Consul**: Follow the [Consul installation guide](https://www.consul.io/docs/install) to set up Consul on your system.
2. **Start Consul Agent**: Run the Consul agent in development mode.

```bash
consul agent -dev
```

3. **Store Configuration**: Use the Consul CLI to store configuration data.

```bash
consul kv put my_app/config/db_url "postgres://user:pass@localhost/db"
```

4. **Fetch Configuration in Elixir**: Use the `ConsulEx` library to fetch configuration data.

```elixir
# Fetching configuration from Consul
{:ok, db_url} = ConsulEx.KV.get("my_app/config/db_url")
```

#### Vault

Vault is a tool for securely accessing secrets. It provides a unified interface to any secret, while providing tight access control and recording a detailed audit log.

##### Setting Up Vault

1. **Install Vault**: Follow the [Vault installation guide](https://www.vaultproject.io/docs/install) to set up Vault on your system.
2. **Start Vault Server**: Run the Vault server in development mode.

```bash
vault server -dev
```

3. **Store Secrets**: Use the Vault CLI to store secrets.

```bash
vault kv put secret/my_app/db_url value="postgres://user:pass@localhost/db"
```

4. **Fetch Secrets in Elixir**: Use the `Vault` library to fetch secrets.

```elixir
# Fetching secrets from Vault
{:ok, db_url} = Vault.read("secret/my_app/db_url")
```

### Environment-Specific Settings

Handling different configurations for development, staging, and production environments is crucial for maintaining consistency and reliability across deployments.

#### Managing Environment-Specific Configurations

Elixir's configuration system allows you to define environment-specific settings using configuration files like `config/dev.exs`, `config/test.exs`, and `config/prod.exs`.

```elixir
# config/dev.exs
use Mix.Config

config :my_app, MyApp.Repo,
  database: "my_app_dev",
  pool_size: 10

# config/prod.exs
use Mix.Config

config :my_app, MyApp.Repo,
  database: "my_app_prod",
  pool_size: 20
```

#### Using Mix Environment

Elixir's build tool, Mix, provides an environment variable `MIX_ENV` to specify the current environment. This variable can be used to load the appropriate configuration file.

```bash
# Set the environment to production
export MIX_ENV=prod

# Run the application
mix run --no-halt
```

### Visualizing Configuration Management

To better understand the flow of configuration management in Elixir microservices, let's visualize the process using a Mermaid.js diagram.

```mermaid
graph TD;
    A[Start Application] --> B{Load Configuration};
    B --> |Environment Variables| C[System.get_env/1];
    B --> |Configuration Files| D[config/config.exs];
    B --> |Configuration Server| E[Consul/Vault];
    C --> F[Application Running];
    D --> F;
    E --> F;
```

**Diagram Description**: This diagram illustrates the process of loading configurations in an Elixir application. The application can load configurations from environment variables, configuration files, or configuration servers like Consul and Vault.

### Best Practices for Configuration Management

- **Keep Configurations Secure**: Use tools like Vault to manage sensitive information securely.
- **Automate Configuration Updates**: Use CI/CD pipelines to automate the deployment of configuration changes.
- **Version Control Configurations**: Store configuration files in version control to track changes and maintain history.
- **Use Consistent Naming Conventions**: Adopt a consistent naming convention for configuration keys to avoid confusion.
- **Test Configurations**: Regularly test configurations in different environments to ensure they work as expected.

### Try It Yourself

Experiment with the code examples provided in this section. Try modifying the configurations and observe how the application behavior changes. For instance, change the database URL in Consul or Vault and see how the application adapts without redeployment.

### Knowledge Check

- What are the benefits of externalized configurations?
- How can you use Consul to manage configurations in Elixir?
- What is the role of the `MIX_ENV` variable in Elixir?

### Embrace the Journey

Remember, mastering configuration management is a journey. As you progress, you'll find more efficient ways to manage configurations in your Elixir applications. Keep experimenting, stay curious, and enjoy the journey!

## Quiz: Configuration Management

{{< quizdown >}}

### What is a primary benefit of externalized configurations?

- [x] Separation of configuration from code
- [ ] Increased application size
- [ ] Slower application performance
- [ ] Harder to manage changes

> **Explanation:** Externalized configurations separate configuration from code, allowing for easier management and updates.

### Which tool is used for securely accessing secrets in configuration management?

- [ ] Consul
- [x] Vault
- [ ] Docker
- [ ] Kubernetes

> **Explanation:** Vault is a tool specifically designed for securely accessing secrets.

### How can environment-specific configurations be managed in Elixir?

- [x] Using environment-specific configuration files like `config/dev.exs`
- [ ] Hardcoding values in the application
- [ ] Using only environment variables
- [ ] Ignoring different environments

> **Explanation:** Elixir allows managing environment-specific configurations using files like `config/dev.exs`.

### What is the purpose of the `MIX_ENV` variable?

- [x] To specify the current environment for the application
- [ ] To increase application speed
- [ ] To decrease application size
- [ ] To manage user authentication

> **Explanation:** `MIX_ENV` is used to specify the current environment, affecting which configuration files are loaded.

### Which library can be used to fetch configuration data from Consul in Elixir?

- [x] ConsulEx
- [ ] Ecto
- [ ] Phoenix
- [ ] Plug

> **Explanation:** ConsulEx is a library used to interact with Consul for fetching configuration data.

### What is a key advantage of using configuration servers like Consul?

- [x] Centralized management of configurations
- [ ] Increased application complexity
- [ ] Slower configuration updates
- [ ] Harder to secure configurations

> **Explanation:** Configuration servers like Consul provide centralized management, making it easier to handle configurations across services.

### How can you fetch an environment variable in Elixir?

- [x] Using `System.get_env/1`
- [ ] Using `File.read/1`
- [ ] Using `Enum.map/2`
- [ ] Using `String.split/1`

> **Explanation:** `System.get_env/1` is used to fetch environment variables in Elixir.

### What is a common practice for storing sensitive information in configurations?

- [x] Using tools like Vault
- [ ] Hardcoding in the application
- [ ] Storing in plain text files
- [ ] Ignoring security concerns

> **Explanation:** Tools like Vault are used to securely store sensitive information in configurations.

### What is the role of configuration files like `config/config.exs` in Elixir?

- [x] To define application settings and configurations
- [ ] To store user data
- [ ] To manage database connections directly
- [ ] To handle HTTP requests

> **Explanation:** Configuration files like `config/config.exs` are used to define application settings and configurations.

### True or False: Configuration management is only important for large applications.

- [ ] True
- [x] False

> **Explanation:** Configuration management is important for applications of all sizes to ensure flexibility, security, and maintainability.

{{< /quizdown >}}
