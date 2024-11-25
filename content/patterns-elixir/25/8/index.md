---
canonical: "https://softwarepatternslexicon.com/patterns-elixir/25/8"
title: "Configuration Management in Elixir: Dynamic Configurations, Secrets Management, and Tool Integration"
description: "Master configuration management in Elixir with dynamic configurations, secrets management, and integration with tools like Consul and Vault. Enhance your Elixir applications' reliability and security."
linkTitle: "25.8. Configuration Management"
categories:
- DevOps
- Infrastructure Automation
- Elixir
tags:
- Configuration Management
- Elixir
- Secrets Management
- Consul
- Vault
date: 2024-11-23
type: docs
nav_weight: 258000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 25.8. Configuration Management

In the realm of Elixir development, configuration management is a cornerstone for building scalable, maintainable, and secure applications. As systems grow in complexity, managing configurations dynamically and securely becomes paramount. This section delves into the intricacies of configuration management in Elixir, focusing on dynamic configurations, secrets management, and the integration of powerful tools like Consul and Vault.

### Dynamic Configurations

Dynamic configurations allow applications to adapt to different environments and conditions without requiring code changes. This flexibility is crucial for deploying applications across various stages, such as development, testing, and production.

#### Using Environment Variables

Environment variables are a common method for managing configurations dynamically. They provide a simple way to inject configuration values into your application at runtime.

**Example: Setting Environment Variables**

```bash
# Setting environment variables in a Unix-based system
export DATABASE_URL="postgres://user:password@localhost/db_name"
export SECRET_KEY_BASE="your_secret_key"
```

**Accessing Environment Variables in Elixir**

In Elixir, you can access environment variables using the `System.get_env/1` function.

```elixir
defmodule MyApp.Config do
  def database_url do
    System.get_env("DATABASE_URL")
  end

  def secret_key_base do
    System.get_env("SECRET_KEY_BASE")
  end
end
```

**Advantages of Using Environment Variables**

- **Portability:** Easily move configurations between environments.
- **Security:** Keep sensitive information out of source code.
- **Flexibility:** Change configurations without redeploying the application.

#### Runtime Configuration

Elixir provides robust support for runtime configuration through the `config/runtime.exs` file. This file is evaluated at runtime, allowing you to tailor configurations based on the environment.

**Example: Using `config/runtime.exs`**

```elixir
import Config

config :my_app, MyApp.Repo,
  url: System.get_env("DATABASE_URL"),
  pool_size: String.to_integer(System.get_env("POOL_SIZE") || "10")
```

**Benefits of Runtime Configuration**

- **Dynamic Adjustments:** Modify configurations based on runtime conditions.
- **Environment-Specific Settings:** Customize settings for different deployment environments.

### Secrets Management

Managing secrets securely is critical to protect sensitive data such as API keys, passwords, and certificates. Elixir applications can leverage various strategies and tools to handle secrets effectively.

#### Securely Managing Sensitive Configurations

**Principles of Secrets Management**

1. **Minimize Exposure:** Limit the number of places where secrets are stored.
2. **Access Control:** Restrict access to secrets based on roles and responsibilities.
3. **Audit and Monitor:** Track access and changes to secrets for security audits.

**Using Encrypted Secrets**

Elixir's `:crypto` module can be used to encrypt and decrypt sensitive data.

```elixir
defmodule MyApp.Encryption do
  @key :crypto.strong_rand_bytes(32)

  def encrypt(data) do
    :crypto.block_encrypt(:aes_gcm, @key, <<>>, data)
  end

  def decrypt(encrypted_data) do
    :crypto.block_decrypt(:aes_gcm, @key, <<>>, encrypted_data)
  end
end
```

**Key Management**

- **Rotate Keys Regularly:** Change encryption keys periodically to reduce the risk of exposure.
- **Store Keys Securely:** Use hardware security modules (HSMs) or key management services (KMS) for storing keys.

### Consul and Vault Integration

Integrating tools like Consul and Vault can enhance your application's configuration management capabilities, especially in distributed systems.

#### Leveraging Consul for Distributed Configuration Management

Consul is a service mesh solution that provides service discovery, configuration, and segmentation functionality.

**Using Consul for Configuration**

Consul can store configuration data in key-value pairs, which can be accessed by your Elixir application.

**Example: Fetching Configuration from Consul**

```elixir
defmodule MyApp.ConsulConfig do
  @consul_url "http://localhost:8500/v1/kv/"

  def get_config(key) do
    url = "#{@consul_url}#{key}"
    {:ok, response} = HTTPoison.get(url)
    response.body |> Jason.decode!() |> List.first() |> Map.get("Value") |> Base.decode64!()
  end
end
```

**Benefits of Using Consul**

- **Centralized Configuration:** Manage configurations from a single location.
- **Dynamic Updates:** Automatically update configurations across services.

#### Vault for Secrets Management

Vault is a tool for securely accessing secrets. It provides a unified interface to any secret while maintaining tight access control and logging a detailed audit trail.

**Integrating Vault with Elixir**

To use Vault in Elixir, you can leverage libraries like `ex_vault`.

**Example: Accessing Secrets from Vault**

```elixir
defmodule MyApp.Vault do
  @vault_url "http://localhost:8200/v1/secret/data/myapp"

  def get_secret(key) do
    {:ok, response} = HTTPoison.get("#{@vault_url}/#{key}")
    response.body |> Jason.decode!() |> Map.get("data") |> Map.get("data") |> Map.get(key)
  end
end
```

**Advantages of Using Vault**

- **Secure Secret Storage:** Store secrets securely with encryption.
- **Access Policies:** Define who can access which secrets.
- **Audit Logging:** Keep track of secret access and modifications.

### Try It Yourself

Experiment with the code examples provided by modifying them to suit your application's needs. Try integrating Consul and Vault into a sample Elixir application to see how they can enhance your configuration management strategy.

### Visualizing Configuration Management

Below is a diagram illustrating the flow of configuration management in an Elixir application using Consul and Vault.

```mermaid
flowchart TD
    A[Start] --> B[Load Environment Variables]
    B --> C{Use Runtime Config}
    C -->|Yes| D[Load from config/runtime.exs]
    C -->|No| E[Use Default Config]
    D --> F[Access Consul for Config]
    E --> F
    F --> G[Access Vault for Secrets]
    G --> H[Application Runs with Config]
```

**Diagram Description:** This flowchart shows how an Elixir application loads configurations. It starts by loading environment variables, checks if runtime configuration is used, accesses Consul for additional configurations, and retrieves secrets from Vault before running the application.

### References and Links

- [Elixir Environment Variables](https://hexdocs.pm/elixir/System.html#get_env/1)
- [Elixir Runtime Configuration](https://hexdocs.pm/mix/Mix.Config.html)
- [Consul by HashiCorp](https://www.consul.io/)
- [Vault by HashiCorp](https://www.vaultproject.io/)
- [ExVault Library](https://hex.pm/packages/ex_vault)

### Knowledge Check

- How can environment variables enhance the security of your Elixir application?
- What are the benefits of using runtime configuration in Elixir?
- How does Vault help in managing secrets securely?

### Embrace the Journey

Remember, mastering configuration management is a journey. As you progress, you'll discover more ways to optimize and secure your Elixir applications. Keep experimenting, stay curious, and enjoy the journey!

## Quiz: Configuration Management

{{< quizdown >}}

### What is a primary benefit of using environment variables for configuration?

- [x] They keep sensitive information out of source code.
- [ ] They increase application performance.
- [ ] They reduce the need for documentation.
- [ ] They simplify code logic.

> **Explanation:** Environment variables help keep sensitive information, such as API keys and passwords, out of the source code, enhancing security.

### How does the `config/runtime.exs` file benefit Elixir applications?

- [x] It allows for dynamic configuration based on runtime conditions.
- [ ] It improves application startup time.
- [ ] It provides a graphical interface for configuration.
- [ ] It replaces the need for environment variables.

> **Explanation:** The `config/runtime.exs` file is evaluated at runtime, allowing for dynamic adjustments based on the environment or other runtime conditions.

### What is a key principle of secrets management?

- [x] Minimize exposure of secrets.
- [ ] Store secrets in plain text for easy access.
- [ ] Share secrets with all team members.
- [ ] Avoid auditing access to secrets.

> **Explanation:** Minimizing exposure of secrets is crucial to reduce the risk of unauthorized access and potential security breaches.

### Which tool provides a unified interface for accessing secrets securely?

- [x] Vault
- [ ] Consul
- [ ] Docker
- [ ] Kubernetes

> **Explanation:** Vault provides a unified interface for accessing secrets securely, with features like encryption, access policies, and audit logging.

### What is a benefit of using Consul for configuration management?

- [x] Centralized configuration management.
- [ ] Increased application speed.
- [ ] Reduced code complexity.
- [ ] Automatic code generation.

> **Explanation:** Consul allows for centralized configuration management, making it easier to manage and update configurations across distributed systems.

### How can Elixir applications access environment variables?

- [x] Using the `System.get_env/1` function.
- [ ] Through the `File.read/1` function.
- [ ] By importing them in `config.exs`.
- [ ] Using the `IO.puts/1` function.

> **Explanation:** Elixir applications can access environment variables using the `System.get_env/1` function, which retrieves the value of the specified environment variable.

### What is a recommended practice for managing encryption keys?

- [x] Rotate keys regularly.
- [ ] Store keys in source code.
- [ ] Share keys with all developers.
- [ ] Use the same key for all applications.

> **Explanation:** Regularly rotating encryption keys helps reduce the risk of exposure and enhances security.

### Which Elixir module can be used for encrypting data?

- [x] `:crypto`
- [ ] `:logger`
- [ ] `:ecto`
- [ ] `:phoenix`

> **Explanation:** The `:crypto` module in Elixir provides functions for encrypting and decrypting data, making it suitable for handling sensitive information securely.

### What is a feature of Vault that enhances security?

- [x] Audit logging of secret access.
- [ ] Automatic code refactoring.
- [ ] Real-time performance monitoring.
- [ ] Built-in machine learning algorithms.

> **Explanation:** Vault's audit logging feature tracks access to secrets, providing a detailed audit trail that enhances security and accountability.

### True or False: Consul can only be used for service discovery, not for configuration management.

- [ ] True
- [x] False

> **Explanation:** Consul can be used for both service discovery and configuration management, providing a versatile tool for managing distributed systems.

{{< /quizdown >}}
