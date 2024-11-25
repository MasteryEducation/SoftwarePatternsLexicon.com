---
canonical: "https://softwarepatternslexicon.com/patterns-elixir/26/5"

title: "Managing Configuration and Secrets in Elixir Applications"
description: "Learn how to effectively manage configuration and secrets in Elixir applications using environment variables, secret management services, and configuration libraries."
linkTitle: "26.5. Managing Configuration and Secrets"
categories:
- Elixir
- Configuration Management
- Secret Management
tags:
- Elixir
- Configuration
- Secrets
- Environment Variables
- Security
date: 2024-11-23
type: docs
nav_weight: 265000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 26.5. Managing Configuration and Secrets

In the realm of software development, managing configuration and secrets is a critical aspect of building secure and maintainable applications. In Elixir, this involves leveraging environment variables, secret management services, and configuration libraries to ensure that sensitive information is handled appropriately and that application settings are flexible and easy to manage.

### Environment Variables

Environment variables are a fundamental way to store configuration outside the codebase. This approach provides flexibility and security, as it allows different configurations for different environments (e.g., development, testing, production) without changing the code.

#### Storing Configuration Outside the Codebase

Storing configuration in environment variables helps in keeping sensitive information out of the codebase. This practice is crucial for maintaining security and ensuring that secrets are not accidentally exposed in version control systems.

```elixir
# Accessing environment variables in Elixir
database_url = System.get_env("DATABASE_URL")
```

In this example, the `DATABASE_URL` is retrieved from the environment, allowing the application to connect to the database without hardcoding the connection string.

#### Using Tools like `dotenv` for Local Development

For local development, managing environment variables can be simplified using tools like `dotenv`. This tool allows developers to define environment variables in a `.env` file, which can be loaded into the application environment.

```elixir
# .env file
DATABASE_URL=postgres://user:password@localhost/db_name
```

To load these variables in Elixir, you can use the `dotenv` library:

```elixir
# Load .env file
Dotenv.load()

# Access the environment variable
database_url = System.get_env("DATABASE_URL")
```

This approach ensures that developers can easily manage their local configurations without affecting the production environment.

### Secret Management

Managing secrets securely is paramount to protecting sensitive information such as API keys, passwords, and certificates. Using dedicated secret management services can help achieve this.

#### Using Services like HashiCorp Vault or AWS Secrets Manager

Services like HashiCorp Vault and AWS Secrets Manager provide robust solutions for managing secrets. They offer features such as encryption, access control, and auditing, which are essential for maintaining the security of sensitive data.

**HashiCorp Vault Example:**

```elixir
# Fetching secrets from HashiCorp Vault
{:ok, secret} = Vault.read("secret/data/myapp")
api_key = secret["data"]["api_key"]
```

**AWS Secrets Manager Example:**

```elixir
# Fetching secrets from AWS Secrets Manager
{:ok, secret} = ExAws.SecretsManager.get_secret_value("myapp/api_key")
api_key = secret["SecretString"]
```

These services ensure that secrets are stored securely and accessed only by authorized applications and users.

#### Avoiding Hard-Coded Secrets in Code Repositories

Hard-coding secrets in code repositories is a common security pitfall. Instead, secrets should be stored in environment variables or secret management services, as discussed earlier. This practice prevents accidental exposure of sensitive information and enhances the security posture of the application.

### Configuration Libraries

Elixir provides powerful configuration libraries that enable developers to manage application settings efficiently.

#### Leveraging Libraries like `Config` for Runtime Configuration

The `Config` module in Elixir is a versatile tool for managing application configuration. It allows for runtime configuration, which means settings can be adjusted without recompiling the application.

```elixir
# config/config.exs
import Config

config :my_app, MyApp.Repo,
  username: System.get_env("DB_USERNAME"),
  password: System.get_env("DB_PASSWORD"),
  database: System.get_env("DB_NAME"),
  hostname: System.get_env("DB_HOST"),
  pool_size: 10
```

This configuration file uses environment variables to set database connection parameters, ensuring that sensitive information is not hardcoded.

#### Structuring Configurations for Clarity and Maintainability

Organizing configuration files is crucial for clarity and maintainability. Group related settings together and use descriptive keys to make the configuration intuitive.

```elixir
# config/config.exs
config :my_app,
  ecto_repos: [MyApp.Repo]

config :my_app, MyApp.Repo,
  username: System.get_env("DB_USERNAME"),
  password: System.get_env("DB_PASSWORD"),
  database: System.get_env("DB_NAME"),
  hostname: System.get_env("DB_HOST"),
  pool_size: 10

config :my_app, :api,
  endpoint: System.get_env("API_ENDPOINT"),
  key: System.get_env("API_KEY")
```

By structuring configurations in this manner, you ensure that the application settings are easy to understand and modify.

### Visualizing Configuration Management

To better understand the flow of configuration management in Elixir applications, let's visualize the process using a Mermaid.js diagram.

```mermaid
flowchart TD
    A[Start] --> B[Load Environment Variables]
    B --> C{Use dotenv for Local Development?}
    C -->|Yes| D[Load .env File]
    C -->|No| E[Skip]
    D --> F[Access Environment Variables]
    E --> F
    F --> G[Use Secret Management Service]
    G --> H[Fetch Secrets Securely]
    H --> I[Configure Application with Config Module]
    I --> J[Run Application]
    J --> K[End]
```

This diagram illustrates the steps involved in managing configuration and secrets in an Elixir application, from loading environment variables to running the application with secure settings.

### Knowledge Check

To reinforce your understanding of managing configuration and secrets in Elixir, consider the following questions:

- Why is it important to store configuration outside the codebase?
- How can tools like `dotenv` simplify local development?
- What are the benefits of using secret management services like HashiCorp Vault?
- How does the `Config` module facilitate runtime configuration in Elixir?
- What are some best practices for structuring configuration files?

### Embrace the Journey

Remember, managing configuration and secrets is a continuous process that evolves with your application's needs. As you progress, you'll find more efficient ways to handle settings and secure sensitive information. Keep experimenting, stay curious, and enjoy the journey!

### Quiz: Managing Configuration and Secrets

{{< quizdown >}}

### What is the primary benefit of storing configuration in environment variables?

- [x] It keeps sensitive information out of the codebase.
- [ ] It makes the codebase larger.
- [ ] It requires more memory.
- [ ] It complicates the deployment process.

> **Explanation:** Storing configuration in environment variables helps keep sensitive information out of the codebase, enhancing security.

### Which tool can be used to manage environment variables for local development?

- [x] dotenv
- [ ] npm
- [ ] pip
- [ ] bundler

> **Explanation:** `dotenv` is a tool that helps manage environment variables for local development by loading them from a `.env` file.

### What is a key feature of secret management services like HashiCorp Vault?

- [x] Encryption and access control
- [ ] Code compilation
- [ ] Database management
- [ ] User interface design

> **Explanation:** Secret management services like HashiCorp Vault provide encryption and access control to securely manage sensitive information.

### How does the `Config` module in Elixir help with application configuration?

- [x] It allows for runtime configuration.
- [ ] It compiles the code.
- [ ] It manages database connections.
- [ ] It designs user interfaces.

> **Explanation:** The `Config` module in Elixir allows for runtime configuration, enabling settings to be adjusted without recompiling the application.

### Why should secrets not be hard-coded in code repositories?

- [x] To prevent accidental exposure of sensitive information
- [ ] To increase code complexity
- [ ] To reduce code readability
- [ ] To make the codebase larger

> **Explanation:** Hard-coding secrets in code repositories can lead to accidental exposure of sensitive information, compromising security.

### What is the purpose of structuring configuration files?

- [x] For clarity and maintainability
- [ ] To increase file size
- [ ] To reduce code readability
- [ ] To complicate deployment

> **Explanation:** Structuring configuration files helps in maintaining clarity and making them easy to understand and modify.

### Which of the following is a benefit of using environment variables?

- [x] Flexibility in different environments
- [ ] Increased code size
- [ ] Reduced security
- [ ] Complicated deployment

> **Explanation:** Environment variables provide flexibility by allowing different configurations for different environments without changing the code.

### What is a common security pitfall in managing secrets?

- [x] Hard-coding secrets in code repositories
- [ ] Using environment variables
- [ ] Using secret management services
- [ ] Structuring configuration files

> **Explanation:** Hard-coding secrets in code repositories is a common security pitfall that can lead to accidental exposure of sensitive information.

### How can secret management services enhance security?

- [x] By providing encryption and access control
- [ ] By increasing code complexity
- [ ] By reducing code readability
- [ ] By making the codebase larger

> **Explanation:** Secret management services enhance security by providing encryption and access control, ensuring that sensitive information is securely stored and accessed.

### True or False: The `Config` module in Elixir allows for compile-time configuration only.

- [ ] True
- [x] False

> **Explanation:** The `Config` module in Elixir allows for runtime configuration, enabling settings to be adjusted without recompiling the application.

{{< /quizdown >}}

By mastering the management of configuration and secrets in Elixir, you can build secure, flexible, and maintainable applications that are ready to adapt to changing environments and requirements.
