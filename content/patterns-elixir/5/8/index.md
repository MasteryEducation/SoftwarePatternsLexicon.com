---
canonical: "https://softwarepatternslexicon.com/patterns-elixir/5/8"
title: "Dynamic Module Creation at Runtime: Mastering Elixir Metaprogramming"
description: "Explore dynamic module creation in Elixir using advanced metaprogramming techniques. Learn how to generate modules and functions at runtime to build scalable and flexible applications."
linkTitle: "5.8. Dynamic Module Creation at Runtime"
categories:
- Elixir
- Metaprogramming
- Functional Programming
tags:
- Elixir
- Metaprogramming
- Dynamic Module Creation
- Macros
- Runtime Code Generation
date: 2024-11-23
type: docs
nav_weight: 58000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 5.8. Dynamic Module Creation at Runtime

In the world of Elixir, dynamic module creation is a powerful feature that allows developers to generate modules and functions at runtime. This capability is particularly useful for building scalable and flexible applications, where the structure of the code can adapt to different inputs or configurations. In this section, we will delve into the intricacies of dynamic module creation using Elixir's metaprogramming techniques, focusing on macros and runtime code generation. We will explore practical examples, such as building Domain-Specific Languages (DSLs) and code generation tools, to illustrate these concepts.

### Metaprogramming Techniques

Metaprogramming in Elixir is the practice of writing code that writes code. This is achieved primarily through macros, which allow developers to manipulate the abstract syntax tree (AST) of Elixir code. By using macros, we can dynamically generate modules and functions, enabling a high degree of flexibility and abstraction in our applications.

#### Using Macros to Generate Modules and Functions

Macros in Elixir are a powerful feature that allows developers to extend the language's capabilities. They operate at compile time, transforming code before it is executed. This makes them ideal for generating modules and functions dynamically.

**Example: Creating a Simple Macro**

Let's start with a simple example to demonstrate how macros can be used to generate functions dynamically.

```elixir
defmodule FunctionGenerator do
  defmacro create_function(name, body) do
    quote do
      def unquote(name)(), do: unquote(body)
    end
  end
end

defmodule MyModule do
  require FunctionGenerator

  FunctionGenerator.create_function(:hello, "Hello, world!")
end

IO.puts MyModule.hello() # Output: Hello, world!
```

In this example, the `create_function` macro generates a function named `hello` with a body that returns the string "Hello, world!". The `unquote` function is used to inject the macro arguments into the generated code.

**Advanced Macro Techniques**

Macros can be used to generate entire modules, not just functions. This is particularly useful for creating boilerplate code or implementing patterns that require repetitive structures.

```elixir
defmodule ModuleGenerator do
  defmacro create_module(module_name, functions) do
    quote do
      defmodule unquote(module_name) do
        Enum.each(unquote(functions), fn {name, body} ->
          def unquote(name)(), do: unquote(body)
        end)
      end
    end
  end
end

defmodule MyDynamicModules do
  require ModuleGenerator

  ModuleGenerator.create_module(MyNewModule, [hello: "Hello", goodbye: "Goodbye"])
end

IO.puts MyNewModule.hello()   # Output: Hello
IO.puts MyNewModule.goodbye() # Output: Goodbye
```

Here, the `create_module` macro generates a new module with multiple functions based on the provided list of function names and bodies.

### Runtime Code Generation

While macros operate at compile time, Elixir also allows for runtime code generation, which can be useful when the structure of the code needs to be determined based on runtime data. This is achieved through Elixir's ability to compile and evaluate code dynamically.

#### Creating Modules Based on Runtime Data

To create modules at runtime, you can use the `Code.eval_string/2` function, which evaluates a string as Elixir code. This can be particularly useful for generating modules based on configuration files or user input.

**Example: Generating a Module from Configuration**

```elixir
defmodule ConfigModuleBuilder do
  def create_module_from_config(config) do
    module_name = config[:module_name]
    functions = config[:functions]

    module_code = """
    defmodule #{module_name} do
      #{Enum.map(functions, fn {name, body} ->
          "def #{name}(), do: #{inspect(body)}"
        end) |> Enum.join("\n")}
    end
    """

    Code.eval_string(module_code)
  end
end

config = %{
  module_name: "DynamicConfigModule",
  functions: [foo: "Foo!", bar: "Bar!"]
}

ConfigModuleBuilder.create_module_from_config(config)

IO.puts DynamicConfigModule.foo() # Output: Foo!
IO.puts DynamicConfigModule.bar() # Output: Bar!
```

In this example, a module is generated at runtime based on a configuration map. The `create_module_from_config` function constructs a string of Elixir code and evaluates it using `Code.eval_string/2`.

### Examples: Building DSLs and Code Generation Tools

Dynamic module creation is a cornerstone for building Domain-Specific Languages (DSLs) and code generation tools. These applications often require the ability to adapt the code structure dynamically based on specific requirements or inputs.

#### Building a DSL for Data Processing

Let's consider an example where we build a simple DSL for data processing tasks. This DSL will allow users to define data transformations in a concise and expressive manner.

```elixir
defmodule DataDSL do
  defmacro transform(data, operations) do
    quote do
      Enum.reduce(unquote(operations), unquote(data), fn {op, arg}, acc ->
        apply(__MODULE__, op, [acc, arg])
      end)
    end
  end

  def filter(data, condition), do: Enum.filter(data, condition)
  def map(data, func), do: Enum.map(data, func)
end

defmodule DataPipeline do
  require DataDSL

  data = [1, 2, 3, 4, 5]

  result = DataDSL.transform(data, [
    {:filter, &(&1 > 2)},
    {:map, &(&1 * 2)}
  ])

  IO.inspect(result) # Output: [6, 8, 10]
end
```

In this example, the `DataDSL` module defines a macro `transform` that applies a series of operations to a data set. The operations are defined as a list of tuples, where each tuple contains the operation name and its argument.

#### Code Generation for API Clients

Another practical application of dynamic module creation is generating API clients. This approach can automate the creation of client functions based on an API specification, reducing boilerplate code and ensuring consistency.

```elixir
defmodule ApiClientGenerator do
  def generate_client(api_spec) do
    module_name = api_spec[:module_name]
    endpoints = api_spec[:endpoints]

    module_code = """
    defmodule #{module_name} do
      #{Enum.map(endpoints, fn {name, url} ->
          """
          def #{name}() do
            HTTPoison.get("#{url}")
          end
          """
        end) |> Enum.join("\n")}
    end
    """

    Code.eval_string(module_code)
  end
end

api_spec = %{
  module_name: "MyApiClient",
  endpoints: [get_users: "https://api.example.com/users", get_posts: "https://api.example.com/posts"]
}

ApiClientGenerator.generate_client(api_spec)

IO.inspect MyApiClient.get_users()
IO.inspect MyApiClient.get_posts()
```

Here, the `ApiClientGenerator` module generates a client module with functions for each API endpoint specified in the `api_spec` map. The functions use `HTTPoison` to make HTTP GET requests to the specified URLs.

### Design Considerations

When using dynamic module creation, it's essential to consider the following:

- **Performance:** Dynamic code generation can introduce overhead, especially if modules are generated frequently at runtime. Consider caching generated modules or limiting dynamic creation to initialization phases.
- **Security:** Evaluating code at runtime can pose security risks, especially if the input is not controlled. Always validate and sanitize inputs before generating code.
- **Maintainability:** Dynamic code can be harder to understand and debug. Ensure that the generated code is well-documented and that the generation logic is clear and concise.

### Elixir Unique Features

Elixir's metaprogramming capabilities, such as macros and runtime code evaluation, make it uniquely suited for dynamic module creation. The language's emphasis on immutability and functional programming ensures that dynamically generated code remains efficient and reliable.

### Differences and Similarities

Dynamic module creation is often confused with other metaprogramming techniques, such as reflection or introspection. While reflection allows a program to examine its own structure, dynamic module creation involves generating new structures at runtime. Understanding these distinctions is crucial for applying the right technique to a given problem.

### Try It Yourself

To deepen your understanding of dynamic module creation, try modifying the examples provided:

- **Experiment with Different Macros:** Create macros that generate modules with different structures or behaviors.
- **Generate Modules from User Input:** Build a simple application that generates modules based on user-defined configurations.
- **Extend the DSL Example:** Add more operations to the data processing DSL, such as sorting or reducing operations.

### Conclusion

Dynamic module creation in Elixir is a powerful tool for building flexible and scalable applications. By leveraging metaprogramming techniques, developers can generate modules and functions that adapt to different inputs and requirements. As you continue to explore Elixir's capabilities, remember to balance the power of dynamic code with considerations for performance, security, and maintainability.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of macros in Elixir?

- [x] To transform code at compile time
- [ ] To execute code at runtime
- [ ] To handle errors in Elixir applications
- [ ] To manage concurrency in Elixir

> **Explanation:** Macros in Elixir are used to transform code at compile time, allowing developers to extend the language's capabilities.

### How can you evaluate a string as Elixir code at runtime?

- [x] Using `Code.eval_string/2`
- [ ] Using `Kernel.apply/3`
- [ ] Using `Enum.map/2`
- [ ] Using `String.to_atom/1`

> **Explanation:** `Code.eval_string/2` is used to evaluate a string as Elixir code at runtime.

### What is a potential risk of evaluating code at runtime?

- [x] Security vulnerabilities
- [ ] Increased code readability
- [ ] Improved performance
- [ ] Reduced code maintainability

> **Explanation:** Evaluating code at runtime can pose security risks, especially if the input is not controlled or sanitized.

### Which of the following is a benefit of using dynamic module creation?

- [x] Increased flexibility in application design
- [ ] Improved static analysis
- [ ] Simplified error handling
- [ ] Enhanced type safety

> **Explanation:** Dynamic module creation allows for increased flexibility, enabling applications to adapt to different inputs and configurations.

### In the context of Elixir, what does DSL stand for?

- [x] Domain-Specific Language
- [ ] Dynamic Scripting Language
- [ ] Data Serialization Language
- [ ] Distributed Systems Language

> **Explanation:** DSL stands for Domain-Specific Language, which is a specialized language tailored to a specific application domain.

### What should be considered when using dynamic module creation?

- [x] Performance, security, and maintainability
- [ ] Only performance
- [ ] Only security
- [ ] Only maintainability

> **Explanation:** When using dynamic module creation, it's important to consider performance, security, and maintainability.

### How can macros be used to generate entire modules?

- [x] By manipulating the abstract syntax tree (AST)
- [ ] By using reflection
- [ ] By using introspection
- [ ] By using pattern matching

> **Explanation:** Macros can generate entire modules by manipulating the abstract syntax tree (AST) of Elixir code.

### What is the role of `unquote` in a macro?

- [x] To inject macro arguments into the generated code
- [ ] To define a new module
- [ ] To handle exceptions
- [ ] To manage process state

> **Explanation:** `unquote` is used in macros to inject macro arguments into the generated code.

### Which function can be used to create a module from configuration data?

- [x] `Code.eval_string/2`
- [ ] `Kernel.spawn/1`
- [ ] `Enum.reduce/3`
- [ ] `Process.send/2`

> **Explanation:** `Code.eval_string/2` can be used to evaluate a string as Elixir code, allowing for module creation from configuration data.

### True or False: Dynamic module creation is only possible at compile time in Elixir.

- [ ] True
- [x] False

> **Explanation:** Dynamic module creation can occur both at compile time using macros and at runtime using functions like `Code.eval_string/2`.

{{< /quizdown >}}

Remember, this is just the beginning. As you progress, you'll build more complex and interactive applications using Elixir's dynamic capabilities. Keep experimenting, stay curious, and enjoy the journey!
