---
canonical: "https://softwarepatternslexicon.com/patterns-erlang/14/3"
title: "RESTful Services and HTTP Clients in Erlang: Mastering `httpc` and `hackney`"
description: "Explore how to effectively work with RESTful services in Erlang using `httpc` and `hackney`. Learn about HTTP requests, response handling, authentication, error management, and best practices."
linkTitle: "14.3 Working with RESTful Services and HTTP Clients (`httpc`, `hackney`)"
categories:
- Erlang
- RESTful Services
- HTTP Clients
tags:
- Erlang
- HTTP
- REST
- httpc
- hackney
date: 2024-11-23
type: docs
nav_weight: 143000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 14.3 Working with RESTful Services and HTTP Clients (`httpc`, `hackney`)

In the modern software landscape, interacting with RESTful services is a common requirement. Erlang, with its robust concurrency model, provides several tools to facilitate HTTP communication. In this section, we will explore how to work with RESTful services using Erlang's built-in HTTP client `httpc` and the popular third-party library `hackney`. We will cover making HTTP requests, handling responses, managing headers and authentication, and best practices for error handling and resource management.

### Introduction to HTTP Clients in Erlang

Erlang offers a built-in HTTP client, `httpc`, which is part of the `inets` application. It provides basic functionality for making HTTP requests. However, for more advanced features, such as streaming responses or handling complex authentication schemes, the `hackney` library is often preferred.

#### The `httpc` Client

The `httpc` client is a straightforward choice for basic HTTP operations. It supports synchronous and asynchronous requests, making it suitable for various use cases.

#### The `hackney` Library

`hackney` is a more feature-rich HTTP client library for Erlang. It supports connection pooling, streaming, and advanced authentication mechanisms. It is widely used in the Erlang community for its flexibility and performance.

### Making HTTP Requests

Let's start by making simple HTTP requests using both `httpc` and `hackney`.

#### Using `httpc`

To use `httpc`, you must first start the `inets` application:

```erlang
% Start the inets application
ok = application:start(inets).

% Make a GET request
{ok, {{_, 200, _}, _, Body}} = httpc:request(get, {"http://example.com", []}, [], []).

% Print the response body
io:format("Response: ~s~n", [Body]).
```

In this example, we make a GET request to `http://example.com` and print the response body. The `httpc:request/4` function takes the HTTP method, URL, options, and headers as arguments.

#### Using `hackney`

To use `hackney`, you need to add it to your project's dependencies and start it:

```erlang
% Add hackney to your rebar.config
{deps, [
    {hackney, "4.0.0"}
]}.

% Start hackney
ok = application:start(hackney).

% Make a GET request
{ok, StatusCode, Headers, ClientRef} = hackney:get("http://example.com", []).

% Read the response body
{ok, Body} = hackney:body(ClientRef).

% Print the response body
io:format("Response: ~s~n", [Body]).
```

Here, we use `hackney:get/2` to make a GET request. The response is handled using a client reference, which allows for more complex operations like streaming.

### Handling HTTP Methods

Both `httpc` and `hackney` support various HTTP methods, including GET, POST, PUT, DELETE, and more. Let's see how to handle these methods.

#### POST Requests with `httpc`

```erlang
% Prepare the POST data
Data = "key1=value1&key2=value2".

% Make a POST request
{ok, {{_, 200, _}, _, Body}} = httpc:request(post, {"http://example.com", [], "application/x-www-form-urlencoded", Data}, [], []).

% Print the response body
io:format("Response: ~s~n", [Body]).
```

#### POST Requests with `hackney`

```erlang
% Prepare the POST data
Data = <<"key1=value1&key2=value2">>.

% Make a POST request
{ok, StatusCode, Headers, ClientRef} = hackney:post("http://example.com", [{"Content-Type", "application/x-www-form-urlencoded"}], Data, []).

% Read the response body
{ok, Body} = hackney:body(ClientRef).

% Print the response body
io:format("Response: ~s~n", [Body]).
```

### Managing Headers and Authentication

Handling headers and authentication is crucial for interacting with RESTful services.

#### Setting Headers

Headers can be set by passing a list of key-value tuples to the request functions.

```erlang
% Set custom headers with httpc
Headers = [{"Authorization", "Bearer token"}, {"Accept", "application/json"}].
{ok, {{_, 200, _}, _, Body}} = httpc:request(get, {"http://example.com", Headers}, [], []).

% Set custom headers with hackney
Headers = [{"Authorization", "Bearer token"}, {"Accept", "application/json"}].
{ok, StatusCode, Headers, ClientRef} = hackney:get("http://example.com", Headers).
```

#### Basic Authentication

For basic authentication, you can include the credentials in the URL or use the `Authorization` header.

```erlang
% Basic authentication with httpc
{ok, {{_, 200, _}, _, Body}} = httpc:request(get, {"http://user:pass@example.com", []}, [], []).

% Basic authentication with hackney
Options = [{basic_auth, {"user", "pass"}}].
{ok, StatusCode, Headers, ClientRef} = hackney:get("http://example.com", [], <<>>, Options).
```

### Error Handling and Retries

Handling errors and implementing retries are essential for robust HTTP communication.

#### Error Handling

Both `httpc` and `hackney` return tuples indicating success or failure. You should handle these cases appropriately.

```erlang
% Error handling with httpc
case httpc:request(get, {"http://example.com", []}, [], []) of
    {ok, {{_, 200, _}, _, Body}} ->
        io:format("Success: ~s~n", [Body]);
    {error, Reason} ->
        io:format("Error: ~p~n", [Reason])
end.

% Error handling with hackney
case hackney:get("http://example.com", []) of
    {ok, StatusCode, Headers, ClientRef} when StatusCode == 200 ->
        {ok, Body} = hackney:body(ClientRef),
        io:format("Success: ~s~n", [Body]);
    {error, Reason} ->
        io:format("Error: ~p~n", [Reason])
end.
```

#### Implementing Retries

Retries can be implemented using recursion or loops. Here's a simple example using recursion:

```erlang
% Retry logic with httpc
retry_request(Url, Attempts) when Attempts > 0 ->
    case httpc:request(get, {Url, []}, [], []) of
        {ok, {{_, 200, _}, _, Body}} ->
            io:format("Success: ~s~n", [Body]);
        {error, Reason} ->
            io:format("Retrying due to: ~p~n", [Reason]),
            retry_request(Url, Attempts - 1)
    end;
retry_request(_, 0) ->
    io:format("Failed after retries~n").
```

### Resource Management

Proper resource management is crucial to prevent leaks and ensure efficient use of system resources.

#### Closing Connections

With `hackney`, it's important to close connections when they are no longer needed.

```erlang
% Ensure to close the connection
{ok, StatusCode, Headers, ClientRef} = hackney:get("http://example.com", []),
{ok, Body} = hackney:body(ClientRef),
hackney:close(ClientRef).
```

### Best Practices

- **Use Connection Pooling**: For `hackney`, enable connection pooling to reuse connections and reduce latency.
- **Handle Timeouts**: Set appropriate timeouts to avoid hanging requests.
- **Log Errors**: Log errors for monitoring and debugging purposes.
- **Secure Credentials**: Never hardcode credentials; use environment variables or secure vaults.

### Visualizing HTTP Client Interaction

To better understand the flow of HTTP client interaction, let's visualize it using a sequence diagram.

```mermaid
sequenceDiagram
    participant Client
    participant Server
    Client->>Server: HTTP Request
    Server-->>Client: HTTP Response
    Client->>Client: Process Response
```

This diagram illustrates the basic interaction between a client and a server during an HTTP request and response cycle.

### Try It Yourself

Experiment with the code examples provided. Try modifying the URLs, headers, and data to see how the clients handle different scenarios. Implement additional error handling or retry logic to strengthen your understanding.

### References and Further Reading

- [Erlang `httpc` Documentation](https://erlang.org/doc/man/httpc.html)
- [Hackney GitHub Repository](https://github.com/benoitc/hackney)
- [RESTful Web Services](https://restfulapi.net/)

### Knowledge Check

- What are the key differences between `httpc` and `hackney`?
- How can you implement retries for failed HTTP requests?
- Why is it important to manage resources like connections?

### Embrace the Journey

Remember, mastering HTTP clients in Erlang is just one step in building robust, scalable applications. Keep experimenting, stay curious, and enjoy the journey!

## Quiz: Working with RESTful Services and HTTP Clients (`httpc`, `hackney`)

{{< quizdown >}}

### What is the primary purpose of the `httpc` client in Erlang?

- [x] To make HTTP requests
- [ ] To compile Erlang code
- [ ] To manage databases
- [ ] To handle file I/O

> **Explanation:** The `httpc` client is used to make HTTP requests in Erlang.

### Which library provides more advanced features like connection pooling and streaming?

- [ ] `httpc`
- [x] `hackney`
- [ ] `inets`
- [ ] `gen_server`

> **Explanation:** `hackney` provides advanced features such as connection pooling and streaming.

### How do you start the `inets` application to use `httpc`?

- [x] `ok = application:start(inets).`
- [ ] `ok = inets:start().`
- [ ] `start(inets).`
- [ ] `application:load(inets).`

> **Explanation:** You start the `inets` application with `ok = application:start(inets).`

### What is the correct way to set headers in a `hackney` request?

- [x] Pass a list of key-value tuples as the second argument
- [ ] Include them in the URL
- [ ] Use the `set_headers` function
- [ ] Add them to the request body

> **Explanation:** Headers are set by passing a list of key-value tuples as the second argument in a `hackney` request.

### How can you implement basic authentication with `hackney`?

- [x] Use the `basic_auth` option in the request
- [ ] Include credentials in the URL
- [ ] Use the `auth` function
- [ ] Set the `Authorization` header manually

> **Explanation:** Basic authentication can be implemented using the `basic_auth` option in the request.

### What should you do to prevent resource leaks in `hackney`?

- [x] Close connections when they are no longer needed
- [ ] Use more memory
- [ ] Increase the number of processes
- [ ] Ignore the issue

> **Explanation:** Closing connections when they are no longer needed prevents resource leaks.

### What is a common practice to handle failed HTTP requests?

- [x] Implement retries
- [ ] Ignore them
- [ ] Log them and move on
- [ ] Increase timeout

> **Explanation:** Implementing retries is a common practice to handle failed HTTP requests.

### Why is it important to set timeouts for HTTP requests?

- [x] To avoid hanging requests
- [ ] To increase request speed
- [ ] To reduce server load
- [ ] To improve security

> **Explanation:** Setting timeouts helps avoid hanging requests.

### What is a best practice for handling credentials in HTTP requests?

- [x] Use environment variables or secure vaults
- [ ] Hardcode them in the source code
- [ ] Share them publicly
- [ ] Store them in plain text files

> **Explanation:** Using environment variables or secure vaults is a best practice for handling credentials.

### True or False: `hackney` supports streaming responses.

- [x] True
- [ ] False

> **Explanation:** `hackney` supports streaming responses, making it suitable for handling large data transfers.

{{< /quizdown >}}
