---
canonical: "https://softwarepatternslexicon.com/patterns-erlang/14/7"
title: "Communicating with External Services and APIs in Erlang"
description: "Explore strategies for integrating Erlang applications with external services and APIs, including REST, SOAP, and GraphQL, with a focus on authentication, error handling, and testing."
linkTitle: "14.7 Communicating with External Services and APIs"
categories:
- Erlang
- Integration
- APIs
tags:
- Erlang
- External Services
- APIs
- REST
- SOAP
- GraphQL
date: 2024-11-23
type: docs
nav_weight: 147000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 14.7 Communicating with External Services and APIs

In today's interconnected world, applications rarely operate in isolation. They often need to communicate with external services and APIs to fetch data, perform operations, or integrate with other systems. Erlang, with its robust concurrency model and fault-tolerant design, is well-suited for building applications that interact with external services. In this section, we will explore various strategies for integrating Erlang applications with external services and APIs, including REST, SOAP, and GraphQL. We will also discuss authentication mechanisms, rate limiting, API versioning, error handling, and testing.

### Understanding External Services and APIs

External services and APIs provide a way for applications to interact with other systems over the network. They can be used to access data, perform operations, or integrate with third-party services. There are several types of APIs, including REST, SOAP, and GraphQL, each with its own characteristics and use cases.

#### REST APIs

REST (Representational State Transfer) is a popular architectural style for designing networked applications. REST APIs use HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs. They are stateless, meaning each request from a client contains all the information needed to process the request.

#### SOAP APIs

SOAP (Simple Object Access Protocol) is a protocol for exchanging structured information in web services. It uses XML for message format and relies on other application layer protocols, such as HTTP or SMTP, for message negotiation and transmission. SOAP APIs are known for their robustness and extensibility.

#### GraphQL APIs

GraphQL is a query language for APIs and a runtime for executing those queries by using a type system you define for your data. Unlike REST, which exposes multiple endpoints, GraphQL exposes a single endpoint and allows clients to specify exactly what data they need.

### Strategies for Consuming External APIs

When integrating with external APIs, it's essential to choose the right strategy based on the API type and your application's requirements. Let's explore some common strategies for consuming external APIs in Erlang.

#### Using HTTP Clients

Erlang provides several libraries for making HTTP requests, such as `httpc` (part of the `inets` application) and `hackney`. These libraries allow you to send HTTP requests and handle responses, making them suitable for consuming REST and GraphQL APIs.

Here's an example of using `httpc` to make a GET request to a REST API:

```erlang
-module(api_client).
-export([fetch_data/1]).

fetch_data(Url) ->
    % Make a GET request to the specified URL
    case httpc:request(get, {Url, []}, [], []) of
        {ok, {_, _, Body}} ->
            % Handle the response body
            io:format("Response: ~s~n", [Body]);
        {error, Reason} ->
            % Handle the error
            io:format("Error: ~p~n", [Reason])
    end.
```

In this example, we use the `httpc:request/4` function to send a GET request to the specified URL. The response is pattern-matched to handle both successful responses and errors.

#### Consuming SOAP APIs

Consuming SOAP APIs in Erlang can be more complex due to the XML-based message format. Libraries like `erlsom` can be used to parse and generate XML, making it easier to work with SOAP APIs.

Here's a basic example of sending a SOAP request:

```erlang
-module(soap_client).
-export([send_request/1]).

send_request(Url) ->
    % Construct the SOAP request XML
    SoapRequest = "<soapenv:Envelope xmlns:soapenv='http://schemas.xmlsoap.org/soap/envelope/' xmlns:ser='http://service.example.com/'>" ++
                  "<soapenv:Header/>" ++
                  "<soapenv:Body>" ++
                  "<ser:MyRequest>" ++
                  "<ser:Parameter>Value</ser:Parameter>" ++
                  "</ser:MyRequest>" ++
                  "</soapenv:Body>" ++
                  "</soapenv:Envelope>",
    % Send the SOAP request
    case httpc:request(post, {Url, [], "text/xml", SoapRequest}, [], []) of
        {ok, {_, _, Body}} ->
            % Handle the response body
            io:format("Response: ~s~n", [Body]);
        {error, Reason} ->
            % Handle the error
            io:format("Error: ~p~n", [Reason])
    end.
```

In this example, we construct a SOAP request XML and send it using an HTTP POST request. The response is handled similarly to the REST example.

#### Interacting with GraphQL APIs

GraphQL APIs can be consumed using HTTP clients by sending POST requests with the query in the request body. The response is typically in JSON format, which can be parsed using libraries like `jsx` or `jiffy`.

Here's an example of sending a GraphQL query:

```erlang
-module(graphql_client).
-export([fetch_data/1]).

fetch_data(Url) ->
    % Construct the GraphQL query
    Query = "{\"query\": \"{ user(id: \\\"1\\\") { name email } }\"}",
    % Send the GraphQL request
    case httpc:request(post, {Url, [], "application/json", Query}, [], []) of
        {ok, {_, _, Body}} ->
            % Parse and handle the JSON response
            {ok, Json} = jsx:decode(Body),
            io:format("Response: ~p~n", [Json]);
        {error, Reason} ->
            % Handle the error
            io:format("Error: ~p~n", [Reason])
    end.
```

In this example, we construct a GraphQL query and send it as a JSON payload in a POST request. The response is parsed using the `jsx` library.

### Authentication Mechanisms

When communicating with external services, authentication is often required to ensure secure access. Common authentication mechanisms include API keys, OAuth, and JWT (JSON Web Tokens).

#### API Keys

API keys are simple tokens that are included in requests to authenticate the client. They are often passed as headers or query parameters.

Here's an example of using an API key in a request:

```erlang
-module(api_key_client).
-export([fetch_data/2]).

fetch_data(Url, ApiKey) ->
    % Set the API key in the headers
    Headers = [{"Authorization", "Bearer " ++ ApiKey}],
    % Make a GET request with the API key
    case httpc:request(get, {Url, Headers}, [], []) of
        {ok, {_, _, Body}} ->
            io:format("Response: ~s~n", [Body]);
        {error, Reason} ->
            io:format("Error: ~p~n", [Reason])
    end.
```

#### OAuth

OAuth is a more complex authentication mechanism that involves obtaining an access token through an authorization server. This token is then used to authenticate requests.

Here's a simplified example of using OAuth:

```erlang
-module(oauth_client).
-export([fetch_data/3]).

fetch_data(Url, ClientId, ClientSecret) ->
    % Obtain the access token (this is a simplified example)
    AccessToken = obtain_access_token(ClientId, ClientSecret),
    % Set the access token in the headers
    Headers = [{"Authorization", "Bearer " ++ AccessToken}],
    % Make a GET request with the access token
    case httpc:request(get, {Url, Headers}, [], []) of
        {ok, {_, _, Body}} ->
            io:format("Response: ~s~n", [Body]);
        {error, Reason} ->
            io:format("Error: ~p~n", [Reason])
    end.

obtain_access_token(ClientId, ClientSecret) ->
    % This function should implement the OAuth flow to obtain an access token
    "dummy_access_token".
```

In this example, the `obtain_access_token/2` function should implement the OAuth flow to obtain an access token. This is a simplified example, and a real implementation would involve more steps.

### Handling Rate Limiting and API Versioning

When consuming external APIs, it's important to consider rate limiting and API versioning to ensure reliable and stable integration.

#### Rate Limiting

Rate limiting is a mechanism used by APIs to control the number of requests a client can make in a given time period. It's important to handle rate limiting gracefully to avoid being blocked by the API.

Here's an example of handling rate limiting:

```erlang
-module(rate_limit_client).
-export([fetch_data/1]).

fetch_data(Url) ->
    % Make a GET request
    case httpc:request(get, {Url, []}, [], []) of
        {ok, {_, Headers, Body}} ->
            % Check for rate limiting headers
            case lists:keyfind("X-RateLimit-Remaining", 1, Headers) of
                {_, Remaining} when Remaining =:= "0" ->
                    io:format("Rate limit exceeded, try again later~n");
                _ ->
                    io:format("Response: ~s~n", [Body])
            end;
        {error, Reason} ->
            io:format("Error: ~p~n", [Reason])
    end.
```

In this example, we check for rate limiting headers in the response and handle the case where the rate limit is exceeded.

#### API Versioning

APIs often evolve over time, and versioning is a way to manage changes without breaking existing clients. It's important to specify the API version in requests to ensure compatibility.

Here's an example of specifying an API version:

```erlang
-module(versioned_api_client).
-export([fetch_data/2]).

fetch_data(Url, Version) ->
    % Set the API version in the headers
    Headers = [{"Accept", "application/vnd.example.v" ++ Version ++ "+json"}],
    % Make a GET request with the API version
    case httpc:request(get, {Url, Headers}, [], []) of
        {ok, {_, _, Body}} ->
            io:format("Response: ~s~n", [Body]);
        {error, Reason} ->
            io:format("Error: ~p~n", [Reason])
    end.
```

In this example, we specify the API version in the `Accept` header to request a specific version of the API.

### Handling Network Errors Gracefully

Network errors are inevitable when communicating with external services. It's important to handle these errors gracefully to ensure a robust application.

#### Retrying Requests

One strategy for handling network errors is to retry requests after a delay. This can be useful for transient errors, such as network timeouts.

Here's an example of retrying a request:

```erlang
-module(retry_client).
-export([fetch_data/2]).

fetch_data(Url, Retries) when Retries > 0 ->
    case httpc:request(get, {Url, []}, [], []) of
        {ok, {_, _, Body}} ->
            io:format("Response: ~s~n", [Body]);
        {error, Reason} ->
            io:format("Error: ~p, retrying...~n", [Reason]),
            timer:sleep(1000), % Wait for 1 second
            fetch_data(Url, Retries - 1)
    end;
fetch_data(_, 0) ->
    io:format("Failed to fetch data after retries~n").
```

In this example, we retry the request up to a specified number of times, waiting for 1 second between retries.

#### Circuit Breaker Pattern

The circuit breaker pattern is a design pattern used to detect failures and encapsulate the logic of preventing a failure from constantly recurring. It can be used to handle network errors by temporarily blocking requests to a failing service.

Here's a basic implementation of a circuit breaker:

```erlang
-module(circuit_breaker).
-export([fetch_data/1, reset/0]).

-define(THRESHOLD, 3).
-define(TIMEOUT, 5000).

fetch_data(Url) ->
    case get_state() of
        open ->
            io:format("Circuit is open, skipping request~n");
        closed ->
            case httpc:request(get, {Url, []}, [], []) of
                {ok, {_, _, Body}} ->
                    reset(),
                    io:format("Response: ~s~n", [Body]);
                {error, Reason} ->
                    increment_failure(),
                    io:format("Error: ~p~n", [Reason])
            end
    end.

get_state() ->
    case ets:lookup(circuit_breaker, state) of
        [{state, open}] ->
            open;
        _ ->
            closed
    end.

increment_failure() ->
    case ets:lookup(circuit_breaker, failures) of
        [{failures, Count}] when Count >= ?THRESHOLD ->
            ets:insert(circuit_breaker, {state, open}),
            timer:apply_after(?TIMEOUT, ?MODULE, reset, []);
        [{failures, Count}] ->
            ets:insert(circuit_breaker, {failures, Count + 1});
        [] ->
            ets:insert(circuit_breaker, {failures, 1})
    end.

reset() ->
    ets:insert(circuit_breaker, {state, closed}),
    ets:insert(circuit_breaker, {failures, 0}).
```

In this example, we use ETS (Erlang Term Storage) to store the state of the circuit breaker. The circuit opens after a specified number of failures and resets after a timeout.

### Testing Against External Dependencies

Testing against external dependencies is crucial to ensure that your application behaves correctly when interacting with external services. Here are some strategies for testing against external APIs.

#### Mocking External Services

Mocking external services allows you to simulate API responses without making actual network requests. This can be useful for testing error handling and edge cases.

Here's an example of mocking an external service:

```erlang
-module(mock_client).
-export([fetch_data/1]).

fetch_data(Url) ->
    % Simulate a successful response
    Response = case Url of
        "https://api.example.com/data" ->
            {ok, {200, [], "{\"data\": \"mocked\"}"}};
        _ ->
            {error, not_found}
    end,
    handle_response(Response).

handle_response({ok, {_, _, Body}}) ->
    io:format("Mocked Response: ~s~n", [Body]);
handle_response({error, Reason}) ->
    io:format("Mocked Error: ~p~n", [Reason]).
```

In this example, we simulate responses based on the URL, allowing us to test different scenarios.

#### Integration Testing

Integration testing involves testing the interaction between your application and external services. This can be done by setting up a test environment with access to the actual APIs.

Here's an example of an integration test:

```erlang
-module(integration_test).
-export([test_fetch_data/0]).

test_fetch_data() ->
    % Set up the test environment
    Url = "https://api.example.com/data",
    % Call the function to be tested
    Response = api_client:fetch_data(Url),
    % Assert the expected outcome
    case Response of
        {ok, _} ->
            io:format("Integration test passed~n");
        {error, _} ->
            io:format("Integration test failed~n")
    end.
```

In this example, we call the function to be tested and assert the expected outcome based on the actual API response.

### Summary

Communicating with external services and APIs is a critical aspect of modern application development. In this section, we explored strategies for consuming REST, SOAP, and GraphQL APIs in Erlang. We discussed authentication mechanisms, rate limiting, API versioning, error handling, and testing. By following these guidelines, you can build robust and reliable applications that integrate seamlessly with external services.

Remember, this is just the beginning. As you progress, you'll build more complex and interactive applications. Keep experimenting, stay curious, and enjoy the journey!

## Quiz: Communicating with External Services and APIs

{{< quizdown >}}

### What is the primary architectural style used by REST APIs?

- [x] Representational State Transfer
- [ ] Simple Object Access Protocol
- [ ] Graph Query Language
- [ ] Remote Procedure Call

> **Explanation:** REST stands for Representational State Transfer, which is an architectural style for designing networked applications.

### Which Erlang library can be used to make HTTP requests?

- [x] httpc
- [ ] erlsom
- [ ] jsx
- [ ] jiffy

> **Explanation:** `httpc` is part of the `inets` application and can be used to make HTTP requests in Erlang.

### What format does SOAP use for its message structure?

- [ ] JSON
- [x] XML
- [ ] YAML
- [ ] CSV

> **Explanation:** SOAP uses XML for its message format, which allows for structured information exchange.

### How can you specify an API version in a request?

- [x] By setting the API version in the headers
- [ ] By including the version in the URL path
- [ ] By using a query parameter
- [ ] By sending a separate version request

> **Explanation:** API versioning can be specified in the headers, such as using the `Accept` header to request a specific version.

### What is a common authentication mechanism for APIs?

- [x] OAuth
- [ ] FTP
- [ ] SMTP
- [ ] DNS

> **Explanation:** OAuth is a common authentication mechanism used to obtain access tokens for secure API access.

### What is the purpose of rate limiting in APIs?

- [x] To control the number of requests a client can make
- [ ] To increase the speed of requests
- [ ] To encrypt the data being sent
- [ ] To cache responses

> **Explanation:** Rate limiting is used to control the number of requests a client can make in a given time period to prevent abuse.

### How can network errors be handled gracefully?

- [x] By retrying requests after a delay
- [ ] By ignoring the errors
- [ ] By terminating the application
- [ ] By logging the errors only

> **Explanation:** Retrying requests after a delay is a common strategy for handling transient network errors gracefully.

### What is the circuit breaker pattern used for?

- [x] To detect failures and prevent them from recurring
- [ ] To encrypt data
- [ ] To cache responses
- [ ] To increase request speed

> **Explanation:** The circuit breaker pattern is used to detect failures and encapsulate the logic of preventing a failure from constantly recurring.

### What is the benefit of mocking external services in tests?

- [x] To simulate API responses without making actual network requests
- [ ] To increase the speed of tests
- [ ] To encrypt test data
- [ ] To cache test responses

> **Explanation:** Mocking external services allows you to simulate API responses without making actual network requests, which is useful for testing error handling and edge cases.

### True or False: GraphQL APIs expose multiple endpoints for different operations.

- [ ] True
- [x] False

> **Explanation:** GraphQL APIs expose a single endpoint and allow clients to specify exactly what data they need, unlike REST APIs which expose multiple endpoints.

{{< /quizdown >}}
