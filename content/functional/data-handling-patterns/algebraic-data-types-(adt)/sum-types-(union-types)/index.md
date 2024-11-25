---
linkTitle: "Sum Types"
title: "Sum Types (Union Types): Types defined as a choice between multiple possible values"
description: "An exploration of Sum Types, also known as Union Types, in functional programming, their definition, usage, and benefits. We'll look into how they enable alternatives representation within types, enhancing type safety and expressiveness in code."
categories:
- FunctionalProgramming
- DesignPatterns
tags:
- SumTypes
- UnionTypes
- TypeSafety
- FunctionalProgramming
- PatternMatching
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/data-handling-patterns/algebraic-data-types-(adt)/sum-types-(union-types)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to Sum Types

In functional programming, Sum Types, also referred to as Union Types, are a way to define a type that can be one of several distinct types. This concept enables you to create data structures that can store multiple different kinds of values but only one kind at a time. Sum Types enhance type safety, make code more expressive, and facilitate pattern matching.

### Definition

A **Sum Type** is a composite type formed by combining multiple types into a single new type. The resulting type is valid if it matches any one of the constituent types. In other words, a Sum Type represents a choice from a fixed set of possibilities.

In algebraic data type (ADT) notation, a Sum Type can be defined as follows:

```scala
sealed trait Payment
case class CreditCard(number: String, expiry: String) extends Payment
case class PayPal(accountEmail: String) extends Payment
case class BankTransfer(accountNumber: String) extends Payment
```

Here, `Payment` is a Sum Type that could be a `CreditCard`, `PayPal`, or `BankTransfer`.

### Applications

Sum Types are fundamental in defining scenarios where a variable can assume one out of several types, making them very useful for:

- **Error handling:** Represent different kinds of errors.
- **Abstract syntax trees (ASTs):** Different types of expressions and statements.
- **Complex input handling:** Different types of user input or configurations.

### Algebraic Data Types

Sum Types are commonly used in the context of Algebraic Data Types (ADTs). ADTs are derived from Algebra and consist of Sum Types and Product Types. Here's a quick differentiation:

- **Sum Types** represent a choice between options (`either A or B`).
- **Product Types** represent combinations of values (`both A and B`).

### Pattern Matching

Pattern matching is a feature that complements Sum Types beautifully:

```scala
def processPayment(payment: Payment): String = payment match {
  case CreditCard(number, expiry) => s"Processing credit card: number=$number, expiry=$expiry"
  case PayPal(accountEmail) => s"Processing PayPal payment for account: $accountEmail"
  case BankTransfer(accountNumber) => s"Processing bank transfer to account: $accountNumber"
}
```

Pattern matching ensures all potential cases are covered, enhancing code safety and readability.

## Related Design Patterns

1. **Option Type or Maybe Monad:** Represents a value that might be present (`Some`) or absent (`None`/`Nothing`), commonly used for optional values instead of `null` references.
   
   ```scala
   case class User(name: String, email: Option[String])
   ```

2. **Product Types:** Opposite of Sum Types, Product Types combine multiple values into one type. For example, tuples and case classes.

   ```scala
   case class Point(x: Int, y: Int)
   ```

3. **Visitor Pattern:** Often used with sum types, the visitor pattern separates algorithms from the object structure, encouraging scalable and maintainable code.

   ```scala
   trait PaymentVisitor[T] {
     def visit(creditCard: CreditCard): T
     def visit(payPal: PayPal): T
     def visit(bankTransfer: BankTransfer): T
   }
   ```

## Additional Resources

- **Books:**
  - "Functional Programming in Scala" by Paul Chiusano and Rúnar Bjarnason
  - "Programming in Haskell" by Graham Hutton

- **Online Courses:**
  - "Functional Programming Principles in Scala" on Coursera (offered by École Polytechnique Fédérale de Lausanne)
  - "Introduction to Functional Programming" on edX (offered by TU Delft)

- **Further Reading:**
  - [Algebraic Data Types in Functional Programming](https://en.wikipedia.org/wiki/Algebraic_data_type)
  - [Pattern Matching in Haskell](https://www.haskell.org/tutorial/patterns.html)

## Summary

Sum Types, or Union Types, are a powerful construct in functional programming that represents a choice between types. They enhance type safety, facilitate expressive code through pattern matching, and play a crucial role in algebraic data types. Essential in error handling, user input variations, and representing complex domain models, Sum Types are a fundamental tool for any functional programmer's toolkit.
