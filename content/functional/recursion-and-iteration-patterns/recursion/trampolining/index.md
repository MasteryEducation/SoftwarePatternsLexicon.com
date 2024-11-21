---
linkTitle: "Trampolining"
title: "Trampolining: Converting recursive calls into an iterative process to avoid stack overflow"
description: "Trampolining is a technique in functional programming where recursive calls are transformed into an iterative process to avoid stack overflow, enhancing robustness and enabling tail-recursive optimization."
categories:
- Functional Programming
- Design Patterns
tags:
- Trampolining
- Recursion
- Iterative Processes
- Tail-call Optimization
- Functional Programming Patterns
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/recursion-and-iteration-patterns/recursion/trampolining"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Trampolining is an essential design pattern in functional programming that addresses the common problem of stack overflow resulting from heavy recursive calls. This pattern converts recursive calls into an iterative process using an intermediate function called a "trampoline." By doing so, trampolining helps in managing and optimizing recursive functions, effectively enabling tail-recursive behavior even when the language runtime does not explicitly support tail-call optimization.

## How Trampolining Works

In essence, trampolining involves a function that repeatedly calls other functions (which ideally return their results without going deep into another function call), allowing the same mechanism to handle what traditionally requires deep recursion. This effectively restructures the recursive process into iterative loops.

### Step-by-Step Process

1. **Define base functions:** Implement the recursive functions that will eventually be called iteratively.
2. **Introduce Trampoline:** Create a trampoline function that manages these calls and ensures they execute iteratively.
3. **Execution Loop:** The trampoline function replaces deep calls with an iterative execution loop, maintaining a manageable call stack.

### Example in JavaScript

Here's a simple JavaScript example to illustrate the concept:

```javascript
function trampoline(fn) {
    let result = fn.apply(fn, Array.prototype.slice.call(arguments, 1));

    while (typeof result === 'function') {
        result = result();
    }

    return result;
}

function sum(x, y, cont = (v) => v) {
    if (y === 0) return cont(x);
    return () => sum(x + 1, y - 1, cont);
}

const result = trampoline(sum, 9999, 5000);
console.log(result);  // Outputs: 14999
```

## Related Patterns

Understanding trampolining can be easier when you are familiar with other related functional programming concepts and patterns:

- **Tail-Call Optimization (TCO):** TCO is an optimization strategy where the compiler optimizes recursive calls to avoid growing the call stack. While trampolining can be used as a workaround, languages supporting TCO directly can handle certain recursive calls efficiently without additional constructs.
  
- **Continuation-Passing Style (CPS):** CPS transforms function calls such that results are passed via higher-order functions (continuations). Trampolining and CPS often go hand-in-hand in converting deep recursive calls into an iterative style.

## Additional Resources

To deepen your understanding of trampolining and related concepts, here are some useful resources:

1. **Books:**
   - *"Functional Programming in JavaScript"* by Luis Atencio.
   - *"Structure and Interpretation of Computer Programs"* by Harold Abelson and Gerald Jay Sussman.

2. **Online Articles and Tutorials:**
   - [JavaScript Trampoline Function Explained](https://medium.com/@gmchaturvedi23/javascript-trampoline-function-d19fd989583b)
   - [Trampolining: Why and How](https://www.researchgate.net/publication/221211804_Trampolining_Why_and_How)

3. **Academic Papers:**
   - "Compiling with Continuations" by A. W. Appel.
   - "Revised Report on the Algorithmic Language Scheme" by Guiyou Qiu and Shaocheng Zhang.

## Summary

Trampolining is a crucial pattern in functional programming, enabling the conversion of recursive calls into an iterative process to avoid stack overflow. By introducing a trampoline function, recursive calls are managed iteratively, preserving system stability and enhancing performance.

Understanding related patterns such as Tail-Call Optimization and Continuation-Passing Style can broaden the application and comprehension of trampolining. With the resources provided, you can explore the depths of these functional programming concepts further, leading to a stronger grasp and better implementation strategies in your programming practices.
