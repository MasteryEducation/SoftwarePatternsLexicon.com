---
canonical: "https://softwarepatternslexicon.com/patterns-haskell/14/13"
title: "Benchmarking and Performance Testing with Criterion in Haskell"
description: "Master the art of benchmarking and performance testing in Haskell using the Criterion library. Learn how to measure, analyze, and optimize your code for superior performance."
linkTitle: "14.13 Benchmarking and Performance Testing with Criterion"
categories:
- Haskell
- Performance
- Benchmarking
tags:
- Criterion
- Haskell
- Performance Testing
- Benchmarking
- Optimization
date: 2024-11-23
type: docs
nav_weight: 153000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 14.13 Benchmarking and Performance Testing with Criterion

In the world of software development, performance is a critical aspect that can make or break an application. As expert software engineers and architects, understanding how to measure and optimize performance is essential. In Haskell, the Criterion library is a powerful tool for benchmarking and performance testing. This section will guide you through the process of using Criterion to analyze and enhance the performance of your Haskell applications.

### Introduction to Criterion

**Criterion** is a Haskell library designed for robust and reliable benchmarking. It provides a framework for measuring and analyzing the performance of Haskell code, allowing developers to identify bottlenecks and optimize their applications effectively.

#### Key Features of Criterion

- **Statistical Analysis**: Criterion uses statistical techniques to provide accurate and reliable performance measurements.
- **Ease of Use**: With a simple API, Criterion makes it easy to set up and run benchmarks.
- **Comprehensive Reports**: It generates detailed reports, including mean execution time, standard deviation, and more.
- **Comparison Capabilities**: Criterion allows you to compare the performance of different code versions or algorithms.

#### Why Use Criterion?

Criterion is particularly useful for Haskell developers due to its ability to handle the unique challenges of benchmarking in a lazy, functional language. It helps in understanding the performance characteristics of Haskell code, which can often be non-intuitive due to lazy evaluation and other functional paradigms.

### Setting Up Criterion

Before diving into benchmarking, you need to set up Criterion in your Haskell project. Follow these steps to get started:

1. **Install Criterion**: Add Criterion to your project's dependencies. If you're using Stack, add it to your `stack.yaml` file:

   ```yaml
   extra-deps:
     - criterion-1.5.10.0
   ```

   For Cabal, add it to your `.cabal` file:

   ```cabal
   build-depends: base >=4.7 && <5, criterion >=1.5.10.0
   ```

2. **Import Criterion**: In your Haskell source file, import the Criterion library:

   ```haskell
   import Criterion.Main
   ```

3. **Set Up a Benchmark**: Define a benchmark using Criterion's `defaultMain` function. Here's a simple example:

   ```haskell
   import Criterion.Main

   main :: IO ()
   main = defaultMain [
       bgroup "Sorting Algorithms" [
         bench "quicksort" $ nf quicksort [1..1000],
         bench "mergesort" $ nf mergesort [1..1000]
       ]
     ]

   quicksort :: Ord a => [a] -> [a]
   quicksort [] = []
   quicksort (x:xs) = quicksort [y | y <- xs, y < x] ++ [x] ++ quicksort [y | y <- xs, y >= x]

   mergesort :: Ord a => [a] -> [a]
   mergesort = undefined -- Implement mergesort here
   ```

   In this example, we're benchmarking two sorting algorithms: `quicksort` and `mergesort`. The `nf` function is used to evaluate the functions to normal form, ensuring that the entire computation is performed during benchmarking.

### Understanding Criterion's Output

When you run a Criterion benchmark, it produces detailed output that includes:

- **Mean Execution Time**: The average time taken for the benchmark to complete.
- **Standard Deviation**: A measure of the variability in execution time.
- **Confidence Intervals**: Statistical confidence intervals for the mean execution time.
- **Garbage Collection Statistics**: Information about garbage collection activity during the benchmark.

Here's an example of Criterion's output:

```
benchmarking Sorting Algorithms/quicksort
time                 1.234 ms   (1.200 ms .. 1.270 ms)
                     0.997 R²   (0.995 R² .. 0.999 R²)
mean                 1.250 ms   (1.230 ms .. 1.270 ms)
std dev              50.0 μs    (40.0 μs .. 60.0 μs)
variance introduced by outliers: 10% (moderately inflated)
```

### Analyzing Performance with Criterion

To effectively use Criterion for performance analysis, follow these steps:

1. **Identify Bottlenecks**: Use Criterion to measure different parts of your code and identify areas that are slower than expected.

2. **Compare Implementations**: Benchmark different implementations of the same functionality to determine which is more efficient.

3. **Optimize Code**: Use the insights gained from Criterion to optimize your code. This might involve algorithmic improvements, data structure changes, or other optimizations.

4. **Re-benchmark**: After making changes, re-run your benchmarks to ensure that your optimizations have the desired effect.

### Advanced Criterion Features

Criterion offers several advanced features that can enhance your benchmarking efforts:

#### Benchmark Groups

You can organize benchmarks into groups using `bgroup`. This is useful for comparing related benchmarks:

```haskell
bgroup "Sorting Algorithms" [
  bench "quicksort" $ nf quicksort [1..1000],
  bench "mergesort" $ nf mergesort [1..1000]
]
```

#### Custom Benchmarking Functions

Criterion allows you to define custom benchmarking functions using `whnf`, `nf`, and `whnfIO`:

- **`whnf`**: Evaluates a function to weak head normal form.
- **`nf`**: Evaluates a function to normal form.
- **`whnfIO`**: Evaluates an IO action to weak head normal form.

#### Environment Setup

You can set up an environment for your benchmarks using `env`. This is useful for preparing data or resources needed for benchmarking:

```haskell
env (return [1..1000]) $ \xs ->
  bgroup "Sorting Algorithms" [
    bench "quicksort" $ nf quicksort xs,
    bench "mergesort" $ nf mergesort xs
  ]
```

### Visualizing Benchmark Results

Criterion can generate graphical reports of benchmark results, which can be useful for visual analysis. To generate a report, run your benchmark with the `--output` option:

```bash
./your-benchmark --output report.html
```

This will create an HTML report with graphs and detailed statistics.

### Try It Yourself

Now that you have a solid understanding of Criterion, it's time to experiment. Try modifying the sorting algorithms or adding new ones to the benchmark. Observe how changes affect performance and use Criterion's output to guide your optimizations.

### Knowledge Check

- **What is the primary purpose of the Criterion library?**
- **How does Criterion handle lazy evaluation in Haskell?**
- **What are the differences between `whnf` and `nf` in Criterion?**
- **How can you use Criterion to compare different implementations of a function?**

### Conclusion

Benchmarking and performance testing are crucial skills for expert Haskell developers. By mastering Criterion, you can ensure that your applications are not only correct but also efficient and performant. Remember, performance optimization is an ongoing process, and Criterion is a valuable tool in your arsenal.

### Further Reading

For more information on Criterion and performance testing in Haskell, check out the following resources:

- [Criterion on Hackage](https://hackage.haskell.org/package/criterion)
- [Haskell Performance Tips](https://wiki.haskell.org/Performance)
- [Real World Haskell](http://book.realworldhaskell.org/)

## Quiz: Benchmarking and Performance Testing with Criterion

{{< quizdown >}}

### What is the primary purpose of the Criterion library?

- [x] To measure and analyze the performance of Haskell code
- [ ] To provide a GUI for Haskell applications
- [ ] To compile Haskell code into machine code
- [ ] To manage Haskell package dependencies

> **Explanation:** Criterion is specifically designed for benchmarking and performance testing in Haskell.

### How does Criterion handle lazy evaluation in Haskell?

- [x] By using functions like `nf` to evaluate expressions to normal form
- [ ] By automatically forcing all evaluations
- [ ] By ignoring lazy evaluation entirely
- [ ] By converting lazy code to strict code

> **Explanation:** Criterion uses functions like `nf` to ensure that expressions are fully evaluated during benchmarking.

### What is the difference between `whnf` and `nf` in Criterion?

- [x] `whnf` evaluates to weak head normal form, while `nf` evaluates to normal form
- [ ] `whnf` is faster than `nf`
- [ ] `nf` is used for IO actions, while `whnf` is not
- [ ] There is no difference

> **Explanation:** `whnf` evaluates only the outermost constructor, while `nf` evaluates the entire expression.

### How can you use Criterion to compare different implementations of a function?

- [x] By setting up benchmarks for each implementation and comparing their results
- [ ] By using a built-in comparison function
- [ ] By manually timing each implementation
- [ ] By using a third-party tool

> **Explanation:** Criterion allows you to set up benchmarks for different implementations and compare their performance metrics.

### Which function would you use to evaluate an IO action in Criterion?

- [x] `whnfIO`
- [ ] `nf`
- [ ] `whnf`
- [ ] `nfIO`

> **Explanation:** `whnfIO` is used to evaluate IO actions to weak head normal form in Criterion.

### What kind of reports can Criterion generate?

- [x] HTML reports with graphs and statistics
- [ ] PDF reports
- [ ] Plain text reports only
- [ ] Audio reports

> **Explanation:** Criterion can generate detailed HTML reports that include graphs and statistical analysis.

### What is the significance of the standard deviation in Criterion's output?

- [x] It measures the variability in execution time
- [ ] It indicates the average execution time
- [ ] It shows the maximum execution time
- [ ] It is not significant

> **Explanation:** The standard deviation provides insight into the consistency of the benchmark results.

### How can you set up an environment for benchmarks in Criterion?

- [x] By using the `env` function
- [ ] By using the `setup` function
- [ ] By manually preparing the environment
- [ ] By using a configuration file

> **Explanation:** The `env` function in Criterion allows you to set up an environment for your benchmarks.

### What does the `bgroup` function do in Criterion?

- [x] It organizes benchmarks into groups
- [ ] It benchmarks a single function
- [ ] It generates reports
- [ ] It compiles the code

> **Explanation:** `bgroup` is used to group related benchmarks together in Criterion.

### True or False: Criterion can only be used for benchmarking pure functions.

- [ ] True
- [x] False

> **Explanation:** Criterion can benchmark both pure functions and IO actions using appropriate functions like `whnfIO`.

{{< /quizdown >}}

Remember, this is just the beginning. As you progress, you'll build more complex and interactive applications. Keep experimenting, stay curious, and enjoy the journey!
