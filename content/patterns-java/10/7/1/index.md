---
canonical: "https://softwarepatternslexicon.com/patterns-java/10/7/1"
title: "RecursiveTask and RecursiveAction in Java Fork/Join Framework"
description: "Explore the use of RecursiveTask and RecursiveAction in Java's Fork/Join framework for efficient parallel computation."
linkTitle: "10.7.1 RecursiveTask and RecursiveAction"
tags:
- "Java"
- "Concurrency"
- "Parallelism"
- "Fork/Join Framework"
- "RecursiveTask"
- "RecursiveAction"
- "Multithreading"
- "Divide and Conquer"
date: 2024-11-25
type: docs
nav_weight: 107100
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 10.7.1 RecursiveTask and RecursiveAction

### Introduction to the Fork/Join Framework

The Fork/Join framework is a powerful tool in Java designed to facilitate parallel computation by breaking down tasks into smaller subtasks, processing them concurrently, and then combining the results. This framework is particularly well-suited for divide-and-conquer algorithms, where a problem is recursively divided into subproblems until they become simple enough to solve directly. Introduced in Java 7, the Fork/Join framework leverages multiple processors, making it ideal for enhancing performance in multi-core systems.

### Understanding RecursiveTask and RecursiveAction

Within the Fork/Join framework, two key classes are used to represent tasks: `RecursiveTask` and `RecursiveAction`. Both are abstract classes that extend `ForkJoinTask`, but they serve different purposes:

- **RecursiveTask**: This class is used when a task returns a result. It is suitable for computations where you need to aggregate results from subtasks.
- **RecursiveAction**: This class is used when a task does not return a result. It is appropriate for operations that modify data in place or perform side effects without needing to return a value.

### RecursiveTask vs. RecursiveAction

The primary distinction between `RecursiveTask` and `RecursiveAction` lies in their return types. `RecursiveTask` is parameterized with a type that represents the result of the computation, while `RecursiveAction` is parameterized with `Void`, indicating no result is returned.

#### RecursiveTask Example: Parallel Array Sum

Let's explore an example of using `RecursiveTask` to compute the sum of an array in parallel. This example demonstrates how to divide the array into smaller segments, compute the sum of each segment concurrently, and then combine the results.

```java
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class ParallelArraySum extends RecursiveTask<Long> {
    private static final int THRESHOLD = 1000;
    private final int[] array;
    private final int start;
    private final int end;

    public ParallelArraySum(int[] array, int start, int end) {
        this.array = array;
        this.start = start;
        this.end = end;
    }

    @Override
    protected Long compute() {
        if (end - start <= THRESHOLD) {
            long sum = 0;
            for (int i = start; i < end; i++) {
                sum += array[i];
            }
            return sum;
        } else {
            int mid = (start + end) / 2;
            ParallelArraySum leftTask = new ParallelArraySum(array, start, mid);
            ParallelArraySum rightTask = new ParallelArraySum(array, mid, end);

            leftTask.fork(); // asynchronously execute the left task
            long rightResult = rightTask.compute(); // compute the right task
            long leftResult = leftTask.join(); // wait for the left task to complete

            return leftResult + rightResult;
        }
    }

    public static void main(String[] args) {
        int[] array = new int[10000];
        for (int i = 0; i < array.length; i++) {
            array[i] = i;
        }

        ForkJoinPool pool = new ForkJoinPool();
        ParallelArraySum task = new ParallelArraySum(array, 0, array.length);
        long result = pool.invoke(task);

        System.out.println("Sum: " + result);
    }
}
```

**Explanation**: In this example, the array is divided into smaller segments based on a threshold. If the segment size is below the threshold, the sum is computed directly. Otherwise, the task is split into two subtasks, which are executed concurrently. The `fork()` method is used to asynchronously execute a subtask, while `join()` waits for its completion.

#### RecursiveAction Example: Parallel QuickSort

Now, let's consider an example of using `RecursiveAction` to implement a parallel version of the QuickSort algorithm. This example demonstrates how to sort an array concurrently without returning a result.

```java
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class ParallelQuickSort extends RecursiveAction {
    private static final int THRESHOLD = 1000;
    private final int[] array;
    private final int low;
    private final int high;

    public ParallelQuickSort(int[] array, int low, int high) {
        this.array = array;
        this.low = low;
        this.high = high;
    }

    @Override
    protected void compute() {
        if (high - low <= THRESHOLD) {
            quickSort(array, low, high);
        } else {
            int pivotIndex = partition(array, low, high);
            ParallelQuickSort leftTask = new ParallelQuickSort(array, low, pivotIndex - 1);
            ParallelQuickSort rightTask = new ParallelQuickSort(array, pivotIndex + 1, high);

            invokeAll(leftTask, rightTask);
        }
    }

    private void quickSort(int[] array, int low, int high) {
        if (low < high) {
            int pivotIndex = partition(array, low, high);
            quickSort(array, low, pivotIndex - 1);
            quickSort(array, pivotIndex + 1, high);
        }
    }

    private int partition(int[] array, int low, int high) {
        int pivot = array[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (array[j] <= pivot) {
                i++;
                swap(array, i, j);
            }
        }
        swap(array, i + 1, high);
        return i + 1;
    }

    private void swap(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    public static void main(String[] args) {
        int[] array = {3, 6, 8, 10, 1, 2, 1};
        ForkJoinPool pool = new ForkJoinPool();
        ParallelQuickSort task = new ParallelQuickSort(array, 0, array.length - 1);
        pool.invoke(task);

        for (int num : array) {
            System.out.print(num + " ");
        }
    }
}
```

**Explanation**: In this example, the array is sorted using the QuickSort algorithm. If the segment size is below the threshold, the array is sorted directly using a sequential QuickSort. Otherwise, the array is partitioned, and two subtasks are created to sort the partitions concurrently using `invokeAll()`.

### Best Practices for Task Splitting

Determining the optimal threshold for task splitting is crucial for maximizing performance in the Fork/Join framework. Here are some best practices:

- **Balance Task Granularity**: Ensure tasks are neither too large (leading to underutilization of processors) nor too small (causing excessive overhead from task management).
- **Experiment with Thresholds**: Test different threshold values to find the optimal balance for your specific application and hardware.
- **Consider Task Overhead**: Factor in the overhead of task creation and management when deciding on the threshold.
- **Monitor Performance**: Use profiling tools to monitor the performance and adjust the threshold as needed.

### Historical Context and Evolution

The Fork/Join framework was inspired by the divide-and-conquer paradigm, which has been a fundamental concept in computer science for decades. The framework's design was influenced by earlier parallel computing models and aimed to provide a more efficient and scalable solution for Java developers. Since its introduction in Java 7, the Fork/Join framework has evolved to incorporate new features and optimizations, making it a vital tool for modern Java applications.

### Practical Applications and Real-World Scenarios

The Fork/Join framework is widely used in scenarios where large datasets need to be processed concurrently. Examples include:

- **Data Analysis**: Processing large datasets in parallel to improve the performance of data analysis applications.
- **Image Processing**: Applying filters or transformations to images by dividing them into smaller sections and processing them concurrently.
- **Financial Modeling**: Performing complex calculations on financial data to generate reports or forecasts.

### Conclusion

The Fork/Join framework, with its `RecursiveTask` and `RecursiveAction` classes, provides a robust solution for parallel computation in Java. By leveraging this framework, developers can efficiently implement divide-and-conquer algorithms, optimize performance on multi-core systems, and handle large datasets with ease. Understanding the nuances of task splitting and threshold determination is key to harnessing the full potential of this powerful tool.

---

## Test Your Knowledge: RecursiveTask and RecursiveAction in Java Fork/Join Framework

{{< quizdown >}}

### What is the primary purpose of the Fork/Join framework in Java?

- [x] To facilitate parallel computation by dividing tasks into smaller subtasks.
- [ ] To manage database connections.
- [ ] To handle network communication.
- [ ] To improve file I/O operations.

> **Explanation:** The Fork/Join framework is designed to break down tasks into smaller subtasks, process them concurrently, and combine the results, making it ideal for parallel computation.

### Which class should be used when a task needs to return a result?

- [x] RecursiveTask
- [ ] RecursiveAction
- [ ] ForkJoinPool
- [ ] Thread

> **Explanation:** `RecursiveTask` is used when a task returns a result, whereas `RecursiveAction` is used for tasks that do not return a result.

### What method is used to asynchronously execute a subtask in the Fork/Join framework?

- [x] fork()
- [ ] join()
- [ ] invoke()
- [ ] execute()

> **Explanation:** The `fork()` method is used to asynchronously execute a subtask, allowing it to run concurrently with other tasks.

### In the Fork/Join framework, what is the role of the `join()` method?

- [x] To wait for the completion of a subtask and retrieve its result.
- [ ] To start a new thread.
- [ ] To divide a task into subtasks.
- [ ] To terminate a running task.

> **Explanation:** The `join()` method waits for the completion of a subtask and retrieves its result, ensuring that the main task can proceed with the combined results of its subtasks.

### What is a key consideration when determining the threshold for task splitting?

- [x] Balancing task granularity to avoid excessive overhead.
- [ ] Ensuring tasks are as large as possible.
- [ ] Minimizing the number of subtasks.
- [ ] Maximizing the number of threads.

> **Explanation:** Balancing task granularity is crucial to avoid excessive overhead from task management while ensuring efficient processor utilization.

### Which of the following is a real-world application of the Fork/Join framework?

- [x] Image processing
- [ ] Network routing
- [ ] File encryption
- [ ] Database indexing

> **Explanation:** The Fork/Join framework is commonly used in image processing to apply filters or transformations concurrently by dividing images into smaller sections.

### What inspired the design of the Fork/Join framework?

- [x] The divide-and-conquer paradigm
- [ ] The client-server model
- [ ] The observer pattern
- [ ] The singleton pattern

> **Explanation:** The Fork/Join framework was inspired by the divide-and-conquer paradigm, which involves breaking down problems into smaller, more manageable parts.

### Which method is used to execute multiple tasks concurrently in the Fork/Join framework?

- [x] invokeAll()
- [ ] executeAll()
- [ ] runAll()
- [ ] startAll()

> **Explanation:** The `invokeAll()` method is used to execute multiple tasks concurrently, ensuring they are processed in parallel.

### What is the default threshold value for task splitting in the provided examples?

- [x] 1000
- [ ] 500
- [ ] 2000
- [ ] 1500

> **Explanation:** In the provided examples, the default threshold value for task splitting is set to 1000, determining when tasks should be divided into subtasks.

### True or False: RecursiveAction is used when a task returns a result.

- [ ] True
- [x] False

> **Explanation:** False. `RecursiveAction` is used for tasks that do not return a result, while `RecursiveTask` is used for tasks that return a result.

{{< /quizdown >}}
