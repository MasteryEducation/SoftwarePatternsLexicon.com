---
canonical: "https://softwarepatternslexicon.com/patterns-java/8/11/4"
title: "Template Method Pattern Use Cases and Examples"
description: "Explore practical applications of the Template Method pattern in Java, including algorithms with varying steps, data processing pipelines, and more."
linkTitle: "8.11.4 Use Cases and Examples"
tags:
- "Java"
- "Design Patterns"
- "Template Method"
- "Behavioral Patterns"
- "Algorithms"
- "Data Processing"
- "Standardization"
- "Software Architecture"
date: 2024-11-25
type: docs
nav_weight: 91400
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 8.11.4 Use Cases and Examples

The Template Method pattern is a behavioral design pattern that defines the skeleton of an algorithm in a method, deferring some steps to subclasses. This pattern lets subclasses redefine certain steps of an algorithm without changing its structure. In this section, we will delve into practical applications of the Template Method pattern, exploring its use in various domains such as game development, report generation, and data processing pipelines. We will also discuss its role in enforcing policy and standardization, along with potential issues like rigidity and subclass proliferation.

### Implementing Algorithms with Varying Steps

One of the primary use cases of the Template Method pattern is in implementing algorithms that have a fixed sequence of steps, but where some of these steps can vary. This is particularly useful in scenarios like game loops or report generation.

#### Game Development

In game development, the game loop is a critical component that manages the flow of the game. The Template Method pattern can be used to define the structure of the game loop, allowing different games to implement specific behaviors for certain steps.

```java
abstract class Game {
    // Template method
    public final void play() {
        initialize();
        startPlay();
        endPlay();
    }

    abstract void initialize();
    abstract void startPlay();
    abstract void endPlay();
}

class Chess extends Game {
    @Override
    void initialize() {
        System.out.println("Chess Game Initialized! Start playing.");
    }

    @Override
    void startPlay() {
        System.out.println("Game Started. Welcome to Chess.");
    }

    @Override
    void endPlay() {
        System.out.println("Game Finished!");
    }
}

class Soccer extends Game {
    @Override
    void initialize() {
        System.out.println("Soccer Game Initialized! Start playing.");
    }

    @Override
    void startPlay() {
        System.out.println("Game Started. Welcome to Soccer.");
    }

    @Override
    void endPlay() {
        System.out.println("Game Finished!");
    }
}

// Client code
public class TemplateMethodPatternDemo {
    public static void main(String[] args) {
        Game game = new Chess();
        game.play();
        System.out.println();
        game = new Soccer();
        game.play();
    }
}
```

In this example, the `Game` class defines the template method `play()`, which outlines the steps of the game loop. The `Chess` and `Soccer` classes provide specific implementations for these steps.

#### Report Generation

Report generation often involves a series of steps such as data retrieval, data processing, and report formatting. The Template Method pattern can be used to define the overall structure of the report generation process, while allowing subclasses to customize specific steps.

```java
abstract class ReportGenerator {
    // Template method
    public final void generateReport() {
        fetchData();
        processData();
        formatReport();
    }

    abstract void fetchData();
    abstract void processData();
    abstract void formatReport();
}

class PDFReportGenerator extends ReportGenerator {
    @Override
    void fetchData() {
        System.out.println("Fetching data for PDF report.");
    }

    @Override
    void processData() {
        System.out.println("Processing data for PDF report.");
    }

    @Override
    void formatReport() {
        System.out.println("Formatting PDF report.");
    }
}

class HTMLReportGenerator extends ReportGenerator {
    @Override
    void fetchData() {
        System.out.println("Fetching data for HTML report.");
    }

    @Override
    void processData() {
        System.out.println("Processing data for HTML report.");
    }

    @Override
    void formatReport() {
        System.out.println("Formatting HTML report.");
    }
}

// Client code
public class ReportGeneratorDemo {
    public static void main(String[] args) {
        ReportGenerator pdfReport = new PDFReportGenerator();
        pdfReport.generateReport();
        System.out.println();
        ReportGenerator htmlReport = new HTMLReportGenerator();
        htmlReport.generateReport();
    }
}
```

In this scenario, the `ReportGenerator` class defines the template method `generateReport()`, which outlines the steps for generating a report. The `PDFReportGenerator` and `HTMLReportGenerator` classes provide specific implementations for these steps.

### Data Processing Pipelines

Data processing pipelines often involve a series of steps such as data extraction, transformation, and loading (ETL). The Template Method pattern can be used to define the structure of the pipeline, allowing different implementations for specific steps.

```java
abstract class DataPipeline {
    // Template method
    public final void executePipeline() {
        extractData();
        transformData();
        loadData();
    }

    abstract void extractData();
    abstract void transformData();
    abstract void loadData();
}

class CSVDataPipeline extends DataPipeline {
    @Override
    void extractData() {
        System.out.println("Extracting data from CSV file.");
    }

    @Override
    void transformData() {
        System.out.println("Transforming CSV data.");
    }

    @Override
    void loadData() {
        System.out.println("Loading data into database.");
    }
}

class JSONDataPipeline extends DataPipeline {
    @Override
    void extractData() {
        System.out.println("Extracting data from JSON file.");
    }

    @Override
    void transformData() {
        System.out.println("Transforming JSON data.");
    }

    @Override
    void loadData() {
        System.out.println("Loading data into database.");
    }
}

// Client code
public class DataPipelineDemo {
    public static void main(String[] args) {
        DataPipeline csvPipeline = new CSVDataPipeline();
        csvPipeline.executePipeline();
        System.out.println();
        DataPipeline jsonPipeline = new JSONDataPipeline();
        jsonPipeline.executePipeline();
    }
}
```

In this example, the `DataPipeline` class defines the template method `executePipeline()`, which outlines the steps for processing data. The `CSVDataPipeline` and `JSONDataPipeline` classes provide specific implementations for these steps.

### Enforcing Policy and Standardization

The Template Method pattern is also useful for enforcing policy and standardization across different implementations. By defining a template method, you can ensure that certain steps are always executed in a specific order, promoting consistency and adherence to best practices.

#### Example: Document Approval Process

Consider a document approval process where certain steps must always be followed, such as review, approval, and archiving. The Template Method pattern can enforce this process, while allowing customization of specific steps.

```java
abstract class DocumentApproval {
    // Template method
    public final void approveDocument() {
        reviewDocument();
        approve();
        archiveDocument();
    }

    abstract void reviewDocument();
    abstract void approve();
    abstract void archiveDocument();
}

class LegalDocumentApproval extends DocumentApproval {
    @Override
    void reviewDocument() {
        System.out.println("Reviewing legal document.");
    }

    @Override
    void approve() {
        System.out.println("Approving legal document.");
    }

    @Override
    void archiveDocument() {
        System.out.println("Archiving legal document.");
    }
}

class FinancialDocumentApproval extends DocumentApproval {
    @Override
    void reviewDocument() {
        System.out.println("Reviewing financial document.");
    }

    @Override
    void approve() {
        System.out.println("Approving financial document.");
    }

    @Override
    void archiveDocument() {
        System.out.println("Archiving financial document.");
    }
}

// Client code
public class DocumentApprovalDemo {
    public static void main(String[] args) {
        DocumentApproval legalApproval = new LegalDocumentApproval();
        legalApproval.approveDocument();
        System.out.println();
        DocumentApproval financialApproval = new FinancialDocumentApproval();
        financialApproval.approveDocument();
    }
}
```

In this scenario, the `DocumentApproval` class defines the template method `approveDocument()`, which outlines the steps for approving a document. The `LegalDocumentApproval` and `FinancialDocumentApproval` classes provide specific implementations for these steps.

### Potential Issues

While the Template Method pattern offers numerous benefits, it also has potential drawbacks that developers should be aware of.

#### Rigidity

The Template Method pattern can introduce rigidity into the codebase, as the algorithm's structure is fixed in the superclass. This can make it difficult to modify the algorithm's sequence of steps without altering the superclass, potentially affecting all subclasses.

#### Subclass Proliferation

The pattern can lead to subclass proliferation, as each variation of the algorithm requires a new subclass. This can result in a large number of subclasses, making the codebase more complex and harder to maintain.

### Conclusion

The Template Method pattern is a powerful tool for defining the structure of an algorithm while allowing customization of specific steps. It is particularly useful in scenarios such as game development, report generation, and data processing pipelines. By enforcing policy and standardization, the pattern promotes consistency and adherence to best practices. However, developers should be mindful of potential issues such as rigidity and subclass proliferation. By understanding these trade-offs, developers can effectively leverage the Template Method pattern to create robust and maintainable software systems.

### Encouragement for Experimentation

To deepen your understanding of the Template Method pattern, try modifying the code examples provided in this section. Experiment with adding new steps to the algorithms or creating additional subclasses to handle different scenarios. Consider how you might apply the Template Method pattern to your own projects, and reflect on the benefits and challenges it presents.

### Key Takeaways

- The Template Method pattern defines the skeleton of an algorithm, allowing subclasses to customize specific steps.
- It is useful in scenarios like game development, report generation, and data processing pipelines.
- The pattern enforces policy and standardization, promoting consistency and adherence to best practices.
- Potential issues include rigidity and subclass proliferation, which can impact maintainability.
- Experimentation and reflection can help deepen understanding and application of the pattern.

### References and Further Reading

- [Java Documentation](https://docs.oracle.com/en/java/)
- [Design Patterns: Elements of Reusable Object-Oriented Software](https://en.wikipedia.org/wiki/Design_Patterns)
- [Cloud Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/)

## Test Your Knowledge: Template Method Pattern Quiz

{{< quizdown >}}

### What is the primary purpose of the Template Method pattern?

- [x] To define the skeleton of an algorithm, allowing subclasses to customize specific steps.
- [ ] To create a single instance of a class.
- [ ] To provide a way to access the elements of an aggregate object sequentially.
- [ ] To encapsulate a request as an object.

> **Explanation:** The Template Method pattern defines the skeleton of an algorithm, allowing subclasses to customize specific steps without altering the algorithm's structure.

### In which scenarios is the Template Method pattern particularly useful?

- [x] Game development
- [x] Report generation
- [ ] Singleton implementation
- [ ] Iterator pattern

> **Explanation:** The Template Method pattern is useful in scenarios like game development and report generation, where the structure of an algorithm is fixed but certain steps can vary.

### What is a potential drawback of using the Template Method pattern?

- [x] Rigidity
- [ ] Increased flexibility
- [ ] Simplified codebase
- [ ] Enhanced performance

> **Explanation:** The Template Method pattern can introduce rigidity, as the algorithm's structure is fixed in the superclass, making it difficult to modify without affecting all subclasses.

### How does the Template Method pattern promote standardization?

- [x] By enforcing a fixed sequence of steps in an algorithm
- [ ] By allowing any sequence of steps
- [ ] By eliminating the need for subclasses
- [ ] By providing multiple instances of a class

> **Explanation:** The Template Method pattern promotes standardization by enforcing a fixed sequence of steps in an algorithm, ensuring consistency across different implementations.

### What is a common issue that can arise from subclass proliferation in the Template Method pattern?

- [x] Increased complexity and maintenance difficulty
- [ ] Simplified codebase
- [ ] Reduced number of classes
- [ ] Enhanced performance

> **Explanation:** Subclass proliferation can lead to increased complexity and maintenance difficulty, as each variation of the algorithm requires a new subclass.

### Which of the following is a benefit of using the Template Method pattern?

- [x] Consistency in algorithm implementation
- [ ] Unlimited flexibility in algorithm steps
- [ ] Elimination of subclasses
- [ ] Enhanced performance

> **Explanation:** The Template Method pattern provides consistency in algorithm implementation by defining a fixed sequence of steps, while allowing customization of specific steps.

### How can developers mitigate the rigidity introduced by the Template Method pattern?

- [x] By carefully designing the superclass and considering future changes
- [ ] By avoiding the use of the pattern altogether
- [ ] By creating more subclasses
- [ ] By using the Singleton pattern

> **Explanation:** Developers can mitigate rigidity by carefully designing the superclass and considering future changes, ensuring that the algorithm's structure is flexible enough to accommodate modifications.

### What role does the Template Method pattern play in data processing pipelines?

- [x] It defines the structure of the pipeline, allowing customization of specific steps.
- [ ] It eliminates the need for data transformation.
- [ ] It provides a single instance of the pipeline.
- [ ] It encapsulates data extraction as an object.

> **Explanation:** The Template Method pattern defines the structure of a data processing pipeline, allowing customization of specific steps such as data extraction, transformation, and loading.

### How does the Template Method pattern differ from the Strategy pattern?

- [x] The Template Method pattern defines a fixed sequence of steps, while the Strategy pattern allows for interchangeable algorithms.
- [ ] The Template Method pattern provides multiple instances, while the Strategy pattern provides a single instance.
- [ ] The Template Method pattern is used for iteration, while the Strategy pattern is used for encapsulation.
- [ ] The Template Method pattern is used for object creation, while the Strategy pattern is used for object destruction.

> **Explanation:** The Template Method pattern defines a fixed sequence of steps, allowing customization of specific steps, while the Strategy pattern allows for interchangeable algorithms by encapsulating them as objects.

### True or False: The Template Method pattern can be used to enforce policy and standardization across different implementations.

- [x] True
- [ ] False

> **Explanation:** True. The Template Method pattern can enforce policy and standardization by defining a fixed sequence of steps in an algorithm, ensuring consistency across different implementations.

{{< /quizdown >}}
