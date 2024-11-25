---
canonical: "https://softwarepatternslexicon.com/patterns-ts/14/5/3"
title: "TypeORM and Active Record Pattern in TypeScript"
description: "Explore how TypeORM implements the Active Record pattern in TypeScript, providing an ORM that allows developers to interact with databases using object-oriented principles."
linkTitle: "14.5.3 TypeORM and Active Record Pattern"
categories:
- TypeScript
- Design Patterns
- ORM
tags:
- TypeORM
- Active Record
- TypeScript
- Database
- ORM
date: 2024-11-17
type: docs
nav_weight: 14530
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 14.5.3 TypeORM and Active Record Pattern

In this section, we delve into how TypeORM implements the Active Record pattern in TypeScript, providing an ORM (Object-Relational Mapping) framework that allows developers to interact with databases using object-oriented principles. This exploration will equip you with the knowledge to effectively use TypeORM in your TypeScript applications, leveraging the Active Record pattern to streamline database interactions.

### Introduction to TypeORM

TypeORM is a powerful ORM framework that supports both TypeScript and JavaScript (ES6). It is designed to work seamlessly with a variety of database systems, including MySQL, PostgreSQL, SQLite, Microsoft SQL Server, and more. TypeORM allows developers to work with databases using object-oriented paradigms, making it easier to manage database operations within your application code.

#### Key Features of TypeORM

- **TypeScript Support**: TypeORM is built with TypeScript in mind, providing strong typing and autocompletion features.
- **Database Compatibility**: It supports a wide range of databases, allowing flexibility in choosing the right database for your application.
- **Decorators and Metadata**: TypeORM uses decorators to define entities and their relationships, making the code more readable and maintainable.
- **Active Record and Data Mapper Patterns**: TypeORM supports both Active Record and Data Mapper patterns, giving developers the flexibility to choose the best approach for their needs.

### Active Record Pattern Fundamentals

The Active Record pattern is a design pattern commonly used in ORM frameworks. It involves representing database tables as classes, where each instance of the class corresponds to a row in the table. The class itself contains methods to perform CRUD (Create, Read, Update, Delete) operations on the database.

#### Benefits of the Active Record Pattern

- **Simplicity**: The Active Record pattern simplifies database interactions by encapsulating database operations within the entity class.
- **Readability**: By using object-oriented principles, the code becomes more readable and easier to understand.
- **Rapid Development**: It speeds up development by reducing the amount of boilerplate code needed for database operations.

#### Potential Drawbacks

- **Tight Coupling**: The pattern can lead to tight coupling between the database schema and the application logic.
- **Scalability**: For complex applications with intricate business logic, the Active Record pattern might not be the best fit due to its limitations in separating concerns.

### Setting Up TypeORM

To get started with TypeORM, you need to install it and set up a basic connection to your database. Let's walk through the setup process.

#### Installing TypeORM

First, ensure you have Node.js and npm installed on your system. Then, you can install TypeORM along with a database driver of your choice. For example, to use MySQL, you would run:

```bash
npm install typeorm mysql2
```

#### Basic Configuration

Create a `ormconfig.json` file in the root of your project to define the database connection settings:

```json
{
  "type": "mysql",
  "host": "localhost",
  "port": 3306,
  "username": "test",
  "password": "test",
  "database": "test_db",
  "synchronize": true,
  "logging": false,
  "entities": [
    "src/entity/**/*.ts"
  ]
}
```

#### Connecting to the Database

In your application entry file, establish a connection to the database:

```typescript
import "reflect-metadata";
import { createConnection } from "typeorm";

createConnection().then(async connection => {
  console.log("Connected to the database!");
}).catch(error => console.log("Error: ", error));
```

### Defining Entities and Models

Entities in TypeORM are classes that represent database tables. You define them using decorators to specify the table and column mappings.

#### Creating an Entity

Here's how you can define a simple `User` entity:

```typescript
import { Entity, PrimaryGeneratedColumn, Column } from "typeorm";

@Entity()
export class User {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  firstName: string;

  @Column()
  lastName: string;

  @Column()
  age: number;
}
```

- **`@Entity()`**: Marks the class as a database entity.
- **`@PrimaryGeneratedColumn()`**: Indicates that the `id` field is a primary key and its value will be automatically generated.
- **`@Column()`**: Maps the class property to a database column.

### CRUD Operations Using Active Record

With the Active Record pattern, each entity class has built-in methods for performing CRUD operations.

#### Create Operation

To create a new user, instantiate the `User` class and call the `save` method:

```typescript
import { User } from "./entity/User";

const user = new User();
user.firstName = "John";
user.lastName = "Doe";
user.age = 25;
await user.save();
```

#### Read Operation

To retrieve users, use the `find` method:

```typescript
const users = await User.find();
console.log(users);
```

#### Update Operation

To update a user, modify the properties and call `save` again:

```typescript
const user = await User.findOne({ where: { firstName: "John" } });
if (user) {
  user.lastName = "Smith";
  await user.save();
}
```

#### Delete Operation

To delete a user, use the `remove` method:

```typescript
const user = await User.findOne({ where: { firstName: "John" } });
if (user) {
  await user.remove();
}
```

### Advanced Features

TypeORM provides advanced features to handle complex database interactions.

#### Relationships

TypeORM supports various types of relationships, such as one-to-one, one-to-many, and many-to-many.

##### One-to-One Relationship

```typescript
import { Entity, PrimaryGeneratedColumn, Column, OneToOne, JoinColumn } from "typeorm";
import { Profile } from "./Profile";

@Entity()
export class User {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  firstName: string;

  @OneToOne(() => Profile)
  @JoinColumn()
  profile: Profile;
}
```

##### One-to-Many and Many-to-One Relationship

```typescript
import { Entity, PrimaryGeneratedColumn, Column, OneToMany } from "typeorm";
import { Photo } from "./Photo";

@Entity()
export class User {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @OneToMany(() => Photo, photo => photo.user)
  photos: Photo[];
}
```

##### Many-to-Many Relationship

```typescript
import { Entity, PrimaryGeneratedColumn, Column, ManyToMany, JoinTable } from "typeorm";
import { Category } from "./Category";

@Entity()
export class Post {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  title: string;

  @ManyToMany(() => Category)
  @JoinTable()
  categories: Category[];
}
```

#### Cascade Operations

Cascade operations allow related entities to be automatically persisted or removed. Use the `cascade` option in relationships:

```typescript
@OneToMany(() => Photo, photo => photo.user, { cascade: true })
photos: Photo[];
```

#### Lazy Loading

Lazy loading defers the initialization of a relationship until it is accessed. Use `Promise` type to enable lazy loading:

```typescript
@OneToMany(() => Photo, photo => photo.user)
photos: Promise<Photo[]>;
```

#### Transactions

Transactions ensure that a series of operations are executed atomically. Use the `transaction` method:

```typescript
await connection.transaction(async transactionalEntityManager => {
  await transactionalEntityManager.save(user);
  await transactionalEntityManager.save(profile);
});
```

### Best Practices

#### Organizing Code

- **Modular Structure**: Organize your entities, repositories, and services into separate modules for better maintainability.
- **Configuration Management**: Use environment variables to manage database configurations.

#### Handling Migrations

Use TypeORM's migration tools to manage database schema changes:

```bash
typeorm migration:create -n CreateUsersTable
typeorm migration:run
```

#### Performance Optimization

- **Eager Loading**: Use eager loading to reduce the number of database queries.
- **Indexes**: Create indexes on frequently queried columns to improve performance.

#### Active Record vs. Data Mapper

While Active Record is suitable for simple applications, consider using the Data Mapper pattern for complex business logic to separate domain logic from database operations.

### Integration with Other Patterns

TypeORM can be used alongside other design patterns to enhance application architecture.

#### Repository Pattern

Use repositories to encapsulate database operations:

```typescript
const userRepository = connection.getRepository(User);
const users = await userRepository.find();
```

#### Service Layer

Implement a service layer to handle business logic:

```typescript
class UserService {
  async createUser(data: CreateUserDto): Promise<User> {
    const user = new User();
    user.firstName = data.firstName;
    user.lastName = data.lastName;
    return await user.save();
  }
}
```

### Testing with TypeORM

Testing database operations is crucial for ensuring application reliability.

#### In-Memory Databases

Use in-memory databases for fast and isolated tests:

```typescript
createConnection({
  type: "sqlite",
  database: ":memory:",
  dropSchema: true,
  entities: [User],
  synchronize: true,
  logging: false
});
```

#### Mocking

Mock database operations to test business logic without actual database interactions:

```typescript
jest.mock("typeorm", () => ({
  getRepository: jest.fn().mockReturnValue({
    find: jest.fn().mockResolvedValue([{ id: 1, name: "Test User" }])
  })
}));
```

### Conclusion

TypeORM, with its implementation of the Active Record pattern, provides a robust framework for managing database interactions in TypeScript applications. By leveraging object-oriented principles, TypeORM simplifies CRUD operations and supports advanced features like relationships, transactions, and migrations. While the Active Record pattern is suitable for many use cases, it's important to evaluate its fit for your application's complexity and consider integrating other patterns as needed.

Remember, ORM solutions like TypeORM are powerful tools that can significantly streamline database interactions, but they require careful consideration and understanding to use effectively. Keep experimenting, stay curious, and enjoy the journey of mastering TypeScript and design patterns!

## Quiz Time!

{{< quizdown >}}

### What is TypeORM?

- [x] An ORM framework that supports TypeScript and JavaScript
- [ ] A database management system
- [ ] A JavaScript library for UI development
- [ ] A CSS framework

> **Explanation:** TypeORM is an Object-Relational Mapping (ORM) framework that supports TypeScript and JavaScript, allowing developers to interact with databases using object-oriented principles.

### Which pattern does TypeORM primarily implement?

- [x] Active Record
- [ ] Singleton
- [ ] Observer
- [ ] Factory

> **Explanation:** TypeORM primarily implements the Active Record pattern, which involves representing database tables as classes with methods for CRUD operations.

### What is a potential drawback of the Active Record pattern?

- [x] Tight coupling between database schema and application logic
- [ ] Lack of support for CRUD operations
- [ ] Incompatibility with TypeScript
- [ ] Difficulty in setting up relationships

> **Explanation:** A potential drawback of the Active Record pattern is the tight coupling it creates between the database schema and application logic, which can affect scalability and flexibility.

### How do you define an entity in TypeORM?

- [x] Using decorators like `@Entity` and `@Column`
- [ ] By creating a JSON schema
- [ ] Using XML configuration files
- [ ] By writing raw SQL queries

> **Explanation:** In TypeORM, entities are defined using decorators such as `@Entity` and `@Column`, which map class properties to database columns.

### What is the purpose of the `@PrimaryGeneratedColumn` decorator?

- [x] To mark a property as a primary key with auto-generated values
- [ ] To define a foreign key relationship
- [ ] To specify a unique constraint
- [ ] To create an index on a column

> **Explanation:** The `@PrimaryGeneratedColumn` decorator in TypeORM marks a property as a primary key with values that are automatically generated by the database.

### What is lazy loading in TypeORM?

- [x] Deferring the initialization of a relationship until it is accessed
- [ ] Loading all related entities eagerly
- [ ] Automatically caching query results
- [ ] Executing queries in parallel

> **Explanation:** Lazy loading in TypeORM refers to deferring the initialization of a relationship until it is accessed, which can improve performance by reducing unnecessary data loading.

### How can you perform transactions in TypeORM?

- [x] Using the `transaction` method with a callback
- [ ] By writing raw SQL transaction queries
- [ ] Through a configuration file
- [ ] Using the `@Transaction` decorator

> **Explanation:** Transactions in TypeORM can be performed using the `transaction` method, which takes a callback function where you can execute multiple operations atomically.

### What is the advantage of using in-memory databases for testing?

- [x] Fast and isolated tests
- [ ] Persistent data storage
- [ ] Real-time data synchronization
- [ ] Enhanced security

> **Explanation:** In-memory databases are advantageous for testing because they provide fast and isolated tests, allowing for quick execution without affecting persistent data.

### When should you consider using the Data Mapper pattern over Active Record in TypeORM?

- [x] For complex applications with intricate business logic
- [ ] When working with small, simple applications
- [ ] When using a NoSQL database
- [ ] For applications that do not require database interactions

> **Explanation:** The Data Mapper pattern is more suitable for complex applications with intricate business logic, as it separates domain logic from database operations, unlike the Active Record pattern.

### True or False: TypeORM can only be used with SQL databases.

- [ ] True
- [x] False

> **Explanation:** False. TypeORM supports a variety of databases, including SQL databases like MySQL and PostgreSQL, as well as NoSQL databases like MongoDB.

{{< /quizdown >}}
